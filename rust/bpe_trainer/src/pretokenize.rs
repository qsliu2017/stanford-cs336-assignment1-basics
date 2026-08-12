use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    os::unix::fs::FileExt,
};

use fancy_regex::Regex;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::BpeError;

const PAT: &'static str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

type PretokenCounts = HashMap<String, u64>;

pub(crate) fn pretokenize(
    file: File,
    special_tokens: &Vec<String>,
) -> Result<PretokenCounts, BpeError> {
    let boundaries = find_special_tokens(&file, special_tokens)?;
    let pretokens_in_range = boundaries
        .into_iter()
        .collect::<Vec<_>>()
        .par_windows(2)
        .map(|window| {
            let &[(_, start), (end, _)] = window else {
                unreachable!()
            };
            count_pretokens_in_range(&file, start, end)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(pretokens_in_range
        .into_iter()
        .fold(PretokenCounts::new(), |mut all, partial| {
            for (t, c) in partial.into_iter() {
                *all.entry(t).or_insert(0) += c;
            }
            all
        }))
}

/// Chunk the file into parts that can be counted independently.
///
/// Each nominal chunk boundary is moved forward to the end of the next
/// special token. Multiple nominal boundaries may resolve to the same byte
/// position, so fewer boundaries may be returned.
fn find_special_tokens(
    file: &File,
    special_tokens: &Vec<String>,
) -> Result<BTreeSet<(usize, usize)>, BpeError> {
    let n_chunks = rayon::max_num_threads() * 2;

    let file_size = file.metadata()?.len() as usize;

    let mut boundaries = BTreeSet::new();
    boundaries.insert((0, 0));
    boundaries.insert((file_size, file_size));

    if file_size == 0 {
        return Ok(boundaries);
    }

    let tokens = special_tokens
        .into_iter()
        .inspect(|token| {
            if token.len() > READ_BUFFER_SIZE {
                unimplemented!()
            }
        })
        .filter(|token| !token.is_empty())
        .map(String::as_bytes)
        .collect::<Vec<_>>();

    if tokens.is_empty() {
        return Ok(boundaries);
    }

    // We need NUM_CHUNKS - 1 internal boundaries. Each nominal boundary is
    // rounded forward to the end of the first special token found.
    let internal_boundaries = (1..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let search_start = file_size.saturating_mul(chunk_idx) / n_chunks;
            let search_end = file_size.saturating_mul(chunk_idx + 1) / n_chunks;
            find_next_special_token(
                &file,
                search_start,
                (search_end + READ_BUFFER_SIZE).min(file_size),
                &tokens,
            )
        })
        .collect::<Result<Vec<_>, BpeError>>()?;

    boundaries.extend(internal_boundaries.into_iter().flatten());

    Ok(boundaries)
}

const READ_BUFFER_SIZE: usize = 4096;

/// Find the byte position immediately after the first special token whose
/// start is at or after `search_start`.
fn find_next_special_token(
    file: &File,
    search_start: usize,
    search_end: usize,
    tokens: &[&[u8]],
) -> Result<Option<(usize, usize)>, BpeError> {
    if search_start >= search_end {
        return Ok(None);
    }

    // Retain enough bytes between reads to detect a token spanning two reads.
    let mut carry = Vec::<u8>::with_capacity(READ_BUFFER_SIZE * 2);
    let mut read_offset = search_start;

    while read_offset < search_end {
        let mut read_buffer = [0_u8; READ_BUFFER_SIZE];
        let bytes_read = file.read_at(&mut read_buffer, read_offset as _)?;

        if bytes_read == 0 {
            break;
        }

        let combined_start = read_offset.saturating_sub(carry.len());

        carry.extend_from_slice(&read_buffer[..bytes_read]);

        if let Some((start, end)) = tokens
            .iter()
            .map(|token| {
                find_all_iter(&carry, token).find_map(|relative_start| {
                    let absolute_start = combined_start + relative_start;
                    (absolute_start >= search_start)
                        .then(|| (absolute_start, absolute_start + token.len()))
                })
            })
            .flatten()
            .min_by_key(|&(start, _)| start)
        {
            return Ok(Some((start, end)));
        }

        if carry.len() > READ_BUFFER_SIZE {
            let keep_from = carry.len() - READ_BUFFER_SIZE;
            carry.drain(..keep_from);
        }

        read_offset += bytes_read;
    }

    Ok(None)
}

/// Return every occurrence of `needle` in `haystack`, including overlapping
/// occurrences.
fn find_all_iter<'a>(haystack: &'a [u8], needle: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
    let mut next_start = 0;

    std::iter::from_fn(move || {
        if needle.is_empty() || next_start + needle.len() > haystack.len() {
            return None;
        }

        let relative_position = haystack[next_start..]
            .windows(needle.len())
            .position(|window| window == needle)?;

        let position = next_start + relative_position;
        next_start = position + 1;

        Some(position)
    })
}

/// Incrementally pre-tokenize the file range `[start, end)`.
///
/// The scanner retains the last regex match whenever that match reaches the
/// end of the currently available input. That match may continue into the next
/// read, for example:
///
/// - `" hel"` followed by `"lo"`
/// - `"'"` followed by `"s"`
/// - whitespace followed by a non-whitespace character
fn count_pretokens_in_range(
    file: &File,
    start: usize,
    end: usize,
) -> Result<PretokenCounts, BpeError> {
    if start >= end {
        return Ok(HashMap::new());
    }

    // Compile one regex per worker. This avoids requiring the regex value to
    // be shared between threads.
    //
    // Safety: this regex can be parsed.
    let regex = Regex::new(PAT).unwrap();

    let mut counts = HashMap::new();
    let mut buffer = Vec::<u8>::with_capacity(READ_BUFFER_SIZE * 2);
    let mut read_buffer = [0_u8; READ_BUFFER_SIZE];
    let mut file_offset = start;

    while file_offset < end {
        let bytes_read = file.read_at(
            &mut read_buffer[..std::cmp::min(READ_BUFFER_SIZE, end - file_offset)],
            file_offset as _,
        )?;

        if bytes_read == 0 {
            return Err(BpeError::UnexpectedEof {
                expected_end: end,
                actual_end: file_offset,
            });
        }

        buffer.extend_from_slice(&read_buffer[..bytes_read]);
        file_offset += bytes_read;

        let range_finished = file_offset == end;

        process_available_input(&regex, &mut buffer, range_finished, &mut counts)?;
    }

    // `process_available_input` should have consumed everything at EOF.
    if !buffer.is_empty() {
        return Err(BpeError::UnconsumedInput {
            offset_start: file_offset - buffer.len(),
            offset_end: file_offset,
        });
    }

    Ok(counts)
}

/// Process the complete UTF-8 prefix currently present in `buffer`.
///
/// When `range_finished` is false, a match ending exactly at the available
/// input boundary is retained because later bytes might extend or change it.
fn process_available_input(
    regex: &Regex,
    buffer: &mut Vec<u8>,
    range_finished: bool,
    counts: &mut PretokenCounts,
) -> Result<(), BpeError> {
    let valid_utf8_len = match str::from_utf8(buffer) {
        Ok(_) => buffer.len(),

        Err(error) if error.error_len().is_none() => {
            // The buffer ends partway through a UTF-8 code point. Process only
            // the complete prefix and retain the incomplete bytes.
            error.valid_up_to()
        }

        Err(error) => {
            return Err(BpeError::InvalidUtf8 {
                offset: error.valid_up_to(),
            });
        }
    };

    if range_finished && valid_utf8_len != buffer.len() {
        return Err(BpeError::InvalidUtf8 {
            offset: valid_utf8_len,
        });
    }

    if valid_utf8_len == 0 {
        return Ok(());
    }

    let text = str::from_utf8(&buffer[..valid_utf8_len])
        .expect("valid_utf8_len was obtained from from_utf8");

    let mut cursor = 0;
    let mut drain_through = 0;

    for result in regex.find_iter(text) {
        let matched = result.unwrap();

        // PAT is intended to cover every character. Detect accidental gaps
        // rather than silently dropping bytes.
        if matched.start() != cursor {
            return Err(BpeError::UnmatchedInput { offset: cursor });
        }

        // This match might continue into the next read. Retain it in full.
        //
        // This is especially important for:
        //   \p{L}+
        //   \p{N}+
        //   punctuation runs
        //   contractions such as "'s"
        //   the whitespace negative-lookahead branch
        if !range_finished && matched.end() == text.len() {
            break;
        }

        *counts.entry(matched.as_str().to_owned()).or_insert(0) += 1;

        cursor = matched.end();
        drain_through = matched.end();
    }

    if range_finished {
        if cursor != text.len() {
            return Err(BpeError::UnmatchedInput { offset: cursor });
        }

        drain_through = valid_utf8_len;
    }

    if drain_through != 0 {
        buffer.drain(..drain_through);
    }

    Ok(())
}
