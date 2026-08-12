use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{self};
use std::ops::AddAssign;
use std::path::PathBuf;
use std::{iter, str};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

mod pretokenize;

use pretokenize::pretokenize;

#[pymodule]
fn cs336_bpe(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_train_bpe, module)?)?;
    Ok(())
}

#[derive(thiserror::Error, Debug)]
pub enum BpeError {
    #[error("io {0}")]
    Io(#[from] io::Error),
    #[error("not a valid utf8")]
    Utf8(#[from] str::Utf8Error),
    #[error("file is too large to address on this platform: {0} bytes")]
    FileTooLarge(u64),
    #[error("expected EOF at {expected_end}")]
    UnexpectedEof {
        expected_end: usize,
        actual_end: usize,
    },
    #[error("there is unconsumed content at ({offset_start}, {offset_end})")]
    UnconsumedInput {
        offset_start: usize,
        offset_end: usize,
    },
    #[error("Invalid UTF8 codepoint at byte {offset}")]
    InvalidUtf8 { offset: usize },
    #[error("There is pretokenized gap at byte {offset}")]
    UnmatchedInput { offset: usize },
}

impl From<BpeError> for PyErr {
    fn from(value: BpeError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

type TokenId = u32;

struct Word {
    tokens: Vec<TokenId>,
    repetition: usize,
}

#[pyfunction(signature = (input_path,
vocab_size, special_tokens, **kwargs))]
fn run_train_bpe(
    py: Python<'_>,
    input_path: PathBuf,
    vocab_size: usize,
    special_tokens: Vec<String>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(HashMap<usize, Py<PyBytes>>, Vec<(Py<PyBytes>, Py<PyBytes>)>)> {
    let file = File::open(input_path)?;
    let pretoken_counts = pretokenize(file, &special_tokens)?;
    let mut words: Vec<_> = pretoken_counts
        .into_iter()
        .map(|(word, count)| Word {
            tokens: word.bytes().map(|b| b as _).collect(),
            repetition: count as _,
        })
        .collect();
    let mut pair_to_word_index_list = HashMap::<(TokenId, TokenId), HashSet<usize>>::new();
    let mut pair_count = HashMap::<(TokenId, TokenId), usize>::new();
    for (index, word) in words.iter().enumerate() {
        word.tokens.windows(2).for_each(|window| {
            let &[p, q] = window else { unreachable!() };
            pair_to_word_index_list
                .entry((p, q))
                .or_default()
                .insert(index);
            pair_count
                .entry((p, q))
                .or_default()
                .add_assign(word.repetition);
        });
    }

    // use full bytes to sort
    let mut pair_heap = pair_count
        .iter()
        .map(|(&pair, &count)| (count, (vec![pair.0 as u8], vec![pair.1 as u8]), pair))
        .collect::<BinaryHeap<_>>();
    let mut vocab = (u8::MIN..=u8::MAX)
        .map(|i| (i as TokenId, vec![i]))
        .collect::<HashMap<_, _>>();
    let mut merge_list = Vec::new();

    let mut next_token_id = vocab.len() as TokenId;
    for special_token in special_tokens {
        vocab.insert(next_token_id, special_token.into_bytes());
        next_token_id += 1;
    }

    while let Some((count, (left_bytes, right_bytes), pair)) = pair_heap.pop()
        && next_token_id < (vocab_size as u32)
    {
        if pair_count[&pair] != count {
            // stale entry
            continue;
        }
        let token = iter::chain(&left_bytes, &right_bytes).copied().collect();
        vocab.insert(next_token_id, token);
        merge_list.push((
            PyBytes::new(py, &left_bytes).unbind(),
            PyBytes::new(py, &right_bytes).unbind(),
        ));

        let mut updated_pair_set = HashSet::new();
        for index in pair_to_word_index_list.remove(&pair).unwrap() {
            update_word(&mut words[index], pair, next_token_id, |pair, count| {
                let v = pair_count.entry(pair).or_default();
                *v = v.strict_add_signed(count);
                updated_pair_set.insert(pair);
                pair_to_word_index_list
                    .entry(pair)
                    .or_default()
                    .insert(index);
            });
        }
        pair_heap.extend(
            updated_pair_set
                .into_iter()
                .map(|pair| (pair_count[&pair], pair))
                .filter(|(count, _)| *count > 0)
                .map(|(count, pair)| {
                    (
                        count,
                        (vocab[&pair.0].clone(), vocab[&pair.1].clone()),
                        pair,
                    )
                }),
        );
        next_token_id += 1;
    }

    let vocab = vocab
        .into_iter()
        .map(|(token_id, word)| (token_id as _, PyBytes::new(py, &word).unbind()))
        .collect();

    Ok((vocab, merge_list))
}

fn update_word(
    w: &mut Word,
    merge_pair: (TokenId, TokenId),
    new_token_id: TokenId,
    mut update_pair: impl FnMut((TokenId, TokenId), isize),
) {
    let tokens = &mut w.tokens;
    let count = w.repetition as isize;
    let mut read = 0;
    let mut write = 0;

    while read < tokens.len() - 1 {
        if (tokens[read], tokens[read + 1]) == merge_pair {
            update_pair(merge_pair, -count);
            if write >= 1 {
                update_pair((tokens[write - 1], merge_pair.0), -count);
                update_pair((tokens[write - 1], new_token_id), count);
            }
            if read + 2 < tokens.len() {
                update_pair((merge_pair.1, tokens[read + 2]), -count);
                update_pair((new_token_id, tokens[read + 2]), count);
            }
            tokens[write] = new_token_id;
            read += 2;
        } else {
            if write != read {
                tokens[write] = tokens[read];
            }
            read += 1;
        }
        write += 1;
    }

    if read < tokens.len() {
        tokens[write] = tokens[read];
        write += 1;
    }
    tokens.truncate(write);
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{Word, update_word};

    #[test]
    fn test_update_word() {
        for (word, merge_pair, new_token_id, merged_word, updates) in [
            (
                vec![0, 1, 1],
                (0, 1),
                2,
                vec![2, 1],
                vec![((0, 1), -1), ((1, 1), -1), ((2, 1), 1)],
            ),
            (
                vec![0, 1, 0, 1],
                (0, 1),
                2,
                vec![2, 2],
                vec![((0, 1), -2), ((2, 2), 1), ((1, 0), -1)],
            ),
            (
                vec![0, 1, 1, 1, 0, 1],
                (0, 1),
                2,
                vec![2, 1, 1, 2],
                vec![
                    ((0, 1), -2),
                    ((1, 1), -1),
                    ((1, 0), -1),
                    ((2, 1), 1),
                    ((1, 2), 1),
                ],
            ),
            (
                vec![1, 1, 1],
                (1, 1),
                2,
                vec![2, 1],
                vec![((1, 1), -2), ((2, 1), 1)],
            ),
        ] {
            let mut word = Word {
                tokens: word,
                repetition: 1,
            };
            let mut merge_updates = HashMap::<_, isize>::new();
            update_word(&mut word, merge_pair, new_token_id, |pair, count| {
                *merge_updates.entry(pair).or_default() += count;
            });
            merge_updates.retain(|_, count| *count != 0);
            assert_eq!(word.tokens, merged_word);
            assert_eq!(merge_updates, updates.into_iter().collect());
        }
    }
}
