use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[pyfunction(signature = (input_path,
vocab_size, special_tokens, **kwargs))]
fn run_train_bpe(
    py: Python<'_>,
    input_path: PathBuf,
    vocab_size: usize,
    special_tokens: Vec<String>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(HashMap<usize, Py<PyBytes>>, Vec<(Py<PyBytes>, Py<PyBytes>)>)> {
    Ok((HashMap::new(), vec![]))
}

#[pymodule]
fn cs336_bpe(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_train_bpe, module)?)?;
    Ok(())
}
