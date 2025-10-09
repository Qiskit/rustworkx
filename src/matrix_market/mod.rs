use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::io::{
    load_coo_from_matrix_market_file, load_coo_from_matrix_market_str, save_to_matrix_market,
    save_to_matrix_market_file,
};
use pyo3::prelude::*;
use std::io::Cursor;

type MatrixMarketData = (usize, usize, Vec<usize>, Vec<usize>, Vec<f64>);

#[pyfunction]
pub fn read_matrix_market_data(input: &str, from_file: bool) -> PyResult<MatrixMarketData> {
    // load COO matrix from file or string depending on the flag
    let coo: CooMatrix<f64> = if from_file {
        load_coo_from_matrix_market_file(input)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?
    } else {
        load_coo_from_matrix_market_str(input)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
    };

    Ok((
        coo.nrows(),
        coo.ncols(),
        coo.row_indices().to_vec(),
        coo.col_indices().to_vec(),
        coo.values().to_vec(),
    ))
}

#[pyfunction]
pub fn write_matrix_market_data(
    nrows: usize,
    ncols: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
    path: Option<&str>,
) -> PyResult<Option<String>> {
    let coo = CooMatrix::try_from_triplets(nrows, ncols, rows, cols, vals)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    if let Some(p) = path {
        // Save to file
        save_to_matrix_market_file(&coo, p)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        Ok(None)
    } else {
        // Save to in-memory string
        let mut cursor = Cursor::new(Vec::new());
        save_to_matrix_market(&mut cursor, &coo)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        let result_str = String::from_utf8(cursor.into_inner()).unwrap_or_default();
        Ok(Some(result_str))
    }
}
