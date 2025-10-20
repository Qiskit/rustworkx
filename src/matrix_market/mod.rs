use crate::{StablePyGraph, digraph::PyDiGraph, graph::PyGraph};
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::io::{
    load_coo_from_matrix_market_str, save_to_matrix_market, save_to_matrix_market_file,
};
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use pyo3::prelude::*;
use std::fs;
use std::io::Cursor;

use rustworkx_core::petgraph::algo::DfsSpace;
use rustworkx_core::petgraph::{Directed, EdgeType, Undirected};

type MatrixMarketData = (usize, usize, Vec<usize>, Vec<usize>, Vec<f64>);

/// Convert any StablePyGraph (directed or undirected) into COO triplets
fn graph_to_coo<Ty: EdgeType>(graph: &StablePyGraph<Ty>) -> PyResult<MatrixMarketData> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for edge in graph.edge_references() {
        let src: usize = edge.source().index();
        let dst: usize = edge.target().index();
        rows.push(src);
        cols.push(dst);
        vals.push(1.0);

        if !graph.is_directed() {
            rows.push(dst);
            cols.push(src);
            vals.push(1.0);
        }
    }

    Ok((graph.node_count(), graph.node_count(), rows, cols, vals))
}

/// Determine whether a MatrixMarket header indicates a directed matrix.
fn is_directed_from_header(header_line: &str) -> bool {
    if let Some(symmetry) = header_line
        .split_whitespace()
        .nth(4)
        .map(|s| s.to_lowercase())
    {
        !matches!(
            symmetry.as_str(),
            "symmetric" | "hermitian" | "skew-symmetric"
        )
    } else {
        true // default to directed
    }
}

/// Create a rustworkx graph from a COO matrix. Returns a PyObject (graph instance).
fn coo_to_graph(py: Python<'_>, coo: &CooMatrix<f64>, is_directed: bool) -> PyResult<Py<PyAny>> {
    if is_directed {
        let mut inner_graph: StablePyGraph<Directed> =
            StablePyGraph::with_capacity(coo.nrows(), coo.nnz());

        let nodes: Vec<_> = (0..coo.nrows())
            .map(|_| inner_graph.add_node(py.None()))
            .collect();

        for (r, c, _) in coo.triplet_iter() {
            inner_graph.add_edge(nodes[r], nodes[c], py.None());
        }

        let dg = PyDiGraph {
            graph: inner_graph,
            cycle_state: DfsSpace::default(),
            check_cycle: false,
            node_removed: false,
            multigraph: true,
            attrs: py.None(),
        };

        let obj = dg.into_pyobject(py)?.into_any();
        Ok(obj.into())
    } else {
        let mut inner_graph: StablePyGraph<Undirected> =
            StablePyGraph::with_capacity(coo.nrows(), coo.nnz());

        let nodes: Vec<_> = (0..coo.nrows())
            .map(|_| inner_graph.add_node(py.None()))
            .collect();

        for (r, c, _) in coo.triplet_iter() {
            inner_graph.add_edge(nodes[r], nodes[c], py.None());
        }

        let ug = PyGraph {
            graph: inner_graph,
            node_removed: false,
            multigraph: true,
            attrs: py.None(),
        };

        let obj = ug.into_pyobject(py)?.into_any();
        Ok(obj.into())
    }
}

/// Read a graph from Matrix Market format string contents.
///
/// Matrix Market is a human-readable ASCII file format for storing sparse matrices
/// as coordinate format (row, column, value) triplets. This function parses Matrix
/// Market formatted string contents and converts it to a graph representation.
///
/// For more information about Matrix Market format, see:
/// https://math.nist.gov/MatrixMarket/formats.html
///
/// :param str contents: The Matrix Market formatted string contents to parse.
///
/// :returns: A graph object constructed from the Matrix Market data. Returns a
///          :class:`~rustworkx.PyDiGraph` for directed matrices or a
///          :class:`~rustworkx.PyGraph` for undirected matrices.
/// :rtype: PyGraph or PyDiGraph
///
/// :raises ValueError: when the Matrix Market string contains invalid data or format.
#[pyfunction]
#[pyo3(signature=(contents,),text_signature = "(contents)")]
pub fn read_matrix_market(py: Python<'_>, contents: &str) -> PyResult<Py<PyAny>> {
    // find header (first non-comment/blank line)
    let header_line = contents
        .lines()
        .find(|line| {
            line.trim_start()
                .to_lowercase()
                .starts_with("%%matrixmarket")
        })
        .unwrap_or("%%MatrixMarket matrix coordinate real general");

    let is_directed = is_directed_from_header(header_line);
    let coo: CooMatrix<f64> = load_coo_from_matrix_market_str(contents)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    coo_to_graph(py, &coo, is_directed)
}

/// Read a graph from a Matrix Market format file.
///
/// Matrix Market is a human-readable ASCII file format for storing sparse matrices
/// as coordinate format (row, column, value) triplets. This function reads a Matrix
/// Market file and converts it to a graph representation.
///
/// For more information about Matrix Market format, see:
/// https://math.nist.gov/MatrixMarket/formats.html
///
/// :param str path: The file path to read the Matrix Market data from.
///
/// :returns: A graph object constructed from the Matrix Market file.
/// :rtype: PyGraph or PyDiGraph
///
/// :raises IOError: when an error occurs during file reading or parsing.
/// :raises ValueError: when the Matrix Market file contains invalid data or format.
#[pyfunction]
#[pyo3(signature=(path,),text_signature = "(path)")]
pub fn read_matrix_market_file(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    let contents = fs::read_to_string(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

    read_matrix_market(py, &contents)
}

/// Write a graph to Matrix Market format.
///
/// Matrix Market is a human-readable ASCII file format for storing sparse matrices
/// as coordinate format (row, column, value) triplets. The graph is converted to a
/// coordinate (COO) sparse matrix representation where edges become non-zero entries.
///
/// For more information about Matrix Market format, see:
/// https://math.nist.gov/MatrixMarket/formats.html
///
/// :param PyGraph graph: The graph to write in Matrix Market format.
/// :param Optional[str] path: Optional file path to write the output. If None,
///                            returns the Matrix Market content as a string.
///
/// :returns: None if a file path was provided, otherwise returns the Matrix Market
///          formatted content as a string.
/// :rtype: Optional[str]
///
/// :raises ValueError: when the graph cannot be converted to a valid COO matrix.
/// :raises IOError: when an error occurs during file I/O or format serialization.
#[pyfunction]
#[pyo3(signature=(graph, path=None),text_signature = "(graph, /, path=None)")]
pub fn graph_write_matrix_market(
    _py: Python<'_>,
    graph: &PyGraph,
    path: Option<&str>,
) -> PyResult<Option<String>> {
    let (nrows, ncols, rows, cols, vals) = graph_to_coo(&graph.graph)?;

    let coo = CooMatrix::try_from_triplets(nrows, ncols, rows, cols, vals)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    if let Some(p) = path {
        save_to_matrix_market_file(&coo, p)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        Ok(None)
    } else {
        let mut cursor = Cursor::new(Vec::new());
        save_to_matrix_market(&mut cursor, &coo)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        Ok(Some(String::from_utf8(cursor.into_inner())?))
    }
}

/// Write a directed graph to Matrix Market format.
///
/// Matrix Market is a human-readable ASCII file format for storing sparse matrices
/// as coordinate format (row, column, value) triplets. The directed graph is converted
/// to a coordinate (COO) sparse matrix representation where edges become non-zero entries.
///
/// For more information about Matrix Market format, see:
/// https://math.nist.gov/MatrixMarket/formats.html
///
/// :param PyDiGraph graph: The directed graph to write in Matrix Market format.
/// :param Optional[str] path: Optional file path to write the output. If None,
///                            returns the Matrix Market content as a string.
///
/// :returns: None if a file path was provided, otherwise returns the Matrix Market
///          formatted content as a string.
/// :rtype: Optional[str]
///
/// :raises ValueError: when the graph cannot be converted to a valid COO matrix.
/// :raises IOError: when an error occurs during file I/O or format serialization.
#[pyfunction]
#[pyo3(signature=(graph, path=None),text_signature = "(graph, /, path=None)")]
pub fn digraph_write_matrix_market(
    _py: Python<'_>,
    graph: &PyDiGraph,
    path: Option<&str>,
) -> PyResult<Option<String>> {
    let (nrows, ncols, rows, cols, vals) = graph_to_coo(&graph.graph)?;

    let coo = CooMatrix::try_from_triplets(nrows, ncols, rows, cols, vals)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    if let Some(p) = path {
        save_to_matrix_market_file(&coo, p)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        Ok(None)
    } else {
        let mut cursor = Cursor::new(Vec::new());
        save_to_matrix_market(&mut cursor, &coo)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        Ok(Some(String::from_utf8(cursor.into_inner())?))
    }
}
