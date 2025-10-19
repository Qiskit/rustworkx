use crate::{digraph::PyDiGraph, graph::PyGraph, StablePyGraph};
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
fn graph_to_coo<Ty: EdgeType>(
    _py: Python<'_>,
    graph: &StablePyGraph<Ty>,
) -> PyResult<MatrixMarketData> {
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
fn coo_to_graph(py: Python<'_>, coo: &CooMatrix<f64>, is_directed: bool) -> PyResult<PyObject> {
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

/// Read Matrix Market from string contents and return the graph (PyGraph or PyDiGraph)
#[pyfunction]
#[pyo3(signature=(contents,),text_signature = "(contents)")]
pub fn read_matrix_market(py: Python<'_>, contents: &str) -> PyResult<PyObject> {
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

/// Read Matrix Market from a file path and return the graph
#[pyfunction]
#[pyo3(signature=(path,),text_signature = "(path)")]
pub fn read_matrix_market_file(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let contents = fs::read_to_string(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

    read_matrix_market(py, &contents)
}

/// Write Matrix Market from a python graph (PyGraph or PyDiGraph). If `path` is Some then write file, else return string.
#[pyfunction]
#[pyo3(signature=(graph, path=None),text_signature = "(graph, /, path=None)")]
pub fn graph_write_matrix_market(
    py: Python<'_>,
    graph: &PyGraph,
    path: Option<&str>,
) -> PyResult<Option<String>> {
    let (nrows, ncols, rows, cols, vals) = graph_to_coo(py, &graph.graph)?;

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
        Ok(Some(
            String::from_utf8(cursor.into_inner()).unwrap_or_default(),
        ))
    }
}

/// Write a PyDiGraph (directed) to Matrix Market
#[pyfunction]
#[pyo3(signature=(graph, path=None),text_signature = "(graph, /, path=None)")]
pub fn digraph_write_matrix_market(
    py: Python<'_>,
    graph: &PyDiGraph,
    path: Option<&str>,
) -> PyResult<Option<String>> {
    let (nrows, ncols, rows, cols, vals) = graph_to_coo(py, &graph.graph)?;

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
        Ok(Some(
            String::from_utf8(cursor.into_inner()).unwrap_or_default(),
        ))
    }
}
