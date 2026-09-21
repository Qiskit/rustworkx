// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

use crate::{StablePyGraph, digraph, graph, weight_callable};
use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use numpy::IntoPyArray;
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::{Directed, EdgeType, Undirected, algo};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyImportError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

const SCIPY_REQUIRED_ERROR: &str =
    "scipy is required for biadjacency matrix conversion. Install scipy to use this function.";

type ParallelEdgeFn = fn(f64, f64, usize) -> f64;
type SparseMatrixData = (usize, usize, Vec<usize>, Vec<usize>, Vec<f64>);

pub(super) struct BiadjacencyMatrixOptions<'a> {
    pub(super) row_order: Vec<usize>,
    pub(super) column_order: Vec<usize>,
    pub(super) weight_fn: Option<Py<PyAny>>,
    pub(super) default_weight: f64,
    pub(super) format: &'a str,
    pub(super) parallel_edge: &'a str,
}

fn scipy_sparse_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    py.import("scipy.sparse")
        .map_err(|_| PyImportError::new_err(SCIPY_REQUIRED_ERROR))
}

fn parallel_edge_fn(parallel_edge: &str) -> PyResult<ParallelEdgeFn> {
    match parallel_edge {
        "sum" => Ok(|current, edge_weight, _| current + edge_weight),
        "min" => Ok(|current, edge_weight, _| current.min(edge_weight)),
        "max" => Ok(|current, edge_weight, _| current.max(edge_weight)),
        "avg" => Ok(|current, edge_weight, count| {
            (current * count as f64 + edge_weight) / ((count + 1) as f64)
        }),
        _ => Err(PyValueError::new_err(
            "Parallel edges can currently only be dealt with using \"sum\", \"min\", \"max\", or \"avg\".",
        )),
    }
}

fn validate_biadjacency_node_order<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
    row_order: &[usize],
    column_order: &[usize],
) -> PyResult<()> {
    let mut row_nodes = HashSet::new();
    for node in row_order {
        if !row_nodes.insert(*node) {
            return Err(PyValueError::new_err(format!(
                "row_order contains duplicate node index {node}"
            )));
        }
    }

    let mut column_nodes = HashSet::new();
    for node in column_order {
        if !column_nodes.insert(*node) {
            return Err(PyValueError::new_err(format!(
                "column_order contains duplicate node index {node}"
            )));
        }
        if row_nodes.contains(node) {
            return Err(PyValueError::new_err(format!(
                "row_order and column_order must be disjoint; node index {node} appears in both"
            )));
        }
    }

    for node in row_order.iter().chain(column_order.iter()) {
        if !graph.contains_node(NodeIndex::new(*node)) {
            return Err(PyValueError::new_err(format!(
                "Node index {node} is not present in the graph"
            )));
        }
    }
    Ok(())
}

fn biadjacency_node_map(node_order: &[usize]) -> HashMap<usize, usize> {
    node_order
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect()
}

fn digraph_biadjacency_index(
    source: usize,
    target: usize,
    row_map: &HashMap<usize, usize>,
    column_map: &HashMap<usize, usize>,
) -> Option<(usize, usize)> {
    row_map
        .get(&source)
        .and_then(|row| column_map.get(&target).map(|column| (*row, *column)))
}

fn graph_biadjacency_index(
    source: usize,
    target: usize,
    row_map: &HashMap<usize, usize>,
    column_map: &HashMap<usize, usize>,
) -> Option<(usize, usize)> {
    digraph_biadjacency_index(source, target, row_map, column_map).or_else(|| {
        row_map
            .get(&target)
            .and_then(|row| column_map.get(&source).map(|column| (*row, *column)))
    })
}

fn add_biadjacency_edge(
    weights: &mut HashMap<[usize; 2], (f64, usize)>,
    row: usize,
    column: usize,
    edge_weight: f64,
    parallel_edge_fn: ParallelEdgeFn,
) {
    match weights.entry([row, column]) {
        Entry::Vacant(entry) => {
            entry.insert((edge_weight, 1));
        }
        Entry::Occupied(mut entry) => {
            let (current, count) = entry.get_mut();
            *current = parallel_edge_fn(*current, edge_weight, *count);
            *count += 1;
        }
    }
}

fn build_biadjacency_triplets<'py, Ty, F>(
    py: Python<'py>,
    graph: &StablePyGraph<Ty>,
    options: &'py BiadjacencyMatrixOptions<'_>,
    edge_index: F,
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<f64>)>
where
    Ty: EdgeType,
    F: Fn(usize, usize, &HashMap<usize, usize>, &HashMap<usize, usize>) -> Option<(usize, usize)>,
{
    let parallel_edge_fn = parallel_edge_fn(options.parallel_edge)?;
    let row_map = biadjacency_node_map(&options.row_order);
    let column_map = biadjacency_node_map(&options.column_order);
    let mut weights = HashMap::new();

    for edge in graph.edge_references() {
        let source = edge.source().index();
        let target = edge.target().index();
        if let Some((row, column)) = edge_index(source, target, &row_map, &column_map) {
            let edge_weight = weight_callable(
                py,
                &options.weight_fn,
                edge.weight(),
                options.default_weight,
            )?;
            add_biadjacency_edge(&mut weights, row, column, edge_weight, parallel_edge_fn);
        }
    }

    let mut entries: Vec<(usize, usize, f64)> = weights
        .into_iter()
        .map(|([row, column], (weight, _))| (row, column, weight))
        .collect();
    entries.sort_unstable_by_key(|(row, column, _)| (*row, *column));

    let mut rows = Vec::with_capacity(entries.len());
    let mut columns = Vec::with_capacity(entries.len());
    let mut data = Vec::with_capacity(entries.len());
    for (row, column, weight) in entries {
        rows.push(row as i64);
        columns.push(column as i64);
        data.push(weight);
    }
    Ok((rows, columns, data))
}

fn triplets_to_scipy_sparse<'py>(
    py: Python<'py>,
    rows: Vec<i64>,
    columns: Vec<i64>,
    data: Vec<f64>,
    shape: (usize, usize),
    format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let sparse = scipy_sparse_module(py)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", shape)?;
    // Hand the triplets to SciPy as NumPy arrays so the buffers are copied
    // wholesale instead of boxing every entry into a Python object.
    let coo = sparse.call_method(
        "coo_array",
        ((
            data.into_pyarray(py),
            (rows.into_pyarray(py), columns.into_pyarray(py)),
        ),),
        Some(&kwargs),
    )?;
    coo.call_method1("asformat", (format,))
}

pub(super) fn digraph_to_biadjacency_matrix<'py>(
    py: Python<'py>,
    graph: &StablePyGraph<Directed>,
    options: BiadjacencyMatrixOptions<'_>,
) -> PyResult<Bound<'py, PyAny>> {
    validate_biadjacency_node_order(graph, &options.row_order, &options.column_order)?;
    let (rows, columns, data) =
        build_biadjacency_triplets(py, graph, &options, digraph_biadjacency_index)?;
    triplets_to_scipy_sparse(
        py,
        rows,
        columns,
        data,
        (options.row_order.len(), options.column_order.len()),
        options.format,
    )
}

pub(super) fn graph_to_biadjacency_matrix<'py>(
    py: Python<'py>,
    graph: &StablePyGraph<Undirected>,
    options: BiadjacencyMatrixOptions<'_>,
) -> PyResult<Bound<'py, PyAny>> {
    validate_biadjacency_node_order(graph, &options.row_order, &options.column_order)?;
    let (rows, columns, data) =
        build_biadjacency_triplets(py, graph, &options, graph_biadjacency_index)?;
    triplets_to_scipy_sparse(
        py,
        rows,
        columns,
        data,
        (options.row_order.len(), options.column_order.len()),
        options.format,
    )
}

fn scipy_sparse_to_coo_data<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<SparseMatrixData> {
    let sparse = scipy_sparse_module(py)?;
    let is_sparse: bool = sparse.call_method1("issparse", (matrix,))?.extract()?;
    if !is_sparse {
        return Err(PyTypeError::new_err(
            "matrix must be a scipy sparse matrix or sparse array",
        ));
    }

    let coo = matrix.call_method0("tocoo")?;
    let shape: (usize, usize) = coo.getattr("shape")?.extract()?;
    let rows: Vec<usize> = coo.getattr("row")?.call_method0("tolist")?.extract()?;
    let columns: Vec<usize> = coo.getattr("col")?.call_method0("tolist")?.extract()?;
    let data: Vec<f64> = coo.getattr("data")?.call_method0("tolist")?.extract()?;

    if rows.len() != columns.len() || rows.len() != data.len() {
        return Err(PyValueError::new_err(
            "scipy sparse matrix row, column, and data arrays must have the same length",
        ));
    }

    Ok((shape.0, shape.1, rows, columns, data))
}

fn stable_graph_from_biadjacency_matrix<Ty: EdgeType>(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
) -> PyResult<StablePyGraph<Ty>> {
    let (row_count, column_count, rows, columns, data) = scipy_sparse_to_coo_data(py, matrix)?;
    let mut out_graph = StablePyGraph::<Ty>::with_capacity(row_count + column_count, data.len());

    for node in 0..(row_count + column_count) {
        out_graph.add_node(node.into_py_any(py)?);
    }

    for ((row, column), weight) in rows.into_iter().zip(columns).zip(data) {
        if row >= row_count || column >= column_count {
            return Err(PyValueError::new_err(
                "scipy sparse matrix contains an entry outside its shape",
            ));
        }
        out_graph.add_edge(
            NodeIndex::new(row),
            NodeIndex::new(row_count + column),
            weight.into_py_any(py)?,
        );
    }

    Ok(out_graph)
}

pub(crate) fn graph_from_biadjacency_matrix(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
) -> PyResult<graph::PyGraph> {
    Ok(graph::PyGraph {
        graph: stable_graph_from_biadjacency_matrix::<Undirected>(py, matrix)?,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    })
}

pub(crate) fn digraph_from_biadjacency_matrix(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
) -> PyResult<digraph::PyDiGraph> {
    Ok(digraph::PyDiGraph {
        graph: stable_graph_from_biadjacency_matrix::<Directed>(py, matrix)?,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    })
}
