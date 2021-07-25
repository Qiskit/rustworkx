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

#![allow(clippy::float_cmp)]

mod longest_path;

use hashbrown::{HashMap, HashSet};

use crate::digraph;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;

use super::InvalidNode;

use super::iterators::NodeIndices;

/// Find the longest path in a DAG
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
/// :param weight_fn: A python callable that if set will be passed the 3
///     positional arguments, the source node, the target node, and the edge
///     weight for each edge as the function traverses the graph. It is expected
///     to return an unsigned integer weight for that edge. For example,
///     ``dag_longest_path(graph, lambda: _, __, weight: weight)`` could be
///     use to just use an integer edge weight. It's also worth noting that this
///     function traverses in topological order and only checks incoming edges to
///     each node.
///
/// :returns: The node indices of the longest path on the DAG
/// :rtype: NodeIndices
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[pyo3(text_signature = "(graph, /, weight_fn=None)")]
pub fn dag_longest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
) -> PyResult<NodeIndices> {
    let edge_weight_callable =
        |source: usize, target: usize, weight: &PyObject| -> PyResult<usize> {
            match &weight_fn {
                Some(weight_fn) => {
                    let res = weight_fn.call1(py, (source, target, weight))?;
                    res.extract(py)
                }
                None => Ok(1),
            }
        };
    Ok(NodeIndices {
        nodes: longest_path::longest_path(graph, edge_weight_callable)?.0,
    })
}

/// Find the length of the longest path in a DAG
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
/// :param weight_fn: A python callable that if set will be passed the 3
///     positional arguments, the source node, the target node, and the edge
///     weight for each edge as the function traverses the graph. It is expected
///     to return an unsigned integer weight for that edge. For example,
///     ``dag_longest_path(graph, lambda: _, __, weight: weight)`` could be
///     use to just use an integer edge weight. It's also worth noting that this
///     function traverses in topological order and only checks incoming edges to
///     each node.
///
/// :returns: The longest path length on the DAG
/// :rtype: int
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[pyo3(text_signature = "(graph, /, weight_fn=None)")]
pub fn dag_longest_path_length(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
) -> PyResult<usize> {
    let edge_weight_callable =
        |source: usize, target: usize, weight: &PyObject| -> PyResult<usize> {
            match &weight_fn {
                Some(weight_fn) => {
                    let res = weight_fn.call1(py, (source, target, weight))?;
                    res.extract(py)
                }
                None => Ok(1),
            }
        };
    let (_, path_weight) =
        longest_path::longest_path(graph, edge_weight_callable)?;
    Ok(path_weight)
}

/// Find the weighted longest path in a DAG
///
/// This function differs from :func:`retworkx.dag_longest_path` in that
/// this function requires a ``weight_fn`` parameter, and the ``weight_fn`` is
/// expected to return a ``float`` not an ``int``.
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
/// :param weight_fn: A python callable that will be passed the 3
///     positional arguments, the source node, the target node, and the edge
///     weight for each edge as the function traverses the graph. It is expected
///     to return a float weight for that edge. For example,
///     ``dag_longest_path(graph, lambda: _, __, weight: weight)`` could be
///     used to just use a float edge weight. It's also worth noting that this
///     function traverses in topological order and only checks incoming edges to
///     each node.
///
/// :returns: The node indices of the longest path on the DAG
/// :rtype: NodeIndices
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[pyo3(text_signature = "(graph, weight_fn, /)")]
pub fn dag_weighted_longest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: PyObject,
) -> PyResult<NodeIndices> {
    let edge_weight_callable =
        |source: usize, target: usize, weight: &PyObject| -> PyResult<f64> {
            let res = weight_fn.call1(py, (source, target, weight))?;
            let float_res: f64 = res.extract(py)?;
            if float_res.is_nan() {
                return Err(PyValueError::new_err(
                    "NaN is not a valid edge weight",
                ));
            }
            Ok(float_res)
        };
    Ok(NodeIndices {
        nodes: longest_path::longest_path(graph, edge_weight_callable)?.0,
    })
}

/// Find the length of the weighted longest path in a DAG
///
/// This function differs from :func:`retworkx.dag_longest_path_length` in that
/// this function requires a ``weight_fn`` parameter, and the ``weight_fn`` is
/// expected to return a ``float`` not an ``int``.
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
/// :param weight_fn: A python callable that will be passed the 3
///     positional arguments, the source node, the target node, and the edge
///     weight for each edge as the function traverses the graph. It is expected
///     to return a float weight for that edge. For example,
///     ``dag_longest_path(graph, lambda: _, __, weight: weight)`` could be
///     used to just use a float edge weight. It's also worth noting that this
///     function traverses in topological order and only checks incoming edges to
///     each node.
///
/// :returns: The longest path length on the DAG
/// :rtype: float
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[pyo3(text_signature = "(graph, weight_fn, /)")]
pub fn dag_weighted_longest_path_length(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: PyObject,
) -> PyResult<f64> {
    let edge_weight_callable =
        |source: usize, target: usize, weight: &PyObject| -> PyResult<f64> {
            let res = weight_fn.call1(py, (source, target, weight))?;
            let float_res: f64 = res.extract(py)?;
            if float_res.is_nan() {
                return Err(PyValueError::new_err(
                    "NaN is not a valid edge weight",
                ));
            }
            Ok(float_res)
        };
    let (_, path_weight) =
        longest_path::longest_path(graph, edge_weight_callable)?;
    Ok(path_weight)
}

/// Check that the PyDiGraph or PyDAG doesn't have a cycle
///
/// :param PyDiGraph graph: The graph to check for cycles
///
/// :returns: ``True`` if there are no cycles in the input graph, ``False``
///     if there are cycles
/// :rtype: bool
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn is_directed_acyclic_graph(graph: &digraph::PyDiGraph) -> bool {
    match algo::toposort(graph, None) {
        Ok(_nodes) => true,
        Err(_err) => false,
    }
}

/// Return a list of layers
///
/// A layer is a subgraph whose nodes are disjoint, i.e.,
/// a layer has depth 1. The layers are constructed using a greedy algorithm.
///
/// :param PyDiGraph graph: The DAG to get the layers from
/// :param list first_layer: A list of node ids for the first layer. This
///     will be the first layer in the output
///
/// :returns: A list of layers, each layer is a list of node data
/// :rtype: list
///
/// :raises InvalidNode: If a node index in ``first_layer`` is not in the graph
#[pyfunction]
#[pyo3(text_signature = "(dag, first_layer, /)")]
pub fn layers(
    py: Python,
    dag: &digraph::PyDiGraph,
    first_layer: Vec<usize>,
) -> PyResult<PyObject> {
    let mut output: Vec<Vec<&PyObject>> = Vec::new();
    // Convert usize to NodeIndex
    let mut first_layer_index: Vec<NodeIndex> = Vec::new();
    for index in first_layer {
        first_layer_index.push(NodeIndex::new(index));
    }

    let mut cur_layer = first_layer_index;
    let mut next_layer: Vec<NodeIndex> = Vec::new();
    let mut predecessor_count: HashMap<NodeIndex, usize> = HashMap::new();

    let mut layer_node_data: Vec<&PyObject> = Vec::new();
    for layer_node in &cur_layer {
        let node_data = match dag.graph.node_weight(*layer_node) {
            Some(data) => data,
            None => {
                return Err(InvalidNode::new_err(format!(
                    "An index input in 'first_layer' {} is not a valid node index in the graph",
                    layer_node.index()),
                ))
            }
        };
        layer_node_data.push(node_data);
    }
    output.push(layer_node_data);

    // Iterate until there are no more
    while !cur_layer.is_empty() {
        for node in &cur_layer {
            let children = dag
                .graph
                .neighbors_directed(*node, petgraph::Direction::Outgoing);
            let mut used_indexes: HashSet<NodeIndex> = HashSet::new();
            for succ in children {
                // Skip duplicate successors
                if used_indexes.contains(&succ) {
                    continue;
                }
                used_indexes.insert(succ);
                let mut multiplicity: usize = 0;
                let raw_edges = dag
                    .graph
                    .edges_directed(*node, petgraph::Direction::Outgoing);
                for edge in raw_edges {
                    if edge.target() == succ {
                        multiplicity += 1;
                    }
                }
                predecessor_count
                    .entry(succ)
                    .and_modify(|e| *e -= multiplicity)
                    .or_insert(dag.in_degree(succ.index()) - multiplicity);
                if *predecessor_count.get(&succ).unwrap() == 0 {
                    next_layer.push(succ);
                    predecessor_count.remove(&succ);
                }
            }
        }
        let mut layer_node_data: Vec<&PyObject> = Vec::new();
        for layer_node in &next_layer {
            layer_node_data.push(&dag[*layer_node]);
        }
        if !layer_node_data.is_empty() {
            output.push(layer_node_data);
        }
        cur_layer = next_layer;
        next_layer = Vec::new();
    }
    Ok(PyList::new(py, output).into())
}
