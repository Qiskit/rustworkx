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

use std::cmp::Ordering;

use hashbrown::HashMap;
use rayon::prelude::*;

use pyo3::IntoPyObjectExt;
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use petgraph::stable_graph::{EdgeIndex, EdgeReference, NodeIndex};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use crate::{graph, is_valid_weight};

use rustworkx_core::steiner_tree::metric_closure as core_metric_closure;
use rustworkx_core::steiner_tree::steiner_tree as core_steiner_tree;

/// Return the metric closure of a graph
///
/// The metric closure of a graph is the complete graph in which each edge is
/// weighted by the shortest path distance between the nodes in the graph.
///
/// :param PyGraph graph: The input graph to find the metric closure for
/// :param weight_fn: A callable object that will be passed an edge's
///     weight/data payload and expected to return a ``float``. For example,
///     you can use ``weight_fn=float`` to cast every weight as a float
///
/// :return: A metric closure graph from the input graph
/// :rtype: PyGraph
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, weight_fn, /)")]
pub fn metric_closure(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: PyObject,
) -> PyResult<graph::PyGraph> {
    let callable = |e: EdgeReference<PyObject>| -> PyResult<f64> {
        let data = e.weight();
        let raw = weight_fn.call1(py, (data,))?;
        let weight = raw.extract(py)?;
        is_valid_weight(weight)
    };
    if let Some(result_graph) = core_metric_closure(&graph.graph, callable)? {
        let mut out_graph = graph.clone();
        out_graph.graph.clear_edges();
        for edge in result_graph.edge_indices() {
            let (source, target) = result_graph.edge_endpoints(edge).unwrap();
            let weight = result_graph.edge_weight(edge).unwrap();
            out_graph.graph.add_edge(
                *result_graph.node_weight(source).unwrap(),
                *result_graph.node_weight(target).unwrap(),
                weight.into_py_any(py)?,
            );
        }
        Ok(out_graph)
    } else {
        Err(PyValueError::new_err(
            "The input graph must be a connected graph. The metric closure is \
            not defined for a graph with unconnected nodes",
        ))
    }
}

/// Return an approximation to the minimum Steiner tree of a graph.
///
/// The minimum tree of ``graph`` with regard to a set of ``terminal_nodes``
/// is a tree within ``graph`` that spans those nodes and has a minimum size
/// (measured as the sum of edge weights) among all such trees.
///
/// The minimum steiner tree can be approximated by computing the minimum
/// spanning tree of the subgraph of the metric closure of ``graph`` induced
/// by the terminal nodes, where the metric closure of ``graph`` is the
/// complete graph in which each edge is weighted by the shortest path distance
/// between nodes in ``graph``.
///
/// This algorithm [1]_ produces a tree whose weight is within a
/// :math:`(2 - (2 / t))` factor of the weight of the optimal Steiner tree
/// where :math:`t` is the number of terminal nodes. The algorithm implemented
/// here is due to [2]_ . It avoids computing all pairs shortest paths but rather
/// reduces the problem to a single source shortest path and a minimum spanning tree
/// problem.
///
/// :param PyGraph graph: The graph to compute the minimum Steiner tree for
/// :param list terminal_nodes: The list of node indices for which the Steiner
///     tree is to be computed for.
/// :param weight_fn: A callable object that will be passed an edge's
///     weight/data payload and expected to return a ``float``. For example,
///     you can use ``weight_fn=float`` to cast every weight as a float.
///
/// :returns: An approximation to the minimal steiner tree of ``graph`` induced
///     by ``terminal_nodes``.
/// :rtype: PyGraph
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
///
/// .. [1] Kou, Markowsky & Berman,
///    "A fast algorithm for Steiner trees"
///    Acta Informatica 15, 141–145 (1981).
///    https://link.springer.com/article/10.1007/BF00288961
/// .. [2] Kurt Mehlhorn,
///    "A faster approximation algorithm for the Steiner problem in graphs"
///    https://doi.org/10.1016/0020-0190(88)90066-X
#[pyfunction]
#[pyo3(text_signature = "(graph, terminal_nodes, weight_fn, /)")]
pub fn steiner_tree(
    py: Python,
    graph: &mut graph::PyGraph,
    terminal_nodes: Vec<usize>,
    weight_fn: PyObject,
) -> PyResult<graph::PyGraph> {
    let callable = |e: EdgeReference<PyObject>| -> PyResult<f64> {
        let data = e.weight();
        let raw = weight_fn.call1(py, (data,))?;
        raw.extract(py)
    };
    let mut terminal_n: Vec<NodeIndex> = Vec::with_capacity(terminal_nodes.len());
    for n in &terminal_nodes {
        let index = NodeIndex::new(*n);
        if graph.graph.node_weight(index).is_none() {
            return Err(PyValueError::new_err(format!(
                "Provided terminal node index {n} is not present in graph"
            )));
        }
        terminal_n.push(index);
    }
    let result = core_steiner_tree(&graph.graph, &terminal_n, callable)?;
    if let Some(result) = result {
        let mut out_graph = graph.clone();
        for node in graph
            .graph
            .node_indices()
            .filter(|node| !result.used_node_indices.contains(&node.index()))
        {
            out_graph.graph.remove_node(node);
        }
        for edge in graph.graph.edge_references().filter(|edge| {
            let source = edge.source().index();
            let target = edge.target().index();
            !result.used_edge_endpoints.contains(&(source, target))
                && !result.used_edge_endpoints.contains(&(target, source))
        }) {
            out_graph.graph.remove_edge(edge.id());
        }
        deduplicate_edges(py, &mut out_graph, &weight_fn)?;
        if out_graph.graph.node_count() != graph.graph.node_count() {
            out_graph.node_removed = true;
        }
        Ok(out_graph)
    } else {
        Err(PyValueError::new_err(
            "The terminal nodes in the input graph must belong to the same connected component. \
            The steiner tree is not defined for a graph with unconnected terminal nodes",
        ))
    }
}

fn deduplicate_edges(
    py: Python,
    out_graph: &mut graph::PyGraph,
    weight_fn: &PyObject,
) -> PyResult<()> {
    if out_graph.multigraph {
        // Find all edges between nodes
        let mut duplicate_map: HashMap<[NodeIndex; 2], Vec<(EdgeIndex, PyObject)>> = HashMap::new();
        for edge in out_graph.graph.edge_references() {
            if duplicate_map.contains_key(&[edge.source(), edge.target()]) {
                duplicate_map
                    .get_mut(&[edge.source(), edge.target()])
                    .unwrap()
                    .push((edge.id(), edge.weight().clone_ref(py)));
            } else if duplicate_map.contains_key(&[edge.target(), edge.source()]) {
                duplicate_map
                    .get_mut(&[edge.target(), edge.source()])
                    .unwrap()
                    .push((edge.id(), edge.weight().clone_ref(py)));
            } else {
                duplicate_map.insert(
                    [edge.source(), edge.target()],
                    vec![(edge.id(), edge.weight().clone_ref(py))],
                );
            }
        }
        // For a node pair with > 1 edge find minimum edge and remove others
        for edges_raw in duplicate_map.values().filter(|x| x.len() > 1) {
            let mut edges: Vec<(EdgeIndex, f64)> = Vec::with_capacity(edges_raw.len());
            for edge in edges_raw {
                let res = weight_fn.call1(py, (&edge.1,))?;
                let raw = res.into_py_any(py)?;
                let weight = raw.extract(py)?;
                edges.push((edge.0, weight));
            }
            edges.par_sort_unstable_by(|a, b| {
                let weight_a = a.1;
                let weight_b = b.1;
                weight_a.partial_cmp(&weight_b).unwrap_or(Ordering::Less)
            });
            edges[1..].iter().for_each(|x| {
                out_graph.graph.remove_edge(x.0);
            });
        }
    }
    Ok(())
}
