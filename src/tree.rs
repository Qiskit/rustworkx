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
use std::collections::VecDeque;

use super::{graph, weight_callable};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList};
use pyo3::Python;

use petgraph::prelude::*;
use petgraph::stable_graph::EdgeReference;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{IntoEdgeReferences, NodeIndexable};

use rayon::prelude::*;

use crate::iterators::WeightedEdgeList;

/// Find the edges in the minimum spanning tree or forest of a graph
/// using Kruskal's algorithm.
///
/// :param PyGraph graph: Undirected graph
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         minimum_spanning_edges(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         minimum_spanning_edges(graph, weight_fn: float)
///
///     to cast the edge object as a float as the weight.
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
///
/// :returns: The :math:`N - |c|` edges of the Minimum Spanning Tree (or Forest, if :math:`|c| > 1`)
///     where :math:`N` is the number of nodes and :math:`|c|` is the number of connected components of the graph
/// :rtype: WeightedEdgeList
#[pyfunction(weight_fn = "None", default_weight = "1.0")]
#[pyo3(text_signature = "(graph, weight_fn=None, default_weight=1.0)")]
pub fn minimum_spanning_edges(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<WeightedEdgeList> {
    let mut subgraphs = UnionFind::<usize>::new(graph.graph.node_bound());

    let mut edge_list: Vec<(f64, EdgeReference<PyObject>)> =
        Vec::with_capacity(graph.graph.edge_count());
    for edge in graph.graph.edge_references() {
        let weight = weight_callable(py, &weight_fn, edge.weight(), default_weight)?;
        if weight.is_nan() {
            return Err(PyValueError::new_err("NaN found as an edge weight"));
        }
        edge_list.push((weight, edge));
    }

    edge_list.par_sort_unstable_by(|a, b| {
        let weight_a = a.0;
        let weight_b = b.0;
        weight_a.partial_cmp(&weight_b).unwrap_or(Ordering::Less)
    });

    let mut answer: Vec<(usize, usize, PyObject)> = Vec::new();
    for float_edge_pair in edge_list.iter() {
        let edge = float_edge_pair.1;
        let u = edge.source().index();
        let v = edge.target().index();
        if subgraphs.union(u, v) {
            let w = edge.weight().clone_ref(py);
            answer.push((u, v, w));
        }
    }

    Ok(WeightedEdgeList { edges: answer })
}

/// Find the minimum spanning tree or forest of a graph
/// using Kruskal's algorithm.
///
/// :param PyGraph graph: Undirected graph
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         minimum_spanning_tree(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         minimum_spanning_tree(graph, weight_fn: float)
///
///     to cast the edge object as a float as the weight.
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
///
/// :returns: A Minimum Spanning Tree (or Forest, if the graph is not connected).
///
/// :rtype: PyGraph
///
/// .. note::
///
///     The new graph will keep the same node indices, but edge indices might differ.
#[pyfunction(weight_fn = "None", default_weight = "1.0")]
#[pyo3(text_signature = "(graph, weight_fn=None, default_weight=1.0)")]
pub fn minimum_spanning_tree(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<graph::PyGraph> {
    let mut spanning_tree = (*graph).clone();
    spanning_tree.graph.clear_edges();

    for edge in minimum_spanning_edges(py, graph, weight_fn, default_weight)?
        .edges
        .iter()
    {
        spanning_tree.add_edge(edge.0, edge.1, edge.2.clone_ref(py));
    }

    Ok(spanning_tree)
}

/// Find balanced cut edge of the minmum spanning tree of a graph using node
/// contraction. Assumes that the tree is connected and is a spanning tree.
///
/// :param PyGraph graph: Undirected graph
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         minimum_spanning_tree(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         minimum_spanning_tree(graph, weight_fn: float)
///
///     to cast the edge object as a float as the weight.
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
///
/// :returns: A set of nodes in one half of the spanning tree
///
#[pyfunction]
#[pyo3(text_signature = "(spanning_tree, pop, target_pop, epsilon)")]
pub fn balanced_cut_edge(
    _py: Python,
    spanning_tree: &graph::PyGraph,
    py_pops: &PyList,
    py_pop_target: &PyFloat,
    py_epsilon: &PyFloat,
) -> PyResult<Vec<(usize, Vec<usize>)>> {
    let epsilon = py_epsilon.value();
    let pop_target = py_pop_target.value();

    let mut pops: Vec<f64> = vec![]; // not sure if the conversions are needed
    for i in 0..py_pops.len() {
        pops.push(py_pops.get_item(i).unwrap().extract::<f64>().unwrap());
    }

    let mut node_queue: VecDeque<NodeIndex> = VecDeque::<NodeIndex>::new();
    for leaf_node in spanning_tree.graph.node_indices() {
        // todo: filter expr
        if spanning_tree.graph.neighbors(leaf_node).count() == 1 {
            node_queue.push_back(leaf_node);
        }
    }

    // eprintln!("leaf nodes: {}", node_queue.len());

    // this process can be multithreaded, if the locking overhead isn't too high
    // (note: locking may not even be needed given the invariants this is assumed to maintain)
    let mut balanced_nodes: Vec<(usize, Vec<usize>)> = vec![];
    let mut same_partition_tracker: Vec<Vec<usize>> =
        vec![vec![]; spanning_tree.graph.node_count()]; // keeps track of all all the nodes on the same side of the partition
    let mut seen_nodes: Vec<bool> =
        vec![false; spanning_tree.graph.node_count()]; // todo: perf test this
    while node_queue.len() > 0 {
        let node = node_queue.pop_front().unwrap();
        let pop = pops[node.index()];

        // todo: factor out expensive clones
        // Mark as seen; push to queue if only one unseen neighbor
        let unseen_neighbors: Vec<NodeIndex> = spanning_tree
            .graph
            .neighbors(node)
            .filter(|node| !seen_nodes[node.index()])
            .collect();
        if unseen_neighbors.len() == 1 {
            // this may be false if root
            let neighbor = unseen_neighbors[0];
            pops[neighbor.index()] += pop.clone();
            same_partition_tracker[node.index()].push(node.index());
            same_partition_tracker[neighbor.index()] =
                same_partition_tracker[node.index()].clone();
            // eprintln!("node pushed to queue (pop = {}): {}", pops[neighbor.index()], neighbor.index());

            if !node_queue.contains(&neighbor) {
                node_queue.push_back(neighbor);
            }
        }
        pops[node.index()] = 0.0;

        // Check if balanced
        if pop >= pop_target * (1.0 - epsilon)
            && pop <= pop_target * (1.0 + epsilon)
        {
            // slightly different
            // eprintln!("balanced node found: {}", node.index());
            balanced_nodes.push((
                node.index(),
                same_partition_tracker[node.index()].clone(),
            ));
        }

        seen_nodes[node.index()] = true;
    }

    Ok(balanced_nodes)
}

/// Bipartition graph into two contiguous, population-balanced components.
/// Assumes that graph is contiguous.
///
/// :param PyGraph graph: Undirected graph
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         minimum_spanning_tree(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         minimum_spanning_tree(graph, weight_fn: float)
///
///     to cast the edge object as a float as the weight.
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
///
/// :returns: A set of nodes in one half of the spanning tree
///
#[pyfunction]
#[pyo3(text_signature = "(graph, weight_fn, pop, target_pop, epsilon)")]
pub fn bipartition_tree(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: PyObject,
    py_pops: &PyList,
    py_pop_target: &PyFloat,
    py_epsilon: &PyFloat,
) -> PyResult<Vec<(usize, Vec<usize>)>> {
    let mut balanced_nodes: Vec<(usize, Vec<usize>)> = vec![];

    while balanced_nodes.len() == 0 {
        let mst =
            minimum_spanning_tree(py, graph, Some(weight_fn.clone()), 1.0)
                .unwrap();
        balanced_nodes =
            balanced_cut_edge(py, &mst, py_pops, py_pop_target, py_epsilon)
                .unwrap();
    }

    Ok(balanced_nodes)
}
