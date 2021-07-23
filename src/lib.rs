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

mod astar;
mod digraph;
mod dijkstra;
mod dot_utils;
mod generators;
mod graph;
mod isomorphism;
mod iterators;
mod k_shortest_path;
mod layout;
mod max_weight_matching;
mod union;

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap};

use hashbrown::{HashMap, HashSet};

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use pyo3::wrap_pymodule;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::EdgeReference;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{
    Bfs, Data, EdgeIndexable, GraphBase, GraphProp, IntoEdgeReferences,
    IntoNeighbors, IntoNodeIdentifiers, NodeCount, NodeIndexable, Reversed,
    VisitMap, Visitable,
};
use petgraph::EdgeType;

use ndarray::prelude::*;
use num_bigint::{BigUint, ToBigUint};
use num_traits::{Num, Zero};
use numpy::IntoPyArray;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::generators::PyInit_generators;
use crate::iterators::{
    AllPairsPathLengthMapping, AllPairsPathMapping, EdgeList, NodeIndices,
    NodesCountMapping, PathLengthMapping, PathMapping, Pos2DMapping,
    WeightedEdgeList,
};

trait NodesRemoved {
    fn nodes_removed(&self) -> bool;
}

fn longest_path<F, T>(
    graph: &digraph::PyDiGraph,
    mut weight_fn: F,
) -> PyResult<(Vec<usize>, T)>
where
    F: FnMut(usize, usize, &PyObject) -> PyResult<T>,
    T: Num + Zero + PartialOrd + Copy,
{
    let dag = &graph.graph;
    let mut path: Vec<usize> = Vec::new();
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::new_err("Sort encountered a cycle"))
        }
    };
    if nodes.is_empty() {
        return Ok((path, T::zero()));
    }
    let mut dist: HashMap<NodeIndex, (T, NodeIndex)> = HashMap::new();
    for node in nodes {
        let parents = dag.edges_directed(node, petgraph::Direction::Incoming);
        let mut us: Vec<(T, NodeIndex)> = Vec::new();
        for p_edge in parents {
            let p_node = p_edge.source();
            let weight: T = weight_fn(
                p_node.index(),
                p_edge.target().index(),
                p_edge.weight(),
            )?;
            let length = dist[&p_node].0 + weight;
            us.push((length, p_node));
        }
        let maxu: (T, NodeIndex) = if !us.is_empty() {
            *us.iter()
                .max_by(|a, b| {
                    let weight_a = a.0;
                    let weight_b = b.0;
                    weight_a.partial_cmp(&weight_b).unwrap()
                })
                .unwrap()
        } else {
            (T::zero(), node)
        };
        dist.insert(node, maxu);
    }
    let first = dist
        .keys()
        .max_by(|a, b| dist[a].partial_cmp(&dist[b]).unwrap())
        .unwrap();
    let mut v = *first;
    let mut u: Option<NodeIndex> = None;
    while match u {
        Some(u) => u != v,
        None => true,
    } {
        path.push(v.index());
        u = Some(v);
        v = dist[&v].1;
    }
    path.reverse();
    let path_weight = dist[first].0;
    Ok((path, path_weight))
}

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
fn dag_longest_path(
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
        nodes: longest_path(graph, edge_weight_callable)?.0,
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
fn dag_longest_path_length(
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
    let (_, path_weight) = longest_path(graph, edge_weight_callable)?;
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
fn dag_weighted_longest_path(
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
        nodes: longest_path(graph, edge_weight_callable)?.0,
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
fn dag_weighted_longest_path_length(
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
    let (_, path_weight) = longest_path(graph, edge_weight_callable)?;
    Ok(path_weight)
}

/// Find the number of weakly connected components in a DAG.
///
/// :param PyDiGraph graph: The graph to find the number of weakly connected
///     components on
///
/// :returns: The number of weakly connected components in the DAG
/// :rtype: int
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
fn number_weakly_connected_components(graph: &digraph::PyDiGraph) -> usize {
    algo::connected_components(graph)
}

/// Find the weakly connected components in a directed graph
///
/// :param PyDiGraph graph: The graph to find the weakly connected components
///     in
///
/// :returns: A list of sets where each set it a weakly connected component of
///     the graph
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn weakly_connected_components(
    graph: &digraph::PyDiGraph,
) -> Vec<BTreeSet<usize>> {
    let mut seen: HashSet<NodeIndex> =
        HashSet::with_capacity(graph.node_count());
    let mut out_vec: Vec<BTreeSet<usize>> = Vec::new();
    for node in graph.graph.node_indices() {
        if !seen.contains(&node) {
            // BFS node generator
            let mut component_set: BTreeSet<usize> = BTreeSet::new();
            let mut bfs_seen: HashSet<NodeIndex> = HashSet::new();
            let mut next_level: HashSet<NodeIndex> = HashSet::new();
            next_level.insert(node);
            while !next_level.is_empty() {
                let this_level = next_level;
                next_level = HashSet::new();
                for bfs_node in this_level {
                    if !bfs_seen.contains(&bfs_node) {
                        component_set.insert(bfs_node.index());
                        bfs_seen.insert(bfs_node);
                        for neighbor in
                            graph.graph.neighbors_undirected(bfs_node)
                        {
                            next_level.insert(neighbor);
                        }
                    }
                }
            }
            out_vec.push(component_set);
            seen.extend(bfs_seen);
        }
    }
    out_vec
}

/// Check if the graph is weakly connected
///
/// :param PyDiGraph graph: The graph to check if it is weakly connected
///
/// :returns: Whether the graph is weakly connected or not
/// :rtype: bool
///
/// :raises NullGraph: If an empty graph is passed in
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn is_weakly_connected(graph: &digraph::PyDiGraph) -> PyResult<bool> {
    if graph.graph.node_count() == 0 {
        return Err(NullGraph::new_err("Invalid operation on a NullGraph"));
    }
    Ok(weakly_connected_components(graph)[0].len() == graph.graph.node_count())
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
fn is_directed_acyclic_graph(graph: &digraph::PyDiGraph) -> bool {
    match algo::toposort(graph, None) {
        Ok(_nodes) => true,
        Err(_err) => false,
    }
}

/// Return a new PyDiGraph by forming a union from two input PyDiGraph objects
///
/// The algorithm in this function operates in three phases:
///
///  1. Add all the nodes from  ``second`` into ``first``. operates in O(n),
///     with n being number of nodes in `b`.
///  2. Merge nodes from ``second`` over ``first`` given that:
///
///     - The ``merge_nodes`` is ``True``. operates in O(n^2), with n being the
///       number of nodes in ``second``.
///     - The respective node in ``second`` and ``first`` share the same
///       weight/data payload.
///
///  3. Adds all the edges from ``second`` to ``first``. If the ``merge_edges``
///     parameter is ``True`` and the respective edge in ``second`` and
///     first`` share the same weight/data payload they will be merged
///     together.
///
///  :param PyDiGraph first: The first directed graph object
///  :param PyDiGraph second: The second directed graph object
///  :param bool merge_nodes: If set to ``True`` nodes will be merged between
///     ``second`` and ``first`` if the weights are equal.
///  :param bool merge_edges: If set to ``True`` edges will be merged between
///     ``second`` and ``first`` if the weights are equal.
///
///  :returns: A new PyDiGraph object that is the union of ``second`` and
///     ``first``. It's worth noting the weight/data payload objects are
///     passed by reference from ``first`` and ``second`` to this new object.
///  :rtype: PyDiGraph
#[pyfunction]
#[pyo3(text_signature = "(first, second, merge_nodes, merge_edges, /)")]
fn digraph_union(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    merge_nodes: bool,
    merge_edges: bool,
) -> PyResult<digraph::PyDiGraph> {
    let res =
        union::digraph_union(py, first, second, merge_nodes, merge_edges)?;
    Ok(res)
}

/// Determine if 2 directed graphs are isomorphic
///
/// This checks if 2 graphs are isomorphic both structurally and also
/// comparing the node data and edge data using the provided matcher functions.
/// The matcher function takes in 2 data objects and will compare them. A simple
/// example that checks if they're just equal would be::
///
///     graph_a = retworkx.PyDiGraph()
///     graph_b = retworkx.PyDiGraph()
///     retworkx.is_isomorphic(graph_a, graph_b,
///                            lambda x, y: x == y)
///
/// .. note::
///
///     For better performance on large graphs, consider setting `id_order=False`.
///
/// :param PyDiGraph first: The first graph to compare
/// :param PyDiGraph second: The second graph to compare
/// :param callable node_matcher: A python callable object that takes 2 positional
///     one for each node data object. If the return of this
///     function evaluates to True then the nodes passed to it are vieded
///     as matching.
/// :param callable edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are vieded
///     as matching.
/// :param bool id_order: If set to ``False`` this function will use a
///     heuristic matching order based on [VF2]_ paper. Otherwise it will
///     default to matching the nodes in order specified by their ids.
///
/// :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
///     not.
/// :rtype: bool
#[pyfunction(id_order = "true")]
#[pyo3(
    text_signature = "(first, second, node_matcher=None, edge_matcher=None, id_order=True, /)"
)]
fn digraph_is_isomorphic(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
) -> PyResult<bool> {
    let compare_nodes = node_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let compare_edges = edge_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let res = isomorphism::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        compare_nodes,
        compare_edges,
        id_order,
        Ordering::Equal,
        true,
    )?;
    Ok(res)
}

/// Determine if 2 undirected graphs are isomorphic
///
/// This checks if 2 graphs are isomorphic both structurally and also
/// comparing the node data and edge data using the provided matcher functions.
/// The matcher function takes in 2 data objects and will compare them. A simple
/// example that checks if they're just equal would be::
///
///     graph_a = retworkx.PyGraph()
///     graph_b = retworkx.PyGraph()
///     retworkx.is_isomorphic(graph_a, graph_b,
///                            lambda x, y: x == y)
///
/// .. note::
///
///     For better performance on large graphs, consider setting `id_order=False`.
///
/// :param PyGraph first: The first graph to compare
/// :param PyGraph second: The second graph to compare
/// :param callable node_matcher: A python callable object that takes 2 positional
///     one for each node data object. If the return of this
///     function evaluates to True then the nodes passed to it are vieded
///     as matching.
/// :param callable edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are vieded
///     as matching.
/// :param bool (default=True) id_order:  If set to true, the algorithm matches the
///     nodes in order specified by their ids. Otherwise, it uses a heuristic
///     matching order based in [VF2]_ paper.
///
/// :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
///     not.
/// :rtype: bool
#[pyfunction(id_order = "true")]
#[pyo3(
    text_signature = "(first, second, node_matcher=None, edge_matcher=None, id_order=True, /)"
)]
fn graph_is_isomorphic(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
) -> PyResult<bool> {
    let compare_nodes = node_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let compare_edges = edge_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let res = isomorphism::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        compare_nodes,
        compare_edges,
        id_order,
        Ordering::Equal,
        true,
    )?;
    Ok(res)
}

/// Determine if 2 directed graphs are subgraph - isomorphic
///
/// This checks if 2 graphs are subgraph isomorphic both structurally and also
/// comparing the node data and edge data using the provided matcher functions.
/// The matcher function takes in 2 data objects and will compare them.
/// Since there is an ambiguity in the term 'subgraph', do note that we check
/// for an node-induced subgraph if argument `induced` is set to `True`. If it is
/// set to `False`, we check for a non induced subgraph, meaning the second graph
/// can have fewer edges than the subgraph of the first. By default it's `True`. A
/// simple example that checks if they're just equal would be::
///
///     graph_a = retworkx.PyDiGraph()
///     graph_b = retworkx.PyDiGraph()
///     retworkx.is_subgraph_isomorphic(graph_a, graph_b,
///                                     lambda x, y: x == y)
///
/// .. note::
///
///     For better performance on large graphs, consider setting `id_order=False`.
///
/// :param PyDiGraph first: The first graph to compare
/// :param PyDiGraph second: The second graph to compare
/// :param callable node_matcher: A python callable object that takes 2 positional
///     one for each node data object. If the return of this
///     function evaluates to True then the nodes passed to it are vieded
///     as matching.
/// :param callable edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are vieded
///     as matching.
/// :param bool id_order: If set to ``True`` this function will match the nodes
///     in order specified by their ids. Otherwise it will default to a heuristic
///     matching order based on [VF2]_ paper.
/// :param bool induced: If set to ``True`` this function will check the existence
///     of a node-induced subgraph of first isomorphic to second graph.
///     Default: ``True``.
///
/// :returns: ``True`` if there is a subgraph of `first` isomorphic to `second`,
///     ``False`` if there is not.
/// :rtype: bool
#[pyfunction(id_order = "false", induced = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None, id_order=False, induced=True)"
)]
fn digraph_is_subgraph_isomorphic(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    induced: bool,
) -> PyResult<bool> {
    let compare_nodes = node_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let compare_edges = edge_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let res = isomorphism::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        compare_nodes,
        compare_edges,
        id_order,
        Ordering::Greater,
        induced,
    )?;
    Ok(res)
}

/// Determine if 2 undirected graphs are subgraph - isomorphic
///
/// This checks if 2 graphs are subgraph isomorphic both structurally and also
/// comparing the node data and edge data using the provided matcher functions.
/// The matcher function takes in 2 data objects and will compare them.
/// Since there is an ambiguity in the term 'subgraph', do note that we check
/// for an node-induced subgraph if argument `induced` is set to `True`. If it is
/// set to `False`, we check for a non induced subgraph, meaning the second graph
/// can have fewer edges than the subgraph of the first. By default it's `True`. A
/// simple example that checks if they're just equal would be::
///
///     graph_a = retworkx.PyGraph()
///     graph_b = retworkx.PyGraph()
///     retworkx.is_subgraph_isomorphic(graph_a, graph_b,
///                                     lambda x, y: x == y)
///
/// .. note::
///
///     For better performance on large graphs, consider setting `id_order=False`.
///
/// :param PyGraph first: The first graph to compare
/// :param PyGraph second: The second graph to compare
/// :param callable node_matcher: A python callable object that takes 2 positional
///     one for each node data object. If the return of this
///     function evaluates to True then the nodes passed to it are vieded
///     as matching.
/// :param callable edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are vieded
///     as matching.
/// :param bool id_order: If set to ``True`` this function will match the nodes
///     in order specified by their ids. Otherwise it will default to a heuristic
///     matching order based on [VF2]_ paper.
/// :param bool induced: If set to ``True`` this function will check the existence
///     of a node-induced subgraph of first isomorphic to second graph.
///     Default: ``True``.
///
/// :returns: ``True`` if there is a subgraph of `first` isomorphic to `second`,
///     ``False`` if there is not.
/// :rtype: bool
#[pyfunction(id_order = "false", induced = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None, id_order=False, induced=True)"
)]
fn graph_is_subgraph_isomorphic(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    induced: bool,
) -> PyResult<bool> {
    let compare_nodes = node_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let compare_edges = edge_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let res = isomorphism::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        compare_nodes,
        compare_edges,
        id_order,
        Ordering::Greater,
        induced,
    )?;
    Ok(res)
}

/// Return the topological sort of node indexes from the provided graph
///
/// :param PyDiGraph graph: The DAG to get the topological sort on
///
/// :returns: A list of node indices topologically sorted.
/// :rtype: NodeIndices
///
/// :raises DAGHasCycle: if a cycle is encountered while sorting the graph
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
fn topological_sort(graph: &digraph::PyDiGraph) -> PyResult<NodeIndices> {
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::new_err("Sort encountered a cycle"))
        }
    };
    Ok(NodeIndices {
        nodes: nodes.iter().map(|node| node.index()).collect(),
    })
}

fn dfs_edges<G>(
    graph: G,
    source: Option<usize>,
    edge_count: usize,
) -> Vec<(usize, usize)>
where
    G: GraphBase<NodeId = NodeIndex>
        + IntoNodeIdentifiers
        + NodeIndexable
        + IntoNeighbors
        + NodeCount
        + Visitable,
    <G as Visitable>::Map: VisitMap<NodeIndex>,
{
    let nodes: Vec<NodeIndex> = match source {
        Some(start) => vec![NodeIndex::new(start)],
        None => graph
            .node_identifiers()
            .map(|ind| NodeIndex::new(graph.to_index(ind)))
            .collect(),
    };
    let node_count = graph.node_count();
    let mut visited: HashSet<NodeIndex> = HashSet::with_capacity(node_count);
    let mut out_vec: Vec<(usize, usize)> = Vec::with_capacity(edge_count);
    for start in nodes {
        if visited.contains(&start) {
            continue;
        }
        visited.insert(start);
        let mut children: Vec<NodeIndex> = graph.neighbors(start).collect();
        children.reverse();
        let mut stack: Vec<(NodeIndex, Vec<NodeIndex>)> =
            vec![(start, children)];
        // Used to track the last position in children vec across iterations
        let mut index_map: HashMap<NodeIndex, usize> =
            HashMap::with_capacity(node_count);
        index_map.insert(start, 0);
        while !stack.is_empty() {
            let temp_parent = stack.last().unwrap();
            let parent = temp_parent.0;
            let children = temp_parent.1.clone();
            let count = *index_map.get(&parent).unwrap();
            let mut found = false;
            let mut index = count;
            for child in &children[index..] {
                index += 1;
                if !visited.contains(child) {
                    out_vec.push((parent.index(), child.index()));
                    visited.insert(*child);
                    let mut grandchildren: Vec<NodeIndex> =
                        graph.neighbors(*child).collect();
                    grandchildren.reverse();
                    stack.push((*child, grandchildren));
                    index_map.insert(*child, 0);
                    *index_map.get_mut(&parent).unwrap() = index;
                    found = true;
                    break;
                }
            }
            if !found || children.is_empty() {
                stack.pop();
            }
        }
    }
    out_vec
}

/// Get edge list in depth first order
///
/// :param PyDiGraph graph: The graph to get the DFS edge list from
/// :param int source: An optional node index to use as the starting node
///     for the depth-first search. The edge list will only return edges in
///     the components reachable from this index. If this is not specified
///     then a source will be chosen arbitrarly and repeated until all
///     components of the graph are searched.
///
/// :returns: A list of edges as a tuple of the form ``(source, target)`` in
///     depth-first order
/// :rtype: EdgeList
#[pyfunction]
#[pyo3(text_signature = "(graph, /, source=None)")]
fn digraph_dfs_edges(
    graph: &digraph::PyDiGraph,
    source: Option<usize>,
) -> EdgeList {
    EdgeList {
        edges: dfs_edges(graph, source, graph.graph.edge_count()),
    }
}

/// Get edge list in depth first order
///
/// :param PyGraph graph: The graph to get the DFS edge list from
/// :param int source: An optional node index to use as the starting node
///     for the depth-first search. The edge list will only return edges in
///     the components reachable from this index. If this is not specified
///     then a source will be chosen arbitrarly and repeated until all
///     components of the graph are searched.
///
/// :returns: A list of edges as a tuple of the form ``(source, target)`` in
///     depth-first order
/// :rtype: EdgeList
#[pyfunction]
#[pyo3(text_signature = "(graph, /, source=None)")]
fn graph_dfs_edges(graph: &graph::PyGraph, source: Option<usize>) -> EdgeList {
    EdgeList {
        edges: dfs_edges(graph, source, graph.graph.edge_count()),
    }
}

/// Return successors in a breadth-first-search from a source node.
///
/// The return format is ``[(Parent Node, [Children Nodes])]`` in a bfs order
/// from the source node provided.
///
/// :param PyDiGraph graph: The DAG to get the bfs_successors from
/// :param int node: The index of the dag node to get the bfs successors for
///
/// :returns: A list of nodes's data and their children in bfs order. The
///     BFSSuccessors class that is returned is a custom container class that
///     implements the sequence protocol. This can be used as a python list
///     with index based access.
/// :rtype: BFSSuccessors
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
fn bfs_successors(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
) -> iterators::BFSSuccessors {
    let index = NodeIndex::new(node);
    let mut bfs = Bfs::new(graph, index);
    let mut out_list: Vec<(PyObject, Vec<PyObject>)> =
        Vec::with_capacity(graph.node_count());
    while let Some(nx) = bfs.next(graph) {
        let children = graph
            .graph
            .neighbors_directed(nx, petgraph::Direction::Outgoing);
        let mut succesors: Vec<PyObject> = Vec::new();
        for succ in children {
            succesors
                .push(graph.graph.node_weight(succ).unwrap().clone_ref(py));
        }
        if !succesors.is_empty() {
            out_list.push((
                graph.graph.node_weight(nx).unwrap().clone_ref(py),
                succesors,
            ));
        }
    }
    iterators::BFSSuccessors {
        bfs_successors: out_list,
    }
}

/// Return the ancestors of a node in a graph.
///
/// This differs from :meth:`PyDiGraph.predecessors` method  in that
/// ``predecessors`` returns only nodes with a direct edge into the provided
/// node. While this function returns all nodes that have a path into the
/// provided node.
///
/// :param PyDiGraph graph: The graph to get the descendants from
/// :param int node: The index of the graph node to get the ancestors for
///
/// :returns: A list of node indexes of ancestors of provided node.
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
fn ancestors(graph: &digraph::PyDiGraph, node: usize) -> HashSet<usize> {
    let index = NodeIndex::new(node);
    let mut out_set: HashSet<usize> = HashSet::new();
    let reverse_graph = Reversed(graph);
    let res = algo::dijkstra(reverse_graph, index, None, |_| 1);
    for n in res.keys() {
        let n_int = n.index();
        out_set.insert(n_int);
    }
    out_set.remove(&node);
    out_set
}

/// Return the descendants of a node in a graph.
///
/// This differs from :meth:`PyDiGraph.successors` method in that
/// ``successors``` returns only nodes with a direct edge out of the provided
/// node. While this function returns all nodes that have a path from the
/// provided node.
///
/// :param PyDiGraph graph: The graph to get the descendants from
/// :param int node: The index of the graph node to get the descendants for
///
/// :returns: A list of node indexes of descendants of provided node.
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
fn descendants(graph: &digraph::PyDiGraph, node: usize) -> HashSet<usize> {
    let index = NodeIndex::new(node);
    let mut out_set: HashSet<usize> = HashSet::new();
    let res = algo::dijkstra(graph, index, None, |_| 1);
    for n in res.keys() {
        let n_int = n.index();
        out_set.insert(n_int);
    }
    out_set.remove(&node);
    out_set
}

/// Get the lexicographical topological sorted nodes from the provided DAG
///
/// This function returns a list of nodes data in a graph lexicographically
/// topologically sorted using the provided key function.
///
/// :param PyDiGraph dag: The DAG to get the topological sorted nodes from
/// :param callable key: key is a python function or other callable that
///     gets passed a single argument the node data from the graph and is
///     expected to return a string which will be used for sorting.
///
/// :returns: A list of node's data lexicographically topologically sorted.
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(dag, key, /)")]
fn lexicographical_topological_sort(
    py: Python,
    dag: &digraph::PyDiGraph,
    key: PyObject,
) -> PyResult<PyObject> {
    let key_callable = |a: &PyObject| -> PyResult<PyObject> {
        let res = key.call1(py, (a,))?;
        Ok(res.to_object(py))
    };
    // HashMap of node_index indegree
    let node_count = dag.node_count();
    let mut in_degree_map: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(node_count);
    for node in dag.graph.node_indices() {
        in_degree_map.insert(node, dag.in_degree(node.index()));
    }

    #[derive(Clone, Eq, PartialEq)]
    struct State {
        key: String,
        node: NodeIndex,
    }

    impl Ord for State {
        fn cmp(&self, other: &State) -> Ordering {
            // Notice that the we flip the ordering on costs.
            // In case of a tie we compare positions - this step is necessary
            // to make implementations of `PartialEq` and `Ord` consistent.
            other
                .key
                .cmp(&self.key)
                .then_with(|| other.node.index().cmp(&self.node.index()))
        }
    }

    // `PartialOrd` needs to be implemented as well.
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &State) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut zero_indegree = BinaryHeap::with_capacity(node_count);
    for (node, degree) in in_degree_map.iter() {
        if *degree == 0 {
            let map_key_raw = key_callable(&dag.graph[*node])?;
            let map_key: String = map_key_raw.extract(py)?;
            zero_indegree.push(State {
                key: map_key,
                node: *node,
            });
        }
    }
    let mut out_list: Vec<&PyObject> = Vec::with_capacity(node_count);
    let dir = petgraph::Direction::Outgoing;
    while let Some(State { node, .. }) = zero_indegree.pop() {
        let neighbors = dag.graph.neighbors_directed(node, dir);
        for child in neighbors {
            let child_degree = in_degree_map.get_mut(&child).unwrap();
            *child_degree -= 1;
            if *child_degree == 0 {
                let map_key_raw = key_callable(&dag.graph[child])?;
                let map_key: String = map_key_raw.extract(py)?;
                zero_indegree.push(State {
                    key: map_key,
                    node: child,
                });
                in_degree_map.remove(&child);
            }
        }
        out_list.push(&dag.graph[node])
    }
    Ok(PyList::new(py, out_list).into())
}

/// Color a PyGraph using a largest_first strategy greedy graph coloring.
///
/// :param PyGraph: The input PyGraph object to color
///
/// :returns: A dictionary where keys are node indices and the value is
///     the color
/// :rtype: dict
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
fn graph_greedy_color(
    py: Python,
    graph: &graph::PyGraph,
) -> PyResult<PyObject> {
    let mut colors: HashMap<usize, usize> = HashMap::new();
    let mut node_vec: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let mut sort_map: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(graph.node_count());
    for k in node_vec.iter() {
        sort_map.insert(*k, graph.graph.edges(*k).count());
    }
    node_vec.par_sort_by_key(|k| Reverse(sort_map.get(k)));
    for u_index in node_vec {
        let mut neighbor_colors: HashSet<usize> = HashSet::new();
        for edge in graph.graph.edges(u_index) {
            let target = edge.target().index();
            let existing_color = match colors.get(&target) {
                Some(node) => node,
                None => continue,
            };
            neighbor_colors.insert(*existing_color);
        }
        let mut count: usize = 0;
        loop {
            if !neighbor_colors.contains(&count) {
                break;
            }
            count += 1;
        }
        colors.insert(u_index.index(), count);
    }
    let out_dict = PyDict::new(py);
    for (index, color) in colors {
        out_dict.set_item(index, color)?;
    }
    Ok(out_dict.into())
}

/// Compute the length of the kth shortest path
///
/// Computes the lengths of the kth shortest path from ``start`` to every
/// reachable node.
///
/// Computes in :math:`O(k * (|E| + |V|*log(|V|)))` time (average).
///
/// :param PyGraph graph: The graph to find the shortest paths in
/// :param int start: The node index to find the shortest paths from
/// :param int k: The kth shortest path to find the lengths of
/// :param edge_cost: A python callable that will receive an edge payload and
///     return a float for the cost of that eedge
/// :param int goal: An optional goal node index, if specified the output
///     dictionary
///
/// :returns: A dict of lengths where the key is the destination node index and
///     the value is the length of the path.
/// :rtype: PathLengthMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, start, k, edge_cost, /, goal=None)")]
fn digraph_k_shortest_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    start: usize,
    k: usize,
    edge_cost: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let out_goal = goal.map(NodeIndex::new);
    let edge_cost_callable = |edge: &PyObject| -> PyResult<f64> {
        let res = edge_cost.call1(py, (edge,))?;
        res.extract(py)
    };

    let out_map = k_shortest_path::k_shortest_path(
        graph,
        NodeIndex::new(start),
        out_goal,
        k,
        edge_cost_callable,
    )?;
    Ok(PathLengthMapping {
        path_lengths: out_map
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if goal.is_some() && goal.unwrap() != k_int {
                    None
                } else {
                    Some((k_int, *v))
                }
            })
            .collect(),
    })
}

/// Compute the length of the kth shortest path
///
/// Computes the lengths of the kth shortest path from ``start`` to every
/// reachable node.
///
/// Computes in :math:`O(k * (|E| + |V|*log(|V|)))` time (average).
///
/// :param PyGraph graph: The graph to find the shortest paths in
/// :param int start: The node index to find the shortest paths from
/// :param int k: The kth shortest path to find the lengths of
/// :param edge_cost: A python callable that will receive an edge payload and
///     return a float for the cost of that eedge
/// :param int goal: An optional goal node index, if specified the output
///     dictionary
///
/// :returns: A dict of lengths where the key is the destination node index and
///     the value is the length of the path.
/// :rtype: PathLengthMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, start, k, edge_cost, /, goal=None)")]
fn graph_k_shortest_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    start: usize,
    k: usize,
    edge_cost: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let out_goal = goal.map(NodeIndex::new);
    let edge_cost_callable = |edge: &PyObject| -> PyResult<f64> {
        let res = edge_cost.call1(py, (edge,))?;
        res.extract(py)
    };

    let out_map = k_shortest_path::k_shortest_path(
        graph,
        NodeIndex::new(start),
        out_goal,
        k,
        edge_cost_callable,
    )?;
    Ok(PathLengthMapping {
        path_lengths: out_map
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if goal.is_some() && goal.unwrap() != k_int {
                    None
                } else {
                    Some((k_int, *v))
                }
            })
            .collect(),
    })
}

fn _floyd_warshall<Ty: EdgeType>(
    py: Python,
    graph: &StableGraph<PyObject, PyObject, Ty>,
    weight_fn: Option<PyObject>,
    as_undirected: bool,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<AllPairsPathLengthMapping> {
    if graph.node_count() == 0 {
        return Ok(AllPairsPathLengthMapping {
            path_lengths: HashMap::new(),
        });
    } else if graph.edge_count() == 0 {
        return Ok(AllPairsPathLengthMapping {
            path_lengths: graph
                .node_indices()
                .map(|i| {
                    (
                        i.index(),
                        PathLengthMapping {
                            path_lengths: HashMap::new(),
                        },
                    )
                })
                .collect(),
        });
    }
    let n = graph.node_bound();

    // Allocate empty matrix
    let mut mat: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];

    // Set diagonal to 0
    for i in 0..n {
        if let Some(row_i) = mat.get_mut(i) {
            row_i.entry(i).or_insert(0.0);
        }
    }

    // Utility to set row_i[j] = min(row_i[j], m_ij)
    macro_rules! insert_or_minimize {
        ($row_i: expr, $j: expr, $m_ij: expr) => {{
            $row_i
                .entry($j)
                .and_modify(|e| {
                    if $m_ij < *e {
                        *e = $m_ij;
                    }
                })
                .or_insert($m_ij);
        }};
    }

    // Build adjacency matrix
    for edge in graph.edge_references() {
        let i = NodeIndexable::to_index(&graph, edge.source());
        let j = NodeIndexable::to_index(&graph, edge.target());
        let weight = edge.weight().clone();

        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
        if let Some(row_i) = mat.get_mut(i) {
            insert_or_minimize!(row_i, j, edge_weight);
        }
        if as_undirected {
            if let Some(row_j) = mat.get_mut(j) {
                insert_or_minimize!(row_j, i, edge_weight);
            }
        }
    }

    // Perform the Floyd-Warshall algorithm.
    // In each loop, this finds the shortest path from point i
    // to point j using intermediate nodes 0..k
    if n < parallel_threshold {
        for k in 0..n {
            let row_k = mat.get(k).cloned().unwrap_or_default();
            mat.iter_mut().for_each(|row_i| {
                if let Some(m_ik) = row_i.get(&k).cloned() {
                    for (j, m_kj) in row_k.iter() {
                        let m_ikj = m_ik + *m_kj;
                        insert_or_minimize!(row_i, *j, m_ikj);
                    }
                }
            })
        }
    } else {
        for k in 0..n {
            let row_k = mat.get(k).cloned().unwrap_or_default();
            mat.par_iter_mut().for_each(|row_i| {
                if let Some(m_ik) = row_i.get(&k).cloned() {
                    for (j, m_kj) in row_k.iter() {
                        let m_ikj = m_ik + *m_kj;
                        insert_or_minimize!(row_i, *j, m_ikj);
                    }
                }
            })
        }
    }

    // Convert to return format
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();

    let out_map: HashMap<usize, PathLengthMapping> = node_indices
        .into_iter()
        .map(|i| {
            let out_map = PathLengthMapping {
                path_lengths: mat[i.index()]
                    .iter()
                    .map(|(k, v)| (*k, *v))
                    .collect(),
            };
            (i.index(), out_map)
        })
        .collect();
    Ok(AllPairsPathLengthMapping {
        path_lengths: out_map,
    })
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// This function is multithreaded and will launch a pool with threads equal
/// to the number of CPUs by default if the number of nodes in the graph is
/// above the value of ``parallel_threshold`` (it defaults to 300).
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads if parallelization was enabled.
///
/// :param PyDiGraph graph: The directed graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         digraph_floyd_warshall(graph, weight_fn= lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         digraph_floyd_warshall(graph, weight_fn=float)
///
///     to cast the edge object as a float as the weight.
/// :param as_undirected: If set to true each directed edge will be treated as
///     bidirectional/undirected.
/// :param int parallel_threshold: The number of nodes to execute
///     the algorithm in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {0: 0.0, 1: 2.0, 2: 2.0},
///             1: {1: 0.0, 2: 1.0},
///             2: {0: 1.0, 2: 0.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
#[pyfunction(
    parallel_threshold = "300",
    as_undirected = "false",
    default_weight = "1.0"
)]
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, as_undirected=False, default_weight=1.0, parallel_threshold=300)"
)]
fn digraph_floyd_warshall(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    as_undirected: bool,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<AllPairsPathLengthMapping> {
    _floyd_warshall(
        py,
        &graph.graph,
        weight_fn,
        as_undirected,
        default_weight,
        parallel_threshold,
    )
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// This function is multithreaded and will launch a pool with threads equal
/// to the number of CPUs by default if the number of nodes in the graph is
/// above the value of ``parallel_threshold`` (it defaults to 300).
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads if parallelization was enabled.
///
/// :param PyGraph graph: The graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_floyd_warshall(graph, weight_fn= lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_floyd_warshall(graph, weight_fn=float)
///
///     to cast the edge object as a float as the weight.
/// :param int parallel_threshold: The number of nodes to execute
///     the algorithm in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {0: 0.0, 1: 2.0, 2: 2.0},
///             1: {1: 0.0, 2: 1.0},
///             2: {0: 1.0, 2: 0.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
#[pyfunction(parallel_threshold = "300", default_weight = "1.0")]
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, default_weight=1.0, parallel_threshold=300)"
)]
fn graph_floyd_warshall(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<AllPairsPathLengthMapping> {
    let as_undirected = true;
    _floyd_warshall(
        py,
        &graph.graph,
        weight_fn,
        as_undirected,
        default_weight,
        parallel_threshold,
    )
}

fn get_edge_iter_with_weights<G>(
    graph: G,
) -> impl Iterator<Item = (usize, usize, PyObject)>
where
    G: GraphBase
        + IntoEdgeReferences
        + IntoNodeIdentifiers
        + NodeIndexable
        + NodeCount
        + GraphProp
        + NodesRemoved,
    G: Data<NodeWeight = PyObject, EdgeWeight = PyObject>,
{
    let node_map: Option<HashMap<NodeIndex, usize>> = if graph.nodes_removed() {
        let mut node_hash_map: HashMap<NodeIndex, usize> =
            HashMap::with_capacity(graph.node_count());
        for (count, node) in graph.node_identifiers().enumerate() {
            let index = NodeIndex::new(graph.to_index(node));
            node_hash_map.insert(index, count);
        }
        Some(node_hash_map)
    } else {
        None
    };

    graph.edge_references().map(move |edge| {
        let i: usize;
        let j: usize;
        match &node_map {
            Some(map) => {
                let source_index =
                    NodeIndex::new(graph.to_index(edge.source()));
                let target_index =
                    NodeIndex::new(graph.to_index(edge.target()));
                i = *map.get(&source_index).unwrap();
                j = *map.get(&target_index).unwrap();
            }
            None => {
                i = graph.to_index(edge.source());
                j = graph.to_index(edge.target());
            }
        }
        (i, j, edge.weight().clone())
    })
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// This function is multithreaded and will launch a pool with threads equal
/// to the number of CPUs by default if the number of nodes in the graph is
/// above the value of ``parallel_threshold`` (it defaults to 300).
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads if parallelization was enabled.
///
/// :param PyGraph graph: The graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight.
/// :param int parallel_threshold: The number of nodes to execute
///     the algorithm in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :returns: A matrix of shortest path distances between nodes. If there is no
///     path between two nodes then the corresponding matrix entry will be
///     ``np.inf``.
/// :rtype: numpy.ndarray
#[pyfunction(parallel_threshold = "300", default_weight = "1.0")]
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, default_weight=1.0, parallel_threshold=300)"
)]
fn graph_floyd_warshall_numpy(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    // Allocate empty matrix
    let mut mat = Array2::<f64>::from_elem((n, n), std::f64::INFINITY);

    // Build adjacency matrix
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
        mat[[i, j]] = mat[[i, j]].min(edge_weight);
        mat[[j, i]] = mat[[j, i]].min(edge_weight);
    }

    // 0 out the diagonal
    for x in mat.diag_mut() {
        *x = 0.0;
    }
    // Perform the Floyd-Warshall algorithm.
    // In each loop, this finds the shortest path from point i
    // to point j using intermediate nodes 0..k
    if n < parallel_threshold {
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let d_ijk = mat[[i, k]] + mat[[k, j]];
                    if d_ijk < mat[[i, j]] {
                        mat[[i, j]] = d_ijk;
                    }
                }
            }
        }
    } else {
        for k in 0..n {
            let row_k = mat.slice(s![k, ..]).to_owned();
            mat.axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row_i| {
                    let m_ik = row_i[k];
                    row_i.iter_mut().zip(row_k.iter()).for_each(
                        |(m_ij, m_kj)| {
                            let d_ijk = m_ik + *m_kj;
                            if d_ijk < *m_ij {
                                *m_ij = d_ijk;
                            }
                        },
                    )
                })
        }
    }
    Ok(mat.into_pyarray(py).into())
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// This function is multithreaded and will launch a pool with threads equal
/// to the number of CPUs by default if the number of nodes in the graph is
/// above the value of ``parallel_threshold`` (it defaults to 300).
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads if parallelization was enabled.
///
/// :param PyDiGraph graph: The directed graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight.
/// :param as_undirected: If set to true each directed edge will be treated as
///     bidirectional/undirected.
/// :param int parallel_threshold: The number of nodes to execute
///     the algorithm in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :returns: A matrix of shortest path distances between nodes. If there is no
///     path between two nodes then the corresponding matrix entry will be
///     ``np.inf``.
/// :rtype: numpy.ndarray
#[pyfunction(
    parallel_threshold = "300",
    as_undirected = "false",
    default_weight = "1.0"
)]
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, as_undirected=False, default_weight=1.0, parallel_threshold=300)"
)]
fn digraph_floyd_warshall_numpy(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    as_undirected: bool,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<PyObject> {
    let n = graph.node_count();

    // Allocate empty matrix
    let mut mat = Array2::<f64>::from_elem((n, n), std::f64::INFINITY);

    // Build adjacency matrix
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
        mat[[i, j]] = mat[[i, j]].min(edge_weight);
        if as_undirected {
            mat[[j, i]] = mat[[j, i]].min(edge_weight);
        }
    }
    // 0 out the diagonal
    for x in mat.diag_mut() {
        *x = 0.0;
    }
    // Perform the Floyd-Warshall algorithm.
    // In each loop, this finds the shortest path from point i
    // to point j using intermediate nodes 0..k
    if n < parallel_threshold {
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let d_ijk = mat[[i, k]] + mat[[k, j]];
                    if d_ijk < mat[[i, j]] {
                        mat[[i, j]] = d_ijk;
                    }
                }
            }
        }
    } else {
        for k in 0..n {
            let row_k = mat.slice(s![k, ..]).to_owned();
            mat.axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row_i| {
                    let m_ik = row_i[k];
                    row_i.iter_mut().zip(row_k.iter()).for_each(
                        |(m_ij, m_kj)| {
                            let d_ijk = m_ik + *m_kj;
                            if d_ijk < *m_ij {
                                *m_ij = d_ijk;
                            }
                        },
                    )
                })
        }
    }
    Ok(mat.into_pyarray(py).into())
}

/// Collect runs that match a filter function
///
/// A run is a path of nodes where there is only a single successor and all
/// nodes in the path match the given condition. Each node in the graph can
/// appear in only a single run.
///
/// :param PyDiGraph graph: The graph to find runs in
/// :param filter_fn: The filter function to use for matching nodes. It takes
///     in one argument, the node data payload/weight object, and will return a
///     boolean whether the node matches the conditions or not. If it returns
///     ``False`` it will skip that node.
///
/// :returns: a list of runs, where each run is a list of node data
///     payload/weight for the nodes in the run
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, filter)")]
fn collect_runs(
    py: Python,
    graph: &digraph::PyDiGraph,
    filter_fn: PyObject,
) -> PyResult<Vec<Vec<PyObject>>> {
    let mut out_list: Vec<Vec<PyObject>> = Vec::new();
    let mut seen: HashSet<NodeIndex> =
        HashSet::with_capacity(graph.node_count());

    let filter_node = |node: &PyObject| -> PyResult<bool> {
        let res = filter_fn.call1(py, (node,))?;
        res.extract(py)
    };

    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => {
            return Err(DAGHasCycle::new_err("Sort encountered a cycle"))
        }
    };
    for node in nodes {
        if !filter_node(&graph.graph[node])? || seen.contains(&node) {
            continue;
        }
        seen.insert(node);
        let mut group: Vec<PyObject> = vec![graph.graph[node].clone_ref(py)];
        let mut successors: Vec<NodeIndex> = graph
            .graph
            .neighbors_directed(node, petgraph::Direction::Outgoing)
            .collect();
        successors.dedup();

        while successors.len() == 1
            && filter_node(&graph.graph[successors[0]])?
            && !seen.contains(&successors[0])
        {
            group.push(graph.graph[successors[0]].clone_ref(py));
            seen.insert(successors[0]);
            successors = graph
                .graph
                .neighbors_directed(
                    successors[0],
                    petgraph::Direction::Outgoing,
                )
                .collect();
            successors.dedup();
        }
        if !group.is_empty() {
            out_list.push(group);
        }
    }
    Ok(out_list)
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
fn layers(
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

/// Get the distance matrix for a directed graph
///
/// This differs from functions like digraph_floyd_warshall_numpy in that the
/// edge weight/data payload is not used and each edge is treated as a
/// distance of 1.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold`` (it
/// defaults to 300). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// :param PyDiGraph graph: The graph to get the distance matrix for
/// :param int parallel_threshold: The number of nodes to calculate the
///     the distance matrix in parallel at. It defaults to 300, but this can
///     be tuned
/// :param bool as_undirected: If set to ``True`` the input directed graph
///     will be treat as if each edge was bidirectional/undirected in the
///     output distance matrix.
///
/// :returns: The distance matrix
/// :rtype: numpy.ndarray
#[pyfunction(parallel_threshold = "300", as_undirected = "false")]
#[pyo3(
    text_signature = "(graph, /, parallel_threshold=300, as_undirected=False)"
)]
pub fn digraph_distance_matrix(
    py: Python,
    graph: &digraph::PyDiGraph,
    parallel_threshold: usize,
    as_undirected: bool,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    let bfs_traversal = |index: usize, mut row: ArrayViewMut1<f64>| {
        let mut seen: HashMap<NodeIndex, usize> = HashMap::with_capacity(n);
        let start_index = NodeIndex::new(index);
        let mut level = 0;
        let mut next_level: HashSet<NodeIndex> = HashSet::new();
        next_level.insert(start_index);
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<NodeIndex> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    row[[v.index()]] = level as f64;
                }
            }
            if seen.len() == n {
                return;
            }
            for node in found {
                for v in graph
                    .graph
                    .neighbors_directed(node, petgraph::Direction::Outgoing)
                {
                    next_level.insert(v);
                }
                if as_undirected {
                    for v in graph
                        .graph
                        .neighbors_directed(node, petgraph::Direction::Incoming)
                    {
                        next_level.insert(v);
                    }
                }
            }
            level += 1
        }
    };
    if n < parallel_threshold {
        matrix
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    } else {
        // Parallelize by row and iterate from each row index in BFS order
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Get the distance matrix for an undirected graph
///
/// This differs from functions like digraph_floyd_warshall_numpy in that the
/// edge weight/data payload is not used and each edge is treated as a
/// distance of 1.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``paralllel_threshold`` (it
/// defaults to 300). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// :param PyGraph graph: The graph to get the distance matrix for
/// :param int parallel_threshold: The number of nodes to calculate the
///     the distance matrix in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :returns: The distance matrix
/// :rtype: numpy.ndarray
#[pyfunction(parallel_threshold = "300")]
#[pyo3(text_signature = "(graph, /, parallel_threshold=300)")]
pub fn graph_distance_matrix(
    py: Python,
    graph: &graph::PyGraph,
    parallel_threshold: usize,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    let bfs_traversal = |index: usize, mut row: ArrayViewMut1<f64>| {
        let mut seen: HashMap<NodeIndex, usize> = HashMap::with_capacity(n);
        let start_index = NodeIndex::new(index);
        let mut level = 0;
        let mut next_level: HashSet<NodeIndex> = HashSet::new();
        next_level.insert(start_index);
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<NodeIndex> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    row[[v.index()]] = level as f64;
                }
            }
            if seen.len() == n {
                return;
            }
            for node in found {
                for v in graph.graph.neighbors(node) {
                    next_level.insert(v);
                }
            }
            level += 1
        }
    };
    if n < parallel_threshold {
        matrix
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    } else {
        // Parallelize by row and iterate from each row index in BFS order
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Return the adjacency matrix for a PyDiGraph object
///
/// In the case where there are multiple edges between nodes the value in the
/// output matrix will be the sum of the edges' weights.
///
/// :param PyDiGraph graph: The DiGraph used to generate the adjacency matrix
///     from
/// :param callable weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         dag_adjacency_matrix(dag, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         dag_adjacency_matrix(dag, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight. If this is not
///     specified a default value (either ``default_weight`` or 1) will be used
///     for all edges.
/// :param float default_weight: If ``weight_fn`` is not used this can be
///     optionally used to specify a default weight to use for all edges.
///
///  :return: The adjacency matrix for the input dag as a numpy array
///  :rtype: numpy.ndarray
#[pyfunction(default_weight = "1.0")]
#[pyo3(text_signature = "(graph, /, weight_fn=None, default_weight=1.0)")]
fn digraph_adjacency_matrix(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
        matrix[[i, j]] += edge_weight;
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Return the adjacency matrix for a PyGraph class
///
/// In the case where there are multiple edges between nodes the value in the
/// output matrix will be the sum of the edges' weights.
///
/// :param PyGraph graph: The graph used to generate the adjacency matrix from
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells retworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_adjacency_matrix(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_adjacency_matrix(graph, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight. If this is not
///     specified a default value (either ``default_weight`` or 1) will be used
///     for all edges.
/// :param float default_weight: If ``weight_fn`` is not used this can be
///     optionally used to specify a default weight to use for all edges.
///
/// :return: The adjacency matrix for the input dag as a numpy array
/// :rtype: numpy.ndarray
#[pyfunction(default_weight = "1.0")]
#[pyo3(text_signature = "(graph, /, weight_fn=None, default_weight=1.0)")]
fn graph_adjacency_matrix(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
        matrix[[i, j]] += edge_weight;
        matrix[[j, i]] += edge_weight;
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Return all simple paths between 2 nodes in a PyGraph object
///
/// A simple path is a path with no repeated nodes.
///
/// :param PyGraph graph: The graph to find the path in
/// :param int from: The node index to find the paths from
/// :param int to: The node index to find the paths to
/// :param int min_depth: The minimum depth of the path to include in the output
///     list of paths. By default all paths are included regardless of depth,
///     setting to 0 will behave like the default.
/// :param int cutoff: The maximum depth of path to include in the output list
///     of paths. By default includes all paths regardless of depth, setting to
///     0 will behave like default.
///
/// :returns: A list of lists where each inner list is a path of node indices
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, from, to, /, min=None, cutoff=None)")]
fn graph_all_simple_paths(
    graph: &graph::PyGraph,
    from: usize,
    to: usize,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let from_index = NodeIndex::new(from);
    if !graph.graph.contains_node(from_index) {
        return Err(InvalidNode::new_err(
            "The input index for 'from' is not a valid node index",
        ));
    }
    let to_index = NodeIndex::new(to);
    if !graph.graph.contains_node(to_index) {
        return Err(InvalidNode::new_err(
            "The input index for 'to' is not a valid node index",
        ));
    }
    let min_intermediate_nodes: usize = match min_depth {
        Some(depth) => depth - 2,
        None => 0,
    };
    let cutoff_petgraph: Option<usize> = cutoff.map(|depth| depth - 2);
    let result: Vec<Vec<usize>> = algo::all_simple_paths(
        graph,
        from_index,
        to_index,
        min_intermediate_nodes,
        cutoff_petgraph,
    )
    .map(|v: Vec<NodeIndex>| v.into_iter().map(|i| i.index()).collect())
    .collect();
    Ok(result)
}

/// Return all simple paths between 2 nodes in a PyDiGraph object
///
/// A simple path is a path with no repeated nodes.
///
/// :param PyDiGraph graph: The graph to find the path in
/// :param int from: The node index to find the paths from
/// :param int to: The node index to find the paths to
/// :param int min_depth: The minimum depth of the path to include in the output
///     list of paths. By default all paths are included regardless of depth,
///     sett to 0 will behave like the default.
/// :param int cutoff: The maximum depth of path to include in the output list
///     of paths. By default includes all paths regardless of depth, setting to
///     0 will behave like default.
///
/// :returns: A list of lists where each inner list is a path
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, from, to, /, min_depth=None, cutoff=None)")]
fn digraph_all_simple_paths(
    graph: &digraph::PyDiGraph,
    from: usize,
    to: usize,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let from_index = NodeIndex::new(from);
    if !graph.graph.contains_node(from_index) {
        return Err(InvalidNode::new_err(
            "The input index for 'from' is not a valid node index",
        ));
    }
    let to_index = NodeIndex::new(to);
    if !graph.graph.contains_node(to_index) {
        return Err(InvalidNode::new_err(
            "The input index for 'to' is not a valid node index",
        ));
    }
    let min_intermediate_nodes: usize = match min_depth {
        Some(depth) => depth - 2,
        None => 0,
    };
    let cutoff_petgraph: Option<usize> = cutoff.map(|depth| depth - 2);
    let result: Vec<Vec<usize>> = algo::all_simple_paths(
        graph,
        from_index,
        to_index,
        min_intermediate_nodes,
        cutoff_petgraph,
    )
    .map(|v: Vec<NodeIndex>| v.into_iter().map(|i| i.index()).collect())
    .collect();
    Ok(result)
}

fn weight_callable(
    py: Python,
    weight_fn: &Option<PyObject>,
    weight: &PyObject,
    default: f64,
) -> PyResult<f64> {
    match weight_fn {
        Some(weight_fn) => {
            let res = weight_fn.call1(py, (weight,))?;
            res.extract(py)
        }
        None => Ok(default),
    }
}

/// Find the shortest path from a node
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param PyGraph graph:
/// :param int source: The node index to find paths from
/// :param int target: An optional target to find a path to
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
/// :param bool as_undirected: If set to true the graph will be treated as
///     undirected for finding the shortest path.
///
/// :return: Dictionary of paths. The keys are destination node indices and
///     the dict values are lists of node indices making the path.
/// :rtype: dict
#[pyfunction(default_weight = "1.0", as_undirected = "false")]
#[pyo3(
    text_signature = "(graph, source, /, target=None weight_fn=None, default_weight=1.0)"
)]
pub fn graph_dijkstra_shortest_paths(
    py: Python,
    graph: &graph::PyGraph,
    source: usize,
    target: Option<usize>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<PathMapping> {
    let start = NodeIndex::new(source);
    let goal_index: Option<NodeIndex> = target.map(NodeIndex::new);
    let mut paths: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(graph.node_count());
    dijkstra::dijkstra(
        graph,
        start,
        goal_index,
        |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
        Some(&mut paths),
    )?;

    Ok(PathMapping {
        paths: paths
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == source
                    || target.is_some() && target.unwrap() != k_int
                {
                    None
                } else {
                    Some((
                        k.index(),
                        v.iter().map(|x| x.index()).collect::<Vec<usize>>(),
                    ))
                }
            })
            .collect(),
    })
}

/// Find the shortest path from a node
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param PyDiGraph graph:
/// :param int source: The node index to find paths from
/// :param int target: An optional target path to find the path
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
/// :param bool as_undirected: If set to true the graph will be treated as
///     undirected for finding the shortest path.
///
/// :return: Dictionary of paths. The keys are destination node indices and
///     the dict values are lists of node indices making the path.
/// :rtype: dict
#[pyfunction(default_weight = "1.0", as_undirected = "false")]
#[pyo3(
    text_signature = "(graph, source, /, target=None weight_fn=None, default_weight=1.0, as_undirected=False)"
)]
pub fn digraph_dijkstra_shortest_paths(
    py: Python,
    graph: &digraph::PyDiGraph,
    source: usize,
    target: Option<usize>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    as_undirected: bool,
) -> PyResult<PathMapping> {
    let start = NodeIndex::new(source);
    let goal_index: Option<NodeIndex> = target.map(NodeIndex::new);
    let mut paths: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(graph.node_count());
    if as_undirected {
        dijkstra::dijkstra(
            // TODO: Use petgraph undirected adapter after
            // https://github.com/petgraph/petgraph/pull/318 is available in
            // a petgraph release.
            &graph.to_undirected(py, true, None)?,
            start,
            goal_index,
            |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
            Some(&mut paths),
        )?;
    } else {
        dijkstra::dijkstra(
            graph,
            start,
            goal_index,
            |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
            Some(&mut paths),
        )?;
    }
    Ok(PathMapping {
        paths: paths
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == source
                    || target.is_some() && target.unwrap() != k_int
                {
                    None
                } else {
                    Some((
                        k_int,
                        v.iter().map(|x| x.index()).collect::<Vec<usize>>(),
                    ))
                }
            })
            .collect(),
    })
}

/// Compute the lengths of the shortest paths for a PyGraph object using
/// Dijkstra's algorithm
///
/// :param PyGraph graph: The input graph to use
/// :param int node: The node index to use as the source for finding the
///     shortest paths from
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the
///     cost/weight of that edge. It must be non-negative
/// :param int goal: An optional node index to use as the end of the path.
///     When specified the traversal will stop when the goal is reached and
///     the output dictionary will only have a single entry with the length
///     of the shortest path to the goal node.
///
/// :returns: A dictionary of the shortest paths from the provided node where
///     the key is the node index of the end of the path and the value is the
///     cost/sum of the weights of path
/// :rtype: PathLengthMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, node, edge_cost_fn, /, goal=None)")]
fn graph_dijkstra_shortest_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        raw.extract(py)
    };

    let start = NodeIndex::new(node);
    let goal_index: Option<NodeIndex> = goal.map(NodeIndex::new);

    let res = dijkstra::dijkstra(
        graph,
        start,
        goal_index,
        |e| edge_cost_callable(e.weight()),
        None,
    )?;
    Ok(PathLengthMapping {
        path_lengths: res
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == node || goal.is_some() && goal.unwrap() != k_int {
                    None
                } else {
                    Some((k_int, *v))
                }
            })
            .collect(),
    })
}

/// Compute the lengths of the shortest paths for a PyDiGraph object using
/// Dijkstra's algorithm
///
/// :param PyDiGraph graph: The input graph to use
/// :param int node: The node index to use as the source for finding the
///     shortest paths from
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the
///     cost/weight of that edge. It must be non-negative
/// :param int goal: An optional node index to use as the end of the path.
///     When specified the traversal will stop when the goal is reached and
///     the output dictionary will only have a single entry with the length
///     of the shortest path to the goal node.
///
/// :returns: A dictionary of the shortest paths from the provided node where
///     the key is the node index of the end of the path and the value is the
///     cost/sum of the weights of path
/// :rtype: PathLengthMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, node, edge_cost_fn, /, goal=None)")]
fn digraph_dijkstra_shortest_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        raw.extract(py)
    };

    let start = NodeIndex::new(node);
    let goal_index: Option<NodeIndex> = goal.map(NodeIndex::new);

    let res = dijkstra::dijkstra(
        graph,
        start,
        goal_index,
        |e| edge_cost_callable(e.weight()),
        None,
    )?;
    Ok(PathLengthMapping {
        path_lengths: res
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == node || goal.is_some() && goal.unwrap() != k_int {
                    None
                } else {
                    Some((k_int, *v))
                }
            })
            .collect(),
    })
}

fn _all_pairs_dijkstra_path_lengths<Ty: EdgeType + Sync>(
    py: Python,
    graph: &StableGraph<PyObject, PyObject, Ty>,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    if graph.node_count() == 0 {
        return Ok(AllPairsPathLengthMapping {
            path_lengths: HashMap::new(),
        });
    } else if graph.edge_count() == 0 {
        return Ok(AllPairsPathLengthMapping {
            path_lengths: graph
                .node_indices()
                .map(|i| {
                    (
                        i.index(),
                        PathLengthMapping {
                            path_lengths: HashMap::new(),
                        },
                    )
                })
                .collect(),
        });
    }
    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        raw.extract(py)
    };
    let mut edge_weights: Vec<Option<f64>> =
        Vec::with_capacity(graph.edge_bound());
    for index in 0..=graph.edge_bound() {
        let raw_weight = graph.edge_weight(EdgeIndex::new(index));
        match raw_weight {
            Some(weight) => {
                edge_weights.push(Some(edge_cost_callable(weight)?))
            }
            None => edge_weights.push(None),
        };
    }
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    let out_map: HashMap<usize, PathLengthMapping> = node_indices
        .into_par_iter()
        .map(|x| {
            let out_map = PathLengthMapping {
                path_lengths: dijkstra::dijkstra(
                    graph,
                    x,
                    None,
                    |e| edge_cost(e.id()),
                    None,
                )
                .unwrap()
                .iter()
                .filter_map(|(index, cost)| {
                    if *index == x {
                        None
                    } else {
                        Some((index.index(), *cost))
                    }
                })
                .collect(),
            };
            (x.index(), out_map)
        })
        .collect();
    Ok(AllPairsPathLengthMapping {
        path_lengths: out_map,
    })
}

fn _all_pairs_dijkstra_shortest_paths<Ty: EdgeType + Sync>(
    py: Python,
    graph: &StableGraph<PyObject, PyObject, Ty>,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    if graph.node_count() == 0 {
        return Ok(AllPairsPathMapping {
            paths: HashMap::new(),
        });
    } else if graph.edge_count() == 0 {
        return Ok(AllPairsPathMapping {
            paths: graph
                .node_indices()
                .map(|i| {
                    (
                        i.index(),
                        PathMapping {
                            paths: HashMap::new(),
                        },
                    )
                })
                .collect(),
        });
    }
    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        raw.extract(py)
    };
    let mut edge_weights: Vec<Option<f64>> =
        Vec::with_capacity(graph.edge_bound());
    for index in 0..=graph.edge_bound() {
        let raw_weight = graph.edge_weight(EdgeIndex::new(index));
        match raw_weight {
            Some(weight) => {
                edge_weights.push(Some(edge_cost_callable(weight)?))
            }
            None => edge_weights.push(None),
        };
    }
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    Ok(AllPairsPathMapping {
        paths: node_indices
            .into_par_iter()
            .map(|x| {
                let mut paths: HashMap<NodeIndex, Vec<NodeIndex>> =
                    HashMap::with_capacity(graph.node_count());
                dijkstra::dijkstra(
                    graph,
                    x,
                    None,
                    |e| edge_cost(e.id()),
                    Some(&mut paths),
                )
                .unwrap();
                let index = x.index();
                let out_paths = PathMapping {
                    paths: paths
                        .iter()
                        .filter_map(|path_mapping| {
                            let path_index = path_mapping.0.index();
                            if index != path_index {
                                Some((
                                    path_index,
                                    path_mapping
                                        .1
                                        .iter()
                                        .map(|x| x.index())
                                        .collect(),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect(),
                };
                (index, out_paths)
            })
            .collect(),
    })
}

/// Calculate the the shortest length from all nodes in a
/// :class:`~retworkx.PyDiGraph` object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm. This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// :param graph: The input :class:`~retworkx.PyDiGraph` to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {1: 2.0, 2: 2.0},
///             1: {2: 1.0},
///             2: {0: 1.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn digraph_all_pairs_dijkstra_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    _all_pairs_dijkstra_path_lengths(py, &graph.graph, edge_cost_fn)
}

/// Find the shortest path from all nodes in a :class:`~retworkx.PyDiGraph`
/// object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm. This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// :param graph: The input :class:`~retworkx.PyDiGraph` object to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of paths. The keys are source node indices
///     and the values are dicts of the target node and the list of the
///     node indices making up the shortest path to that node. For example::
///
///         {
///             0: {1: [0, 1],  2: [0, 1, 2]},
///             1: {2: [1, 2]},
///             2: {0: [2, 0]},
///         }
///
/// :rtype: AllPairsPathMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn digraph_all_pairs_dijkstra_shortest_paths(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    _all_pairs_dijkstra_shortest_paths(py, &graph.graph, edge_cost_fn)
}

/// Calculate the the shortest length from all nodes in a
/// :class:`~retworkx.PyGraph` object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param graph: The input :class:`~retworkx.PyGraph` to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {1: 2.0, 2: 2.0},
///             1: {2: 1.0},
///             2: {0: 1.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn graph_all_pairs_dijkstra_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    _all_pairs_dijkstra_path_lengths(py, &graph.graph, edge_cost_fn)
}

/// Find the shortest path from all nodes in a :class:`~retworkx.PyGraph`
/// object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param graph: The input :class:`~retworkx.PyGraph` object to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of paths. The keys are destination node
///     indices and the values are dicts of the target node and the list of the
///     node indices making up the shortest path to that node. For example::
///
///         {
///             0: {1: [0, 1],  2: [0, 1, 2]},
///             1: {2: [1, 2]},
///             2: {0: [2, 0]},
///         }
///
/// :rtype: AllPairsPathMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn graph_all_pairs_dijkstra_shortest_paths(
    py: Python,
    graph: &graph::PyGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    _all_pairs_dijkstra_shortest_paths(py, &graph.graph, edge_cost_fn)
}

/// Compute the A* shortest path for a PyGraph
///
/// :param PyGraph graph: The input graph to use
/// :param int node: The node index to compute the path from
/// :param goal_fn: A python callable that will take in 1 parameter, a node's data
///     object and will return a boolean which will be True if it is the finish
///     node.
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an edge's
///     data object and will return a float that represents the cost of that
///     edge. It must be non-negative.
/// :param estimate_cost_fn: A python callable that will take in 1 parameter, a
///     node's data object and will return a float which represents the estimated
///     cost for the next node. The return must be non-negative. For the
///     algorithm to find the actual shortest path, it should be admissible,
///     meaning that it should never overestimate the actual cost to get to the
///     nearest goal node.
///
/// :returns: The computed shortest path between node and finish as a list
///     of node indices.
/// :rtype: NodeIndices
#[pyfunction]
#[pyo3(text_signature = "(graph, node, goal_fn, edge_cost, estimate_cost, /)")]
fn graph_astar_shortest_path(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<NodeIndices> {
    let goal_fn_callable = |a: &PyObject| -> PyResult<bool> {
        let res = goal_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: bool = raw.extract(py)?;
        Ok(output)
    };

    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };

    let estimate_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = estimate_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };
    let start = NodeIndex::new(node);

    let astar_res = astar::astar(
        graph,
        start,
        |f| goal_fn_callable(graph.graph.node_weight(f).unwrap()),
        |e| edge_cost_callable(e.weight()),
        |estimate| {
            estimate_cost_callable(graph.graph.node_weight(estimate).unwrap())
        },
    )?;
    let path = match astar_res {
        Some(path) => path,
        None => {
            return Err(NoPathFound::new_err(
                "No path found that satisfies goal_fn",
            ))
        }
    };
    Ok(NodeIndices {
        nodes: path.1.into_iter().map(|x| x.index()).collect(),
    })
}

/// Compute the A* shortest path for a PyDiGraph
///
/// :param PyDiGraph graph: The input graph to use
/// :param int node: The node index to compute the path from
/// :param goal_fn: A python callable that will take in 1 parameter, a node's
///     data object and will return a boolean which will be True if it is the
///     finish node.
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the cost of
///     that edge. It must be non-negative.
/// :param estimate_cost_fn: A python callable that will take in 1 parameter, a
///     node's data object and will return a float which represents the
///     estimated cost for the next node. The return must be non-negative. For
///     the algorithm to find the actual shortest path, it should be
///     admissible, meaning that it should never overestimate the actual cost
///     to get to the nearest goal node.
///
/// :return: The computed shortest path between node and finish as a list
///     of node indices.
/// :rtype: NodeIndices
#[pyfunction]
#[pyo3(text_signature = "(graph, node, goal_fn, edge_cost, estimate_cost, /)")]
fn digraph_astar_shortest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<NodeIndices> {
    let goal_fn_callable = |a: &PyObject| -> PyResult<bool> {
        let res = goal_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: bool = raw.extract(py)?;
        Ok(output)
    };

    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };

    let estimate_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = estimate_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };
    let start = NodeIndex::new(node);

    let astar_res = astar::astar(
        graph,
        start,
        |f| goal_fn_callable(graph.graph.node_weight(f).unwrap()),
        |e| edge_cost_callable(e.weight()),
        |estimate| {
            estimate_cost_callable(graph.graph.node_weight(estimate).unwrap())
        },
    )?;
    let path = match astar_res {
        Some(path) => path,
        None => {
            return Err(NoPathFound::new_err(
                "No path found that satisfies goal_fn",
            ))
        }
    };
    Ok(NodeIndices {
        nodes: path.1.into_iter().map(|x| x.index()).collect(),
    })
}

/// Return a :math:`G_{np}` directed random graph, also known as an
/// Erdős-Rényi graph or a binomial graph.
///
/// For number of nodes :math:`n` and probability :math:`p`, the :math:`G_{n,p}`
/// graph algorithm creates :math:`n` nodes, and for all the :math:`n (n - 1)` possible edges,
/// each edge is created independently with probability :math:`p`.
/// In general, for any probability :math:`p`, the expected number of edges returned
/// is :math:`m = p n (n - 1)`. If :math:`p = 0` or :math:`p = 1`, the returned
/// graph is not random and will always be an empty or a complete graph respectively.
/// An empty graph has zero edges and a complete directed graph has :math:`n (n - 1)` edges.
/// The run time is :math:`O(n + m)` where :math:`m` is the expected number of edges mentioned above.
/// When :math:`p = 0`, run time always reduces to :math:`O(n)`, as the lower bound.
/// When :math:`p = 1`, run time always goes to :math:`O(n + n (n - 1))`, as the upper bound.
/// For other probabilities, this algorithm [1]_ runs in :math:`O(n + m)` time.
///
/// For :math:`0 < p < 1`, the algorithm is based on the implementation of the networkx function
/// ``fast_gnp_random_graph`` [2]_
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param float probability: The probability of creating an edge between two nodes
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyDiGraph object
/// :rtype: PyDiGraph
///
/// .. [1] Vladimir Batagelj and Ulrik Brandes,
///    "Efficient generation of large random networks",
///    Phys. Rev. E, 71, 036113, 2005.
/// .. [2] https://github.com/networkx/networkx/blob/networkx-2.4/networkx/generators/random_graphs.py#L49-L120
#[pyfunction]
#[pyo3(text_signature = "(num_nodes, probability, seed=None, /)")]
pub fn directed_gnp_random_graph(
    py: Python,
    num_nodes: isize,
    probability: f64,
    seed: Option<u64>,
) -> PyResult<digraph::PyDiGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableDiGraph::<PyObject, PyObject>::new();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(PyValueError::new_err(
            "Probability out of range, must be 0 <= p <= 1",
        ));
    }
    if probability > 0.0 {
        if (probability - 1.0).abs() < std::f64::EPSILON {
            for u in 0..num_nodes {
                for v in 0..num_nodes {
                    if u != v {
                        // exclude self-loops
                        let u_index = NodeIndex::new(u as usize);
                        let v_index = NodeIndex::new(v as usize);
                        inner_graph.add_edge(u_index, v_index, py.None());
                    }
                }
            }
        } else {
            let mut v: isize = 0;
            let mut w: isize = -1;
            let lp: f64 = (1.0 - probability).ln();

            let between = Uniform::new(0.0, 1.0);
            while v < num_nodes {
                let random: f64 = between.sample(&mut rng);
                let lr: f64 = (1.0 - random).ln();
                let ratio: isize = (lr / lp) as isize;
                w = w + 1 + ratio;
                // avoid self loops
                if v == w {
                    w += 1;
                }
                while v < num_nodes && num_nodes <= w {
                    w -= v;
                    v += 1;
                    // avoid self loops
                    if v == w {
                        w -= v;
                        v += 1;
                    }
                }
                if v < num_nodes {
                    let v_index = NodeIndex::new(v as usize);
                    let w_index = NodeIndex::new(w as usize);
                    inner_graph.add_edge(v_index, w_index, py.None());
                }
            }
        }
    }

    let graph = digraph::PyDiGraph {
        graph: inner_graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
    };
    Ok(graph)
}

/// Return a :math:`G_{np}` random undirected graph, also known as an
/// Erdős-Rényi graph or a binomial graph.
///
/// For number of nodes :math:`n` and probability :math:`p`, the :math:`G_{n,p}`
/// graph algorithm creates :math:`n` nodes, and for all the :math:`n (n - 1)/2` possible edges,
/// each edge is created independently with probability :math:`p`.
/// In general, for any probability :math:`p`, the expected number of edges returned
/// is :math:`m = p n (n - 1)/2`. If :math:`p = 0` or :math:`p = 1`, the returned
/// graph is not random and will always be an empty or a complete graph respectively.
/// An empty graph has zero edges and a complete undirected graph has :math:`n (n - 1)/2` edges.
/// The run time is :math:`O(n + m)` where :math:`m` is the expected number of edges mentioned above.
/// When :math:`p = 0`, run time always reduces to :math:`O(n)`, as the lower bound.
/// When :math:`p = 1`, run time always goes to :math:`O(n + n (n - 1)/2)`, as the upper bound.
/// For other probabilities, this algorithm [1]_ runs in :math:`O(n + m)` time.
///
/// For :math:`0 < p < 1`, the algorithm is based on the implementation of the networkx function
/// ``fast_gnp_random_graph`` [2]_
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param float probability: The probability of creating an edge between two nodes
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyGraph object
/// :rtype: PyGraph
///
/// .. [1] Vladimir Batagelj and Ulrik Brandes,
///    "Efficient generation of large random networks",
///    Phys. Rev. E, 71, 036113, 2005.
/// .. [2] https://github.com/networkx/networkx/blob/networkx-2.4/networkx/generators/random_graphs.py#L49-L120
#[pyfunction]
#[pyo3(text_signature = "(num_nodes, probability, seed=None, /)")]
pub fn undirected_gnp_random_graph(
    py: Python,
    num_nodes: isize,
    probability: f64,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableUnGraph::<PyObject, PyObject>::default();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(PyValueError::new_err(
            "Probability out of range, must be 0 <= p <= 1",
        ));
    }
    if probability > 0.0 {
        if (probability - 1.0).abs() < std::f64::EPSILON {
            for u in 0..num_nodes {
                for v in u + 1..num_nodes {
                    let u_index = NodeIndex::new(u as usize);
                    let v_index = NodeIndex::new(v as usize);
                    inner_graph.add_edge(u_index, v_index, py.None());
                }
            }
        } else {
            let mut v: isize = 1;
            let mut w: isize = -1;
            let lp: f64 = (1.0 - probability).ln();

            let between = Uniform::new(0.0, 1.0);
            while v < num_nodes {
                let random: f64 = between.sample(&mut rng);
                let lr = (1.0 - random).ln();
                let ratio: isize = (lr / lp) as isize;
                w = w + 1 + ratio;
                while w >= v && v < num_nodes {
                    w -= v;
                    v += 1;
                }
                if v < num_nodes {
                    let v_index = NodeIndex::new(v as usize);
                    let w_index = NodeIndex::new(w as usize);
                    inner_graph.add_edge(v_index, w_index, py.None());
                }
            }
        }
    }

    let graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
        multigraph: true,
    };
    Ok(graph)
}

/// Return a :math:`G_{nm}` of a directed graph
///
/// Generates a random directed graph out of all the possible graphs with :math:`n` nodes and
/// :math:`m` edges. The generated graph will not be a multigraph and will not have self loops.
///
/// For :math:`n` nodes, the maximum edges that can be returned is :math:`n (n - 1)`.
/// Passing :math:`m` higher than that will still return the maximum number of edges.
/// If :math:`m = 0`, the returned graph will always be empty (no edges).
/// When a seed is provided, the results are reproducible. Passing a seed when :math:`m = 0`
/// or :math:`m >= n (n - 1)` has no effect, as the result will always be an empty or a complete graph respectively.
///
/// This algorithm has a time complexity of :math:`O(n + m)`
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param int num_edges: The number of edges to create in the graph
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyDiGraph object
/// :rtype: PyDiGraph
///
#[pyfunction]
#[pyo3(text_signature = "(num_nodes, num_edges, seed=None, /)")]
pub fn directed_gnm_random_graph(
    py: Python,
    num_nodes: isize,
    num_edges: isize,
    seed: Option<u64>,
) -> PyResult<digraph::PyDiGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    if num_edges < 0 {
        return Err(PyValueError::new_err("num_edges must be >= 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableDiGraph::<PyObject, PyObject>::new();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    // if number of edges to be created is >= max,
    // avoid randomly missed trials and directly add edges between every node
    if num_edges >= num_nodes * (num_nodes - 1) {
        for u in 0..num_nodes {
            for v in 0..num_nodes {
                // avoid self-loops
                if u != v {
                    let u_index = NodeIndex::new(u as usize);
                    let v_index = NodeIndex::new(v as usize);
                    inner_graph.add_edge(u_index, v_index, py.None());
                }
            }
        }
    } else {
        let mut created_edges: isize = 0;
        let between = Uniform::new(0, num_nodes);
        while created_edges < num_edges {
            let u = between.sample(&mut rng);
            let v = between.sample(&mut rng);
            let u_index = NodeIndex::new(u as usize);
            let v_index = NodeIndex::new(v as usize);
            // avoid self-loops and multi-graphs
            if u != v && inner_graph.find_edge(u_index, v_index).is_none() {
                inner_graph.add_edge(u_index, v_index, py.None());
                created_edges += 1;
            }
        }
    }
    let graph = digraph::PyDiGraph {
        graph: inner_graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
    };
    Ok(graph)
}

/// Return a :math:`G_{nm}` of an undirected graph
///
/// Generates a random undirected graph out of all the possible graphs with :math:`n` nodes and
/// :math:`m` edges. The generated graph will not be a multigraph and will not have self loops.
///
/// For :math:`n` nodes, the maximum edges that can be returned is :math:`n (n - 1)/2`.
/// Passing :math:`m` higher than that will still return the maximum number of edges.
/// If :math:`m = 0`, the returned graph will always be empty (no edges).
/// When a seed is provided, the results are reproducible. Passing a seed when :math:`m = 0`
/// or :math:`m >= n (n - 1)/2` has no effect, as the result will always be an empty or a complete graph respectively.
///
/// This algorithm has a time complexity of :math:`O(n + m)`
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param int num_edges: The number of edges to create in the graph
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyGraph object
/// :rtype: PyGraph

#[pyfunction]
#[pyo3(text_signature = "(num_nodes, probability, seed=None, /)")]
pub fn undirected_gnm_random_graph(
    py: Python,
    num_nodes: isize,
    num_edges: isize,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    if num_edges < 0 {
        return Err(PyValueError::new_err("num_edges must be >= 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StableUnGraph::<PyObject, PyObject>::default();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    // if number of edges to be created is >= max,
    // avoid randomly missed trials and directly add edges between every node
    if num_edges >= num_nodes * (num_nodes - 1) / 2 {
        for u in 0..num_nodes {
            for v in u + 1..num_nodes {
                let u_index = NodeIndex::new(u as usize);
                let v_index = NodeIndex::new(v as usize);
                inner_graph.add_edge(u_index, v_index, py.None());
            }
        }
    } else {
        let mut created_edges: isize = 0;
        let between = Uniform::new(0, num_nodes);
        while created_edges < num_edges {
            let u = between.sample(&mut rng);
            let v = between.sample(&mut rng);
            let u_index = NodeIndex::new(u as usize);
            let v_index = NodeIndex::new(v as usize);
            // avoid self-loops and multi-graphs
            if u != v && inner_graph.find_edge(u_index, v_index).is_none() {
                inner_graph.add_edge(u_index, v_index, py.None());
                created_edges += 1;
            }
        }
    }
    let graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
        multigraph: true,
    };
    Ok(graph)
}

#[inline]
fn pnorm(x: f64, p: f64) -> f64 {
    if p == 1.0 || p == std::f64::INFINITY {
        x.abs()
    } else if p == 2.0 {
        x * x
    } else {
        x.abs().powf(p)
    }
}

fn distance(x: &[f64], y: &[f64], p: f64) -> f64 {
    let it = x.iter().zip(y.iter()).map(|(xi, yi)| pnorm(xi - yi, p));

    if p == std::f64::INFINITY {
        it.fold(-1.0, |max, x| if x > max { x } else { max })
    } else {
        it.sum()
    }
}

/// Returns a random geometric graph in the unit cube of dimensions `dim`.
///
/// The random geometric graph model places `num_nodes` nodes uniformly at
/// random in the unit cube. Two nodes are joined by an edge if the
/// distance between the nodes is at most `radius`.
///
/// Each node has a node attribute ``'pos'`` that stores the
/// position of that node in Euclidean space as provided by the
/// ``pos`` keyword argument or, if ``pos`` was not provided, as
/// generated by this function.
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param float radius: Distance threshold value
/// :param int dim: Dimension of node positions. Default: 2
/// :param list pos: Optional list with node positions as values
/// :param float p: Which Minkowski distance metric to use.  `p` has to meet the condition
///     ``1 <= p <= infinity``.
///     If this argument is not specified, the :math:`L^2` metric
///     (the Euclidean distance metric), p = 2 is used.
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyGraph object
/// :rtype: PyGraph
#[pyfunction(dim = "2", p = "2.0")]
#[pyo3(
    text_signature = "(num_nodes, radius, /, dim=2, pos=None, p=2.0, seed=None)"
)]
pub fn random_geometric_graph(
    py: Python,
    num_nodes: usize,
    radius: f64,
    dim: usize,
    pos: Option<Vec<Vec<f64>>>,
    p: f64,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    if num_nodes == 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }

    let mut inner_graph = StableUnGraph::<PyObject, PyObject>::default();

    let radius_p = pnorm(radius, p);
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };

    let dist = Uniform::new(0.0, 1.0);
    let pos = pos.unwrap_or_else(|| {
        (0..num_nodes)
            .map(|_| (0..dim).map(|_| dist.sample(&mut rng)).collect())
            .collect()
    });

    if num_nodes != pos.len() {
        return Err(PyValueError::new_err(
            "number of elements in pos and num_nodes must be equal",
        ));
    }

    for pval in pos.iter() {
        let pos_dict = PyDict::new(py);
        pos_dict.set_item("pos", pval.to_object(py))?;

        inner_graph.add_node(pos_dict.into());
    }

    for u in 0..(num_nodes - 1) {
        for v in (u + 1)..num_nodes {
            if distance(&pos[u], &pos[v], p) < radius_p {
                inner_graph.add_edge(
                    NodeIndex::new(u),
                    NodeIndex::new(v),
                    py.None(),
                );
            }
        }
    }

    let graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
        multigraph: true,
    };
    Ok(graph)
}

/// Return a list of cycles which form a basis for cycles of a given PyGraph
///
/// A basis for cycles of a graph is a minimal collection of
/// cycles such that any cycle in the graph can be written
/// as a sum of cycles in the basis.  Here summation of cycles
/// is defined as the exclusive or of the edges.
///
/// This is adapted from algorithm CACM 491 [1]_.
///
/// :param PyGraph graph: The graph to find the cycle basis in
/// :param int root: Optional index for starting node for basis
///
/// :returns: A list of cycle lists. Each list is a list of node ids which
///     forms a cycle (loop) in the input graph
/// :rtype: list
///
/// .. [1] Paton, K. An algorithm for finding a fundamental set of
///    cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.
#[pyfunction]
#[pyo3(text_signature = "(graph, /, root=None)")]
pub fn cycle_basis(
    graph: &graph::PyGraph,
    root: Option<usize>,
) -> Vec<Vec<usize>> {
    let mut root_node = root;
    let mut graph_nodes: HashSet<NodeIndex> =
        graph.graph.node_indices().collect();
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    while !graph_nodes.is_empty() {
        let temp_value: NodeIndex;
        // If root_node is not set get an arbitrary node from the set of graph
        // nodes we've not "examined"
        let root_index = match root_node {
            Some(root_value) => NodeIndex::new(root_value),
            None => {
                temp_value = *graph_nodes.iter().next().unwrap();
                graph_nodes.remove(&temp_value);
                temp_value
            }
        };
        // Stack (ie "pushdown list") of vertices already in the spanning tree
        let mut stack: Vec<NodeIndex> = vec![root_index];
        // Map of node index to predecessor node index
        let mut pred: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        pred.insert(root_index, root_index);
        // Set of examined nodes during this iteration
        let mut used: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();
        used.insert(root_index, HashSet::new());
        // Walk the spanning tree
        while !stack.is_empty() {
            // Use the last element added so that cycles are easier to find
            let z = stack.pop().unwrap();
            for neighbor in graph.graph.neighbors(z) {
                // A new node was encountered:
                if !used.contains_key(&neighbor) {
                    pred.insert(neighbor, z);
                    stack.push(neighbor);
                    let mut temp_set: HashSet<NodeIndex> = HashSet::new();
                    temp_set.insert(z);
                    used.insert(neighbor, temp_set);
                // A self loop:
                } else if z == neighbor {
                    let cycle: Vec<usize> = vec![z.index()];
                    cycles.push(cycle);
                // A cycle was found:
                } else if !used.get(&z).unwrap().contains(&neighbor) {
                    let pn = used.get(&neighbor).unwrap();
                    let mut cycle: Vec<NodeIndex> = vec![neighbor, z];
                    let mut p = pred.get(&z).unwrap();
                    while !pn.contains(p) {
                        cycle.push(*p);
                        p = pred.get(p).unwrap();
                    }
                    cycle.push(*p);
                    cycles.push(cycle.iter().map(|x| x.index()).collect());
                    let neighbor_set = used.get_mut(&neighbor).unwrap();
                    neighbor_set.insert(z);
                }
            }
        }
        let mut temp_hashset: HashSet<NodeIndex> = HashSet::new();
        for key in pred.keys() {
            temp_hashset.insert(*key);
        }
        graph_nodes = graph_nodes.difference(&temp_hashset).copied().collect();
        root_node = None;
    }
    cycles
}

/// Compute a maximum-weighted matching for a :class:`~retworkx.PyGraph`
///
/// A matching is a subset of edges in which no node occurs more than once.
/// The weight of a matching is the sum of the weights of its edges.
/// A maximal matching cannot add more edges and still be a matching.
/// The cardinality of a matching is the number of matched edges.
///
/// This function takes time :math:`O(n^3)` where ``n`` is the number of nodes
/// in the graph.
///
/// This method is based on the "blossom" method for finding augmenting
/// paths and the "primal-dual" method for finding a matching of maximum
/// weight, both methods invented by Jack Edmonds [1]_.
///
/// :param PyGraph graph: The undirected graph to compute the max weight
///     matching for. Expects to have no parallel edges (multigraphs are
///     untested currently).
/// :param bool max_cardinality: If True, compute the maximum-cardinality
///     matching with maximum weight among all maximum-cardinality matchings.
///     Defaults False.
/// :param callable weight_fn: An optional callable that will be passed a
///     single argument the edge object for each edge in the graph. It is
///     expected to return an ``int`` weight for that edge. For example,
///     if the weights are all integers you can use: ``lambda x: x``. If not
///     specified the value for ``default_weight`` will be used for all
///     edge weights.
/// :param int default_weight: The ``int`` value to use for all edge weights
///     in the graph if ``weight_fn`` is not specified. Defaults to ``1``.
/// :param bool verify_optimum: A boolean flag to run a check that the found
///     solution is optimum. If set to true an exception will be raised if
///     the found solution is not optimum. This is mostly useful for testing.
///
/// :returns: A set of tuples ofthe matching, Note that only a single
///     direction will be listed in the output, for example:
///     ``{(0, 1),}``.
/// :rtype: set
///
/// .. [1] "Efficient Algorithms for Finding Maximum Matching in Graphs",
///     Zvi Galil, ACM Computing Surveys, 1986.
///
#[pyfunction(
    max_cardinality = "false",
    default_weight = 1,
    verify_optimum = "false"
)]
#[pyo3(
    text_signature = "(graph, /, max_cardinality=False, weight_fn=None, default_weight=1, verify_optimum=False)"
)]
pub fn max_weight_matching(
    py: Python,
    graph: &graph::PyGraph,
    max_cardinality: bool,
    weight_fn: Option<PyObject>,
    default_weight: i128,
    verify_optimum: bool,
) -> PyResult<HashSet<(usize, usize)>> {
    max_weight_matching::max_weight_matching(
        py,
        graph,
        max_cardinality,
        weight_fn,
        default_weight,
        verify_optimum,
    )
}

/// Compute the strongly connected components for a directed graph
///
/// This function is implemented using Kosaraju's algorithm
///
/// :param PyDiGraph graph: The input graph to find the strongly connected
///     components for.
///
/// :return: A list of list of node ids for strongly connected components
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn strongly_connected_components(
    graph: &digraph::PyDiGraph,
) -> Vec<Vec<usize>> {
    algo::kosaraju_scc(graph)
        .iter()
        .map(|x| x.iter().map(|id| id.index()).collect())
        .collect()
}

/// Return the first cycle encountered during DFS of a given PyDiGraph,
/// empty list is returned if no cycle is found
///
/// :param PyDiGraph graph: The graph to find the cycle in
/// :param int source: Optional index to find a cycle for. If not specified an
///     arbitrary node will be selected from the graph.
///
/// :returns: A list describing the cycle. The index of node ids which
///     forms a cycle (loop) in the input graph
/// :rtype: EdgeList
#[pyfunction]
#[pyo3(text_signature = "(graph, /, source=None)")]
pub fn digraph_find_cycle(
    graph: &digraph::PyDiGraph,
    source: Option<usize>,
) -> EdgeList {
    let mut graph_nodes: HashSet<NodeIndex> =
        graph.graph.node_indices().collect();
    let mut cycle: Vec<(usize, usize)> =
        Vec::with_capacity(graph.graph.edge_count());
    let temp_value: NodeIndex;
    // If source is not set get an arbitrary node from the set of graph
    // nodes we've not "examined"
    let source_index = match source {
        Some(source_value) => NodeIndex::new(source_value),
        None => {
            temp_value = *graph_nodes.iter().next().unwrap();
            graph_nodes.remove(&temp_value);
            temp_value
        }
    };

    // Stack (ie "pushdown list") of vertices already in the spanning tree
    let mut stack: Vec<NodeIndex> = vec![source_index];
    // map to store parent of a node
    let mut pred: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    // a node is in the visiting set if at least one of its child is unexamined
    let mut visiting = HashSet::new();
    // a node is in visited set if all of its children have been examined
    let mut visited = HashSet::new();
    while !stack.is_empty() {
        let mut z = *stack.last().unwrap();
        visiting.insert(z);

        let children = graph
            .graph
            .neighbors_directed(z, petgraph::Direction::Outgoing);

        for child in children {
            //cycle is found
            if visiting.contains(&child) {
                cycle.push((z.index(), child.index()));
                //backtrack
                loop {
                    if z == child {
                        cycle.reverse();
                        break;
                    }
                    cycle.push((pred[&z].index(), z.index()));
                    z = pred[&z];
                }
                return EdgeList { edges: cycle };
            }
            //if an unexplored node is encountered
            if !visited.contains(&child) {
                stack.push(child);
                pred.insert(child, z);
            }
        }

        let top = *stack.last().unwrap();
        //if no further children and explored, move to visited
        if top.index() == z.index() {
            stack.pop();
            visiting.remove(&z);
            visited.insert(z);
        }
    }
    EdgeList { edges: cycle }
}

fn _inner_is_matching(
    graph: &graph::PyGraph,
    matching: &HashSet<(usize, usize)>,
) -> bool {
    let has_edge = |e: &(usize, usize)| -> bool {
        graph
            .graph
            .contains_edge(NodeIndex::new(e.0), NodeIndex::new(e.1))
    };

    if !matching.iter().all(|e| has_edge(e)) {
        return false;
    }
    let mut found: HashSet<usize> = HashSet::with_capacity(2 * matching.len());
    for (v1, v2) in matching {
        if found.contains(v1) || found.contains(v2) {
            return false;
        }
        found.insert(*v1);
        found.insert(*v2);
    }
    true
}

/// Check if matching is valid for graph
///
/// A *matching* in a graph is a set of edges in which no two distinct
/// edges share a common endpoint.
///
/// :param PyDiGraph graph: The graph to check if the matching is valid for
/// :param set matching: A set of node index tuples for each edge in the
///     matching.
///
/// :returns: Whether the provided matching is a valid matching for the graph
/// :rtype: bool
#[pyfunction]
#[pyo3(text_signature = "(graph, matching, /)")]
pub fn is_matching(
    graph: &graph::PyGraph,
    matching: HashSet<(usize, usize)>,
) -> bool {
    _inner_is_matching(graph, &matching)
}

/// Check if a matching is a maximal (**not** maximum) matching for a graph
///
/// A *maximal matching* in a graph is a matching in which adding any
/// edge would cause the set to no longer be a valid matching.
///
/// .. note::
///
///   This is not checking for a *maximum* (globally optimal) matching, but
///   a *maximal* (locally optimal) matching.
///
/// :param PyDiGraph graph: The graph to check if the matching is maximal for.
/// :param set matching: A set of node index tuples for each edge in the
///     matching.
///
/// :returns: Whether the provided matching is a valid matching and whether it
///     is maximal or not.
/// :rtype: bool
#[pyfunction]
#[pyo3(text_signature = "(graph, matching, /)")]
pub fn is_maximal_matching(
    graph: &graph::PyGraph,
    matching: HashSet<(usize, usize)>,
) -> bool {
    if !_inner_is_matching(graph, &matching) {
        return false;
    }
    let edge_list: HashSet<[usize; 2]> = graph
        .edge_references()
        .map(|edge| {
            let mut tmp_array = [edge.source().index(), edge.target().index()];
            tmp_array.sort_unstable();
            tmp_array
        })
        .collect();
    let matched_edges: HashSet<[usize; 2]> = matching
        .iter()
        .map(|edge| {
            let mut tmp_array = [edge.0, edge.1];
            tmp_array.sort_unstable();
            tmp_array
        })
        .collect();
    let mut unmatched_edges = edge_list.difference(&matched_edges);
    unmatched_edges.all(|e| {
        let mut tmp_set = matching.clone();
        tmp_set.insert((e[0], e[1]));
        !_inner_is_matching(graph, &tmp_set)
    })
}

fn _graph_triangles(graph: &graph::PyGraph, node: usize) -> (usize, usize) {
    let mut triangles: usize = 0;

    let index = NodeIndex::new(node);
    let mut neighbors: HashSet<NodeIndex> =
        graph.graph.neighbors(index).collect();
    neighbors.remove(&index);

    for nodev in &neighbors {
        triangles += graph
            .graph
            .neighbors(*nodev)
            .filter(|&x| (x != *nodev) && neighbors.contains(&x))
            .count();
    }

    let d: usize = neighbors.len();
    let triples: usize = match d {
        0 => 0,
        _ => (d * (d - 1)) / 2,
    };

    (triangles / 2, triples)
}

/// Compute the transitivity of an undirected graph.
///
/// The transitivity of a graph is defined as:
///
/// .. math::
///     `c=3 \times \frac{\text{number of triangles}}{\text{number of connected triples}}`
///
/// A “connected triple” means a single vertex with
/// edges running to an unordered pair of others.
///
/// This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyGraph graph: Graph to be used.
///
/// :returns: Transitivity.
/// :rtype: float
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
fn graph_transitivity(graph: &graph::PyGraph) -> f64 {
    let node_indices: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let (triangles, triples) = node_indices
        .par_iter()
        .map(|node| _graph_triangles(graph, node.index()))
        .reduce(
            || (0, 0),
            |(sumx, sumy), (resx, resy)| (sumx + resx, sumy + resy),
        );

    match triangles {
        0 => 0.0,
        _ => triangles as f64 / triples as f64,
    }
}

fn _digraph_triangles(
    graph: &digraph::PyDiGraph,
    node: usize,
) -> (usize, usize) {
    let mut triangles: usize = 0;

    let index = NodeIndex::new(node);
    let mut out_neighbors: HashSet<NodeIndex> = graph
        .graph
        .neighbors_directed(index, petgraph::Direction::Outgoing)
        .collect();
    out_neighbors.remove(&index);

    let mut in_neighbors: HashSet<NodeIndex> = graph
        .graph
        .neighbors_directed(index, petgraph::Direction::Incoming)
        .collect();
    in_neighbors.remove(&index);

    let neighbors = out_neighbors.iter().chain(in_neighbors.iter());

    for nodev in neighbors {
        triangles += graph
            .graph
            .neighbors_directed(*nodev, petgraph::Direction::Outgoing)
            .chain(
                graph
                    .graph
                    .neighbors_directed(*nodev, petgraph::Direction::Incoming),
            )
            .map(|x| {
                let mut res: usize = 0;

                if (x != *nodev) && out_neighbors.contains(&x) {
                    res += 1;
                }
                if (x != *nodev) && in_neighbors.contains(&x) {
                    res += 1;
                }
                res
            })
            .sum::<usize>();
    }

    let din: usize = in_neighbors.len();
    let dout: usize = out_neighbors.len();

    let dtot = dout + din;
    let dbil: usize = out_neighbors.intersection(&in_neighbors).count();
    let triples: usize = match dtot {
        0 => 0,
        _ => dtot * (dtot - 1) - 2 * dbil,
    };

    (triangles / 2, triples)
}

/// Compute the transitivity of a directed graph.
///
/// The transitivity of a directed graph is defined in [Fag]_, Eq.8:
///
/// .. math::
///     `c=3 \times \frac{\text{number of triangles}}{\text{number of all possible triangles}}`
///
/// A triangle is a connected triple of nodes.
/// Different edge orientations counts as different triangles.
///
/// This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyDiGraph graph: Directed graph to be used.
///
/// :returns: Transitivity.
/// :rtype: float
///
/// .. [Fag] Clustering in complex directed networks by G. Fagiolo,
///    Physical Review E, 76(2), 026107 (2007)
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
fn digraph_transitivity(graph: &digraph::PyDiGraph) -> f64 {
    let node_indices: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let (triangles, triples) = node_indices
        .par_iter()
        .map(|node| _digraph_triangles(graph, node.index()))
        .reduce(
            || (0, 0),
            |(sumx, sumy), (resx, resy)| (sumx + resx, sumy + resy),
        );

    match triangles {
        0 => 0.0,
        _ => triangles as f64 / triples as f64,
    }
}

pub fn _core_number<Ty>(
    py: Python,
    graph: &StableGraph<PyObject, PyObject, Ty>,
) -> PyResult<PyObject>
where
    Ty: EdgeType,
{
    let node_num = graph.node_count();
    if node_num == 0 {
        return Ok(PyDict::new(py).into());
    }

    let mut cores: HashMap<NodeIndex, usize> = HashMap::with_capacity(node_num);
    let mut node_vec: Vec<NodeIndex> = graph.node_indices().collect();
    let mut degree_map: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(node_num);
    let mut nbrs: HashMap<NodeIndex, HashSet<NodeIndex>> =
        HashMap::with_capacity(node_num);
    let mut node_pos: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(node_num);

    for k in node_vec.iter() {
        let k_nbrs: HashSet<NodeIndex> =
            graph.neighbors_undirected(*k).collect();
        let k_deg = k_nbrs.len();

        nbrs.insert(*k, k_nbrs);
        cores.insert(*k, k_deg);
        degree_map.insert(*k, k_deg);
    }
    node_vec.par_sort_by_key(|k| degree_map.get(k));

    let mut bin_boundaries: Vec<usize> =
        Vec::with_capacity(degree_map[&node_vec[node_num - 1]] + 1);
    bin_boundaries.push(0);
    let mut curr_degree = 0;
    for (i, v) in node_vec.iter().enumerate() {
        node_pos.insert(*v, i);
        let v_degree = degree_map[v];
        if v_degree > curr_degree {
            for _ in 0..v_degree - curr_degree {
                bin_boundaries.push(i);
            }
            curr_degree = v_degree;
        }
    }

    for v_ind in 0..node_vec.len() {
        let v = node_vec[v_ind];
        let v_nbrs = nbrs[&v].clone();
        for u in v_nbrs {
            if cores[&u] > cores[&v] {
                nbrs.get_mut(&u).unwrap().remove(&v);
                let pos = node_pos[&u];
                let bin_start = bin_boundaries[cores[&u]];
                *node_pos.get_mut(&u).unwrap() = bin_start;
                *node_pos.get_mut(&node_vec[bin_start]).unwrap() = pos;
                node_vec.swap(bin_start, pos);
                bin_boundaries[cores[&u]] += 1;
                *cores.get_mut(&u).unwrap() -= 1;
            }
        }
    }

    let out_dict = PyDict::new(py);
    for (v_index, core) in cores {
        out_dict.set_item(v_index.index(), core)?;
    }
    Ok(out_dict.into())
}

/// Return the core number for each node in the graph.
///
/// A k-core is a maximal subgraph that contains nodes of degree k or more.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyGraph: The graph to get core numbers
///
/// :returns: A dictionary keyed by node index to the core number
/// :rtype: dict
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_core_number(
    py: Python,
    graph: &graph::PyGraph,
) -> PyResult<PyObject> {
    _core_number(py, &graph.graph)
}

/// Return the core number for each node in the directed graph.
///
/// A k-core is a maximal subgraph that contains nodes of degree k or more.
/// For directed graphs, the degree is calculated as in_degree + out_degree.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyDiGraph: The directed graph to get core numbers
///
/// :returns: A dictionary keyed by node index to the core number
/// :rtype: dict
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn digraph_core_number(
    py: Python,
    graph: &digraph::PyDiGraph,
) -> PyResult<PyObject> {
    _core_number(py, &graph.graph)
}

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
    for edge in graph.edge_references() {
        let weight =
            weight_callable(py, &weight_fn, edge.weight(), default_weight)?;
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
///     The new graph will keep the same node indexes, but edge indexes might differ.
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
        spanning_tree.add_edge(edge.0, edge.1, edge.2.clone_ref(py))?;
    }

    Ok(spanning_tree)
}

/// Compute the complement of a graph.
///
/// :param PyGraph graph: The graph to be used.
///
/// :returns: The complement of the graph.
/// :rtype: PyGraph
///
/// .. note::
///
///     Parallel edges and self-loops are never created,
///     even if the :attr:`~retworkx.PyGraph.multigraph`
///     attribute is set to ``True``
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
fn graph_complement(
    py: Python,
    graph: &graph::PyGraph,
) -> PyResult<graph::PyGraph> {
    let mut complement_graph = graph.clone(); // keep same node indexes
    complement_graph.graph.clear_edges();

    for node_a in graph.graph.node_indices() {
        let old_neighbors: HashSet<NodeIndex> =
            graph.graph.neighbors(node_a).collect();
        for node_b in graph.graph.node_indices() {
            if node_a != node_b
                && !old_neighbors.contains(&node_b)
                && (!complement_graph.multigraph
                    || !complement_graph
                        .has_edge(node_a.index(), node_b.index()))
            {
                // avoid creating parallel edges in multigraph
                complement_graph.add_edge(
                    node_a.index(),
                    node_b.index(),
                    py.None(),
                )?;
            }
        }
    }
    Ok(complement_graph)
}

/// Compute the complement of a graph.
///
/// :param PyDiGraph graph: The graph to be used.
///
/// :returns: The complement of the graph.
/// :rtype: :class:`~retworkx.PyDiGraph`
///
/// .. note::
///
///     Parallel edges and self-loops are never created,
///     even if the :attr:`~retworkx.PyDiGraph.multigraph`
///     attribute is set to ``True``
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
fn digraph_complement(
    py: Python,
    graph: &digraph::PyDiGraph,
) -> PyResult<digraph::PyDiGraph> {
    let mut complement_graph = graph.clone(); // keep same node indexes
    complement_graph.graph.clear_edges();

    for node_a in graph.graph.node_indices() {
        let old_neighbors: HashSet<NodeIndex> = graph
            .graph
            .neighbors_directed(node_a, petgraph::Direction::Outgoing)
            .collect();
        for node_b in graph.graph.node_indices() {
            if node_a != node_b && !old_neighbors.contains(&node_b) {
                complement_graph.add_edge(
                    node_a.index(),
                    node_b.index(),
                    py.None(),
                )?;
            }
        }
    }

    Ok(complement_graph)
}

#[allow(clippy::too_many_arguments)]
fn _spring_layout<Ty>(
    py: Python,
    graph: &StableGraph<PyObject, PyObject, Ty>,
    pos: Option<HashMap<usize, layout::Point>>,
    fixed: Option<HashSet<usize>>,
    k: Option<f64>,
    repulsive_exponent: Option<i32>,
    adaptive_cooling: Option<bool>,
    num_iter: Option<usize>,
    tol: Option<f64>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    scale: Option<f64>,
    center: Option<layout::Point>,
    seed: Option<u64>,
) -> PyResult<Pos2DMapping>
where
    Ty: EdgeType,
{
    if fixed.is_some() && pos.is_none() {
        return Err(PyValueError::new_err("`fixed` specified but `pos` not."));
    }

    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };

    let dist = Uniform::new(0.0, 1.0);

    let pos = pos.unwrap_or_default();
    let mut vpos: Vec<layout::Point> = (0..graph.node_bound())
        .map(|_| [dist.sample(&mut rng), dist.sample(&mut rng)])
        .collect();
    for (n, p) in pos.into_iter() {
        vpos[n] = p;
    }

    let fixed = fixed.unwrap_or_default();
    let k = k.unwrap_or(1.0 / (graph.node_count() as f64).sqrt());
    let f_a = layout::AttractiveForce::new(k);
    let f_r = layout::RepulsiveForce::new(k, repulsive_exponent.unwrap_or(2));

    let num_iter = num_iter.unwrap_or(50);
    let tol = tol.unwrap_or(1e-6);
    let step = 0.1;

    let mut weights: HashMap<(usize, usize), f64> =
        HashMap::with_capacity(2 * graph.edge_count());
    for e in graph.edge_references() {
        let w = weight_callable(py, &weight_fn, e.weight(), default_weight)?;
        let source = e.source().index();
        let target = e.target().index();

        weights.insert((source, target), w);
        weights.insert((target, source), w);
    }

    let pos = match adaptive_cooling {
        Some(false) => {
            let cs = layout::LinearCoolingScheme::new(step, num_iter);
            layout::evolve(
                graph, vpos, fixed, f_a, f_r, cs, num_iter, tol, weights,
                scale, center,
            )
        }
        _ => {
            let cs = layout::AdaptiveCoolingScheme::new(step);
            layout::evolve(
                graph, vpos, fixed, f_a, f_r, cs, num_iter, tol, weights,
                scale, center,
            )
        }
    };

    Ok(Pos2DMapping {
        pos_map: graph
            .node_indices()
            .map(|n| {
                let n = n.index();
                (n, pos[n])
            })
            .collect(),
    })
}

/// Position nodes using Fruchterman-Reingold force-directed algorithm.
///
/// The algorithm simulates a force-directed representation of the network
/// treating edges as springs holding nodes close, while treating nodes
/// as repelling objects, sometimes called an anti-gravity force.
/// Simulation continues until the positions are close to an equilibrium.
///
/// :param PyGraph graph: Graph to be used.
/// :param dict pos:
///     Initial node positions as a dictionary with node ids as keys and values
///     as a coordinate list. If ``None``, then use random initial positions. (``default=None``)
/// :param set fixed: Nodes to keep fixed at initial position.
///     Error raised if fixed specified and ``pos`` is not. (``default=None``)
/// :param float  k:
///     Optimal distance between nodes. If ``None`` the distance is set to
///     :math:`\frac{1}{\sqrt{n}}` where :math:`n` is the number of nodes.  Increase this value
///     to move nodes farther apart. (``default=None``)
/// :param int repulsive_exponent:
///     Repulsive force exponent. (``default=2``)
/// :param bool adaptive_cooling:
///     Use an adaptive cooling scheme. If set to ``False``,
///     a linear cooling scheme is used. (``default=True``)
/// :param int num_iter:
///     Maximum number of iterations. (``default=50``)
/// :param float tol:
///     Threshold for relative error in node position changes.
///     The iteration stops if the error is below this threshold.
///     (``default = 1e-6``)
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float
///     which will be used to represent the weight of the edge.
/// :param float (default=1) default_weight: If ``weight_fn`` isn't specified
///     this optional float value will be used for the weight/cost of each edge
/// :param float|None scale: Scale factor for positions.
///     Not used unless fixed is None. If scale is ``None``, no re-scaling is
///     performed. (``default=1.0``)
/// :param list center: Coordinate pair around which to center
///     the layout. Not used unless fixed is ``None``. (``default=None``)
/// :param int seed: An optional seed to use for the random number generator
///
/// :returns: A dictionary of positions keyed by node id.
/// :rtype: dict
#[pyfunction]
#[pyo3(
    text_signature = "(graph, pos=None, fixed=None, k=None, repulsive_exponent=2, adaptive_cooling=True,
                     num_iter=50, tol=1e-6, weight_fn=None, default_weight=1, scale=1,
                     center=None, seed=None, /)"
)]
#[allow(clippy::too_many_arguments)]
pub fn graph_spring_layout(
    py: Python,
    graph: &graph::PyGraph,
    pos: Option<HashMap<usize, layout::Point>>,
    fixed: Option<HashSet<usize>>,
    k: Option<f64>,
    repulsive_exponent: Option<i32>,
    adaptive_cooling: Option<bool>,
    num_iter: Option<usize>,
    tol: Option<f64>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    scale: Option<f64>,
    center: Option<layout::Point>,
    seed: Option<u64>,
) -> PyResult<Pos2DMapping> {
    _spring_layout(
        py,
        &graph.graph,
        pos,
        fixed,
        k,
        repulsive_exponent,
        adaptive_cooling,
        num_iter,
        tol,
        weight_fn,
        default_weight,
        scale,
        center,
        seed,
    )
}

/// Position nodes using Fruchterman-Reingold force-directed algorithm.
///
/// The algorithm simulates a force-directed representation of the network
/// treating edges as springs holding nodes close, while treating nodes
/// as repelling objects, sometimes called an anti-gravity force.
/// Simulation continues until the positions are close to an equilibrium.
///
/// :param PyGraph graph: Graph to be used.
/// :param dict pos:
///     Initial node positions as a dictionary with node ids as keys and values
///     as a coordinate list. If ``None``, then use random initial positions. (``default=None``)
/// :param set fixed: Nodes to keep fixed at initial position.
///     Error raised if fixed specified and ``pos`` is not. (``default=None``)
/// :param float  k:
///     Optimal distance between nodes. If ``None`` the distance is set to
///     :math:`\frac{1}{\sqrt{n}}` where :math:`n` is the number of nodes.  Increase this value
///     to move nodes farther apart. (``default=None``)
/// :param int repulsive_exponent:
///     Repulsive force exponent. (``default=2``)
/// :param bool adaptive_cooling:
///     Use an adaptive cooling scheme. If set to ``False``,
///     a linear cooling scheme is used. (``default=True``)
/// :param int num_iter:
///     Maximum number of iterations. (``default=50``)
/// :param float tol:
///     Threshold for relative error in node position changes.
///     The iteration stops if the error is below this threshold.
///     (``default = 1e-6``)
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float
///     which will be used to represent the weight of the edge.
/// :param float (default=1) default_weight: If ``weight_fn`` isn't specified
///     this optional float value will be used for the weight/cost of each edge
/// :param float|None scale: Scale factor for positions.
///     Not used unless fixed is None. If scale is ``None``, no re-scaling is
///     performed. (``default=1.0``)
/// :param list center: Coordinate pair around which to center
///     the layout. Not used unless fixed is ``None``. (``default=None``)
/// :param int seed: An optional seed to use for the random number generator
///
/// :returns: A dictionary of positions keyed by node id.
/// :rtype: dict
#[pyfunction]
#[pyo3(
    text_signature = "(graph, pos=None, fixed=None, k=None, repulsive_exponent=2, adaptive_cooling=True,
                     num_iter=50, tol=1e-6, weight_fn=None, default_weight=1, scale=1,
                     center=None, seed=None, /)"
)]
#[allow(clippy::too_many_arguments)]
pub fn digraph_spring_layout(
    py: Python,
    graph: &digraph::PyDiGraph,
    pos: Option<HashMap<usize, layout::Point>>,
    fixed: Option<HashSet<usize>>,
    k: Option<f64>,
    repulsive_exponent: Option<i32>,
    adaptive_cooling: Option<bool>,
    num_iter: Option<usize>,
    tol: Option<f64>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    scale: Option<f64>,
    center: Option<layout::Point>,
    seed: Option<u64>,
) -> PyResult<Pos2DMapping> {
    _spring_layout(
        py,
        &graph.graph,
        pos,
        fixed,
        k,
        repulsive_exponent,
        adaptive_cooling,
        num_iter,
        tol,
        weight_fn,
        default_weight,
        scale,
        center,
        seed,
    )
}

fn _random_layout<Ty: EdgeType>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
    center: Option<[f64; 2]>,
    seed: Option<u64>,
) -> Pos2DMapping {
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };

    Pos2DMapping {
        pos_map: graph
            .node_indices()
            .map(|n| {
                let random_tuple: [f64; 2] = rng.gen();
                match center {
                    Some(center) => (
                        n.index(),
                        [
                            random_tuple[0] + center[0],
                            random_tuple[1] + center[1],
                        ],
                    ),
                    None => (n.index(), random_tuple),
                }
            })
            .collect(),
    }
}

/// Generate a random layout
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param int seed: An optional seed to set for the random number generator.
///
/// :returns: The random layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, / center=None, seed=None)")]
pub fn graph_random_layout(
    graph: &graph::PyGraph,
    center: Option<[f64; 2]>,
    seed: Option<u64>,
) -> Pos2DMapping {
    _random_layout(&graph.graph, center, seed)
}

/// Generate a random layout
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param int seed: An optional seed to set for the random number generator.
///
/// :returns: The random layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, / center=None, seed=None)")]
pub fn digraph_random_layout(
    graph: &digraph::PyDiGraph,
    center: Option<[f64; 2]>,
    seed: Option<u64>,
) -> Pos2DMapping {
    _random_layout(&graph.graph, center, seed)
}

/// Generate a bipartite layout of the graph
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param set first_nodes: The set of node indexes on the left (or top if
///     horitontal is true)
/// :param bool horizontal: An optional bool specifying the orientation of the
///     layout
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param float aspect_ratio: An optional number for the ratio of the width to
///     the height of the layout.
///
/// :returns: The bipartite layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, first_nodes, /, horitontal=False, scale=1,
                     center=None, aspect_ratio=1.33333333333333)")]
pub fn graph_bipartite_layout(
    graph: &graph::PyGraph,
    first_nodes: HashSet<usize>,
    horizontal: Option<bool>,
    scale: Option<f64>,
    center: Option<layout::Point>,
    aspect_ratio: Option<f64>,
) -> Pos2DMapping {
    layout::bipartite_layout(
        &graph.graph,
        first_nodes,
        horizontal,
        scale,
        center,
        aspect_ratio,
    )
}

/// Generate a bipartite layout of the graph
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param set first_nodes: The set of node indexes on the left (or top if
///     horizontal is true)
/// :param bool horizontal: An optional bool specifying the orientation of the
///     layout
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param float aspect_ratio: An optional number for the ratio of the width to
///     the height of the layout.
///
/// :returns: The bipartite layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, first_nodes, /, horitontal=False, scale=1,
                     center=None, aspect_ratio=1.33333333333333)")]
pub fn digraph_bipartite_layout(
    graph: &digraph::PyDiGraph,
    first_nodes: HashSet<usize>,
    horizontal: Option<bool>,
    scale: Option<f64>,
    center: Option<layout::Point>,
    aspect_ratio: Option<f64>,
) -> Pos2DMapping {
    layout::bipartite_layout(
        &graph.graph,
        first_nodes,
        horizontal,
        scale,
        center,
        aspect_ratio,
    )
}

/// Generate a circular layout of the graph
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
///
/// :returns: The circular layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, scale=1, center=None)")]
pub fn graph_circular_layout(
    graph: &graph::PyGraph,
    scale: Option<f64>,
    center: Option<layout::Point>,
) -> Pos2DMapping {
    layout::circular_layout(&graph.graph, scale, center)
}

/// Generate a circular layout of the graph
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
///
/// :returns: The circular layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, scale=1, center=None)")]
pub fn digraph_circular_layout(
    graph: &digraph::PyDiGraph,
    scale: Option<f64>,
    center: Option<layout::Point>,
) -> Pos2DMapping {
    layout::circular_layout(&graph.graph, scale, center)
}

/// Generate a shell layout of the graph
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param list nlist: The list of lists of indexes which represents each shell
/// :param float rotate: Angle (in radians) by which to rotate the starting
///     position of each shell relative to the starting position of the
///     previous shell
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
///
/// :returns: The shell layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(
    text_signature = "(graph, /, nlist=None, rotate=None, scale=1, center=None)"
)]
pub fn graph_shell_layout(
    graph: &graph::PyGraph,
    nlist: Option<Vec<Vec<usize>>>,
    rotate: Option<f64>,
    scale: Option<f64>,
    center: Option<layout::Point>,
) -> Pos2DMapping {
    layout::shell_layout(&graph.graph, nlist, rotate, scale, center)
}

/// Generate a shell layout of the graph
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param list nlist: The list of lists of indexes which represents each shell
/// :param float rotate: Angle by which to rotate the starting position of each shell
///     relative to the starting position of the previous shell (in radians)
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
///
/// :returns: The shell layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(
    text_signature = "(graph, /, nlist=None, rotate=None, scale=1, center=None)"
)]
pub fn digraph_shell_layout(
    graph: &digraph::PyDiGraph,
    nlist: Option<Vec<Vec<usize>>>,
    rotate: Option<f64>,
    scale: Option<f64>,
    center: Option<layout::Point>,
) -> Pos2DMapping {
    layout::shell_layout(&graph.graph, nlist, rotate, scale, center)
}

/// Generate a spiral layout of the graph
///
/// :param PyGraph graph: The graph to generate the layout for
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param float resolution: The compactness of the spiral layout returned.
///     Lower values result in more compressed spiral layouts.
/// :param bool equidistant: If true, nodes will be plotted equidistant from
///     each other.
///
/// :returns: The spiral layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, scale=1, center=None, resolution=0.35,
                     equidistant=False)")]
pub fn graph_spiral_layout(
    graph: &graph::PyGraph,
    scale: Option<f64>,
    center: Option<layout::Point>,
    resolution: Option<f64>,
    equidistant: Option<bool>,
) -> Pos2DMapping {
    layout::spiral_layout(&graph.graph, scale, center, resolution, equidistant)
}

/// Generate a spiral layout of the graph
///
/// :param PyDiGraph graph: The graph to generate the layout for
/// :param float scale: An optional scaling factor to scale positions
/// :param tuple center: An optional center position. This is a 2 tuple of two
///     ``float`` values for the center position
/// :param float resolution: The compactness of the spiral layout returned.
///     Lower values result in more compressed spiral layouts.
/// :param bool equidistant: If true, nodes will be plotted equidistant from
///     each other.
///
/// :returns: The spiral layout of the graph.
/// :rtype: Pos2DMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, /, scale=1, center=None, resolution=0.35,
                     equidistant=False)")]
pub fn digraph_spiral_layout(
    graph: &digraph::PyDiGraph,
    scale: Option<f64>,
    center: Option<layout::Point>,
    resolution: Option<f64>,
    equidistant: Option<bool>,
) -> Pos2DMapping {
    layout::spiral_layout(&graph.graph, scale, center, resolution, equidistant)
}

fn _num_shortest_paths_unweighted<Ty: EdgeType>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
    source: usize,
) -> PyResult<HashMap<usize, BigUint>> {
    let mut out_map: Vec<BigUint> =
        vec![0.to_biguint().unwrap(); graph.node_bound()];
    let node_index = NodeIndex::new(source);
    if graph.node_weight(node_index).is_none() {
        return Err(PyIndexError::new_err(format!(
            "No node found for index {}",
            source
        )));
    }
    let mut bfs = Bfs::new(&graph, node_index);
    let mut distance: Vec<Option<usize>> = vec![None; graph.node_bound()];
    distance[node_index.index()] = Some(0);
    out_map[source] = 1.to_biguint().unwrap();
    while let Some(current) = bfs.next(graph) {
        let dist_plus_one = distance[current.index()].unwrap_or_default() + 1;
        let count_current = out_map[current.index()].clone();
        for neighbor_index in
            graph.neighbors_directed(current, petgraph::Direction::Outgoing)
        {
            let neighbor: usize = neighbor_index.index();
            if distance[neighbor].is_none() {
                distance[neighbor] = Some(dist_plus_one);
                out_map[neighbor] = count_current.clone();
            } else if distance[neighbor] == Some(dist_plus_one) {
                out_map[neighbor] += &count_current;
            }
        }
    }

    // Do not count paths to source in output
    distance[source] = None;
    out_map[source] = 0.to_biguint().unwrap();

    // Return only nodes that are reachable in the graph
    Ok(out_map
        .into_iter()
        .zip(distance.iter())
        .enumerate()
        .filter_map(|(index, (count, dist))| {
            if dist.is_some() {
                Some((index, count))
            } else {
                None
            }
        })
        .collect())
}

/// Get the number of unweighted shortest paths from a source node
///
/// :param PyDiGraph graph: The graph to find the number of shortest paths on
/// :param int source: The source node to find the shortest paths from
///
/// :returns: A mapping of target node indices to the number of shortest paths
///     from ``source`` to that node. If there is no path from ``source`` to
///     a node in the graph that node will not be preset in the output mapping.
/// :rtype: NodesCountMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, source, /)")]
pub fn digraph_num_shortest_paths_unweighted(
    graph: &digraph::PyDiGraph,
    source: usize,
) -> PyResult<NodesCountMapping> {
    Ok(NodesCountMapping {
        map: _num_shortest_paths_unweighted(&graph.graph, source)?,
    })
}

/// Get the number of unweighted shortest paths from a source node
///
/// :param PyGraph graph: The graph to find the number of shortest paths on
/// :param int source: The source node to find the shortest paths from
///
/// :returns: A mapping of target node indices to the number of shortest paths
///     from ``source`` to that node. If there is no path from ``source`` to
///     a node in the graph that node will not be preset in the output mapping.
/// :rtype: NumPathsMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, source, /)")]
pub fn graph_num_shortest_paths_unweighted(
    graph: &graph::PyGraph,
    source: usize,
) -> PyResult<NodesCountMapping> {
    Ok(NodesCountMapping {
        map: _num_shortest_paths_unweighted(&graph.graph, source)?,
    })
}

// The provided node is invalid.
create_exception!(retworkx, InvalidNode, PyException);
// Performing this operation would result in trying to add a cycle to a DAG.
create_exception!(retworkx, DAGWouldCycle, PyException);
// There is no edge present between the provided nodes.
create_exception!(retworkx, NoEdgeBetweenNodes, PyException);
// The specified Directed Graph has a cycle and can't be treated as a DAG.
create_exception!(retworkx, DAGHasCycle, PyException);
// No neighbors found matching the provided predicate.
create_exception!(retworkx, NoSuitableNeighbors, PyException);
// Invalid operation on a null graph
create_exception!(retworkx, NullGraph, PyException);
// No path was found between the specified nodes.
create_exception!(retworkx, NoPathFound, PyException);

#[pymodule]
fn retworkx(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("InvalidNode", py.get_type::<InvalidNode>())?;
    m.add("DAGWouldCycle", py.get_type::<DAGWouldCycle>())?;
    m.add("NoEdgeBetweenNodes", py.get_type::<NoEdgeBetweenNodes>())?;
    m.add("DAGHasCycle", py.get_type::<DAGHasCycle>())?;
    m.add("NoSuitableNeighbors", py.get_type::<NoSuitableNeighbors>())?;
    m.add("NoPathFound", py.get_type::<NoPathFound>())?;
    m.add("NullGraph", py.get_type::<NullGraph>())?;
    m.add_wrapped(wrap_pyfunction!(bfs_successors))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path_length))?;
    m.add_wrapped(wrap_pyfunction!(dag_weighted_longest_path))?;
    m.add_wrapped(wrap_pyfunction!(dag_weighted_longest_path_length))?;
    m.add_wrapped(wrap_pyfunction!(number_weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(is_weakly_connected))?;
    m.add_wrapped(wrap_pyfunction!(is_directed_acyclic_graph))?;
    m.add_wrapped(wrap_pyfunction!(digraph_is_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(graph_is_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(digraph_is_subgraph_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(graph_is_subgraph_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(digraph_union))?;
    m.add_wrapped(wrap_pyfunction!(topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(descendants))?;
    m.add_wrapped(wrap_pyfunction!(ancestors))?;
    m.add_wrapped(wrap_pyfunction!(lexicographical_topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(graph_floyd_warshall))?;
    m.add_wrapped(wrap_pyfunction!(digraph_floyd_warshall))?;
    m.add_wrapped(wrap_pyfunction!(graph_floyd_warshall_numpy))?;
    m.add_wrapped(wrap_pyfunction!(digraph_floyd_warshall_numpy))?;
    m.add_wrapped(wrap_pyfunction!(collect_runs))?;
    m.add_wrapped(wrap_pyfunction!(layers))?;
    m.add_wrapped(wrap_pyfunction!(graph_distance_matrix))?;
    m.add_wrapped(wrap_pyfunction!(digraph_distance_matrix))?;
    m.add_wrapped(wrap_pyfunction!(digraph_adjacency_matrix))?;
    m.add_wrapped(wrap_pyfunction!(graph_adjacency_matrix))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_dijkstra_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dijkstra_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_pairs_dijkstra_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_pairs_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_pairs_dijkstra_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_pairs_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_astar_shortest_path))?;
    m.add_wrapped(wrap_pyfunction!(digraph_astar_shortest_path))?;
    m.add_wrapped(wrap_pyfunction!(graph_greedy_color))?;
    m.add_wrapped(wrap_pyfunction!(directed_gnp_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(undirected_gnp_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_gnm_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(undirected_gnm_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(random_geometric_graph))?;
    m.add_wrapped(wrap_pyfunction!(cycle_basis))?;
    m.add_wrapped(wrap_pyfunction!(strongly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dfs_edges))?;
    m.add_wrapped(wrap_pyfunction!(graph_dfs_edges))?;
    m.add_wrapped(wrap_pyfunction!(digraph_find_cycle))?;
    m.add_wrapped(wrap_pyfunction!(digraph_k_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(graph_k_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(is_matching))?;
    m.add_wrapped(wrap_pyfunction!(is_maximal_matching))?;
    m.add_wrapped(wrap_pyfunction!(max_weight_matching))?;
    m.add_wrapped(wrap_pyfunction!(minimum_spanning_edges))?;
    m.add_wrapped(wrap_pyfunction!(minimum_spanning_tree))?;
    m.add_wrapped(wrap_pyfunction!(graph_transitivity))?;
    m.add_wrapped(wrap_pyfunction!(digraph_transitivity))?;
    m.add_wrapped(wrap_pyfunction!(graph_core_number))?;
    m.add_wrapped(wrap_pyfunction!(digraph_core_number))?;
    m.add_wrapped(wrap_pyfunction!(graph_complement))?;
    m.add_wrapped(wrap_pyfunction!(digraph_complement))?;
    m.add_wrapped(wrap_pyfunction!(graph_random_layout))?;
    m.add_wrapped(wrap_pyfunction!(digraph_random_layout))?;
    m.add_wrapped(wrap_pyfunction!(graph_bipartite_layout))?;
    m.add_wrapped(wrap_pyfunction!(digraph_bipartite_layout))?;
    m.add_wrapped(wrap_pyfunction!(graph_circular_layout))?;
    m.add_wrapped(wrap_pyfunction!(digraph_circular_layout))?;
    m.add_wrapped(wrap_pyfunction!(graph_shell_layout))?;
    m.add_wrapped(wrap_pyfunction!(digraph_shell_layout))?;
    m.add_wrapped(wrap_pyfunction!(graph_spiral_layout))?;
    m.add_wrapped(wrap_pyfunction!(digraph_spiral_layout))?;
    m.add_wrapped(wrap_pyfunction!(graph_spring_layout))?;
    m.add_wrapped(wrap_pyfunction!(digraph_spring_layout))?;
    m.add_wrapped(wrap_pyfunction!(digraph_num_shortest_paths_unweighted))?;
    m.add_wrapped(wrap_pyfunction!(graph_num_shortest_paths_unweighted))?;
    m.add_class::<digraph::PyDiGraph>()?;
    m.add_class::<graph::PyGraph>()?;
    m.add_class::<iterators::BFSSuccessors>()?;
    m.add_class::<iterators::NodeIndices>()?;
    m.add_class::<iterators::EdgeIndices>()?;
    m.add_class::<iterators::EdgeList>()?;
    m.add_class::<iterators::EdgeIndexMap>()?;
    m.add_class::<iterators::WeightedEdgeList>()?;
    m.add_class::<iterators::PathMapping>()?;
    m.add_class::<iterators::PathLengthMapping>()?;
    m.add_class::<iterators::Pos2DMapping>()?;
    m.add_class::<iterators::AllPairsPathLengthMapping>()?;
    m.add_class::<iterators::AllPairsPathMapping>()?;
    m.add_class::<iterators::NodesCountMapping>()?;
    m.add_class::<iterators::NodeMap>()?;
    m.add_wrapped(wrap_pymodule!(generators))?;
    Ok(())
}
