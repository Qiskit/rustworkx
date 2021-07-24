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
mod dag_algorithms;
mod digraph;
mod dijkstra;
mod dot_utils;
mod generators;
mod graph;
mod isomorphism;
mod iterators;
mod k_shortest_path;
mod layout;
mod matching;
mod max_weight_matching_algo;
mod random_circuit;
mod shortest_path;
mod union;

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap};

use dag_algorithms::*;
use layout::*;
use matching::*;
use random_circuit::*;
use shortest_path::*;

use hashbrown::{HashMap, HashSet};

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
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
    Bfs, Data, GraphBase, GraphProp, IntoEdgeReferences, IntoNeighbors,
    IntoNodeIdentifiers, NodeCount, NodeIndexable, Reversed, VisitMap,
    Visitable,
};
use petgraph::EdgeType;

use ndarray::prelude::*;
use num_traits::{Num, Zero};
use numpy::IntoPyArray;
use rayon::prelude::*;

use crate::generators::PyInit_generators;
use crate::iterators::{EdgeList, NodeIndices, WeightedEdgeList};

pub trait NodesRemoved {
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

pub fn get_edge_iter_with_weights<G>(
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
