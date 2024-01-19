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

mod all_pairs_all_simple_paths;
mod johnson_simple_cycles;

use super::{
    digraph, get_edge_iter_with_weights, graph, score, weight_callable, InvalidNode, NullGraph,
};

use hashbrown::{HashMap, HashSet};

use petgraph::algo;
use petgraph::stable_graph::NodeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeCount, NodeIndexable, Visitable};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;
use rayon::prelude::*;

use ndarray::prelude::*;
use numpy::IntoPyArray;

use crate::iterators::{
    AllPairsMultiplePathMapping, BiconnectedComponents, Chains, EdgeList, NodeIndices,
};
use crate::{EdgeType, StablePyGraph};

use rustworkx_core::coloring::two_color;
use rustworkx_core::connectivity;

/// Return a list of cycles which form a basis for cycles of a given PyGraph
///
/// A basis for cycles of a graph is a minimal collection of
/// cycles such that any cycle in the graph can be written
/// as a sum of cycles in the basis.  Here summation of cycles
/// is defined as the exclusive or of the edges.
///
/// This is adapted from algorithm CACM 491 [1]_.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges.
///     It may produce incorrect/unexpected results if the input graph has
///     parallel edges.
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
pub fn cycle_basis(graph: &graph::PyGraph, root: Option<usize>) -> Vec<Vec<usize>> {
    connectivity::cycle_basis(&graph.graph, root.map(NodeIndex::new))
        .into_iter()
        .map(|res_map| res_map.into_iter().map(|x| x.index()).collect())
        .collect()
}

/// Find all simple cycles of a :class:`~.PyDiGraph`
///
/// A "simple cycle" (called an elementary circuit in [1]) is a cycle (or closed path)
/// where no node appears more than once.
///
/// This function is a an implementation of Johnson's algorithm [1] also based
/// on the non-recursive implementation found in NetworkX. [2][3]
///
/// [1] https://doi.org/10.1137/0204007
/// [2] https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.simple_cycles.html
/// [3] https://github.com/networkx/networkx/blob/networkx-2.8.4/networkx/algorithms/cycles.py#L98-L222
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn simple_cycles(graph: &digraph::PyDiGraph) -> johnson_simple_cycles::SimpleCycleIter {
    johnson_simple_cycles::SimpleCycleIter::new(graph)
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
pub fn strongly_connected_components(graph: &digraph::PyDiGraph) -> Vec<Vec<usize>> {
    algo::kosaraju_scc(&graph.graph)
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
pub fn digraph_find_cycle(graph: &digraph::PyDiGraph, source: Option<usize>) -> EdgeList {
    EdgeList {
        edges: connectivity::find_cycle(&graph.graph, source.map(NodeIndex::new))
            .iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect(),
    }
}

/// Find the number of connected components in an undirected graph.
///
/// :param PyGraph graph: The graph to find the number of connected
///     components on.
///
/// :returns: The number of connected components in the graph
/// :rtype: int
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn number_connected_components(graph: &graph::PyGraph) -> usize {
    connectivity::number_connected_components(&graph.graph)
}

/// Find the connected components in an undirected graph
///
/// :param PyGraph graph: The graph to find the connected components.
///
/// :returns: A list of sets where each set is a connected component of
///     the graph
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn connected_components(graph: &graph::PyGraph) -> Vec<HashSet<usize>> {
    connectivity::connected_components(&graph.graph)
        .into_iter()
        .map(|res_map| res_map.into_iter().map(|x| x.index()).collect())
        .collect()
}

/// Returns the set of nodes in the component of graph containing `node`.
///
/// :param PyGraph graph: The graph to be used.
/// :param int node: A node in the graph.
///
/// :returns: A set of nodes in the component of graph containing `node`.
/// :rtype: set
///
/// :raises InvalidNode: When an invalid node index is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, node, /)")]
pub fn node_connected_component(graph: &graph::PyGraph, node: usize) -> PyResult<HashSet<usize>> {
    let node = NodeIndex::new(node);

    if !graph.graph.contains_node(node) {
        return Err(InvalidNode::new_err(
            "The input index for 'node' is not a valid node index",
        ));
    }

    Ok(
        connectivity::bfs_undirected(&graph.graph, node, &mut graph.graph.visit_map())
            .into_iter()
            .map(|x| x.index())
            .collect(),
    )
}

/// Check if the graph is connected.
///
/// :param PyGraph graph: The graph to check if it is connected.
///
/// :returns: Whether the graph is connected or not
/// :rtype: bool
///
/// :raises NullGraph: If an empty graph is passed in
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn is_connected(graph: &graph::PyGraph) -> PyResult<bool> {
    match graph.graph.node_indices().next() {
        Some(node) => {
            let component = node_connected_component(graph, node.index())?;
            Ok(component.len() == graph.graph.node_count())
        }
        None => Err(NullGraph::new_err("Invalid operation on a NullGraph")),
    }
}

/// Find the number of weakly connected components in a directed graph
///
/// :param PyDiGraph graph: The graph to find the number of weakly connected
///     components on
///
/// :returns: The number of weakly connected components in the graph
/// :rtype: int
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn number_weakly_connected_components(graph: &digraph::PyDiGraph) -> usize {
    let mut weak_components = graph.node_count();
    let mut vertex_sets = UnionFind::new(graph.graph.node_bound());
    for edge in graph.graph.edge_references() {
        let (a, b) = (edge.source(), edge.target());
        // union the two vertices of the edge
        if vertex_sets.union(a.index(), b.index()) {
            weak_components -= 1
        };
    }
    weak_components
}

/// Find the weakly connected components in a directed graph
///
/// :param PyDiGraph graph: The graph to find the weakly connected components
///     in
///
/// :returns: A list of sets where each set is a weakly connected component of
///     the graph
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn weakly_connected_components(graph: &digraph::PyDiGraph) -> Vec<HashSet<usize>> {
    connectivity::connected_components(&graph.graph)
        .into_iter()
        .map(|res_map| res_map.into_iter().map(|x| x.index()).collect())
        .collect()
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

/// Return the adjacency matrix for a PyDiGraph object
///
/// In the case where there are multiple edges between nodes the value in the
/// output matrix will be assigned based on a given parameter. Currently, the minimum, maximum, average, and default sum are supported.
///
/// :param PyDiGraph graph: The DiGraph used to generate the adjacency matrix
///     from
/// :param callable weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells rustworkx/rust how to extract a numerical weight as a ``float``
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
/// :param float null_value: An optional float that will treated as a null
///     value. This is the default value in the output matrix and it is used
///     to indicate the absence of an edge between 2 nodes. By default this is
///     ``0.0``.
/// :param String parallel_edge: Optional argument that determines how the function handles parallel edges.
///     ``"min"`` causes the value in the output matrix to be the minimum of the edges' weights, and similar behavior can be expected for ``"max"`` and ``"avg"``.
///     The function defaults to ``"sum"`` behavior, where the value in the output matrix is the sum of all parallel edge weights.
///
///  :return: The adjacency matrix for the input directed graph as a numpy array
///  :rtype: numpy.ndarray
#[pyfunction]
#[pyo3(
    signature=(graph, weight_fn=None, default_weight=1.0, null_value=0.0, parallel_edge="sum"),
    text_signature = "(graph, /, weight_fn=None, default_weight=1.0, null_value=0.0, parallel_edge=\"sum\")"
)]
pub fn digraph_adjacency_matrix(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    null_value: f64,
    parallel_edge: &str,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::from_elem((n, n), null_value);
    let mut parallel_edge_count = HashMap::new();
    for (i, j, weight) in get_edge_iter_with_weights(&graph.graph) {
        let edge_weight = weight_callable(py, &weight_fn, &weight, default_weight)?;
        if matrix[[i, j]] == null_value || (null_value.is_nan() && matrix[[i, j]].is_nan()) {
            matrix[[i, j]] = edge_weight;
        } else {
            match parallel_edge {
                "sum" => {
                    matrix[[i, j]] += edge_weight;
                }
                "min" => {
                    let weight_min = matrix[[i, j]].min(edge_weight);
                    matrix[[i, j]] = weight_min;
                }
                "max" => {
                    let weight_max = matrix[[i, j]].max(edge_weight);
                    matrix[[i, j]] = weight_max;
                }
                "avg" => {
                    if parallel_edge_count.contains_key(&[i, j]) {
                        matrix[[i, j]] = (matrix[[i, j]] * parallel_edge_count[&[i, j]] as f64
                            + edge_weight)
                            / ((parallel_edge_count[&[i, j]] + 1) as f64);
                        *parallel_edge_count.get_mut(&[i, j]).unwrap() += 1;
                    } else {
                        parallel_edge_count.insert([i, j], 2);
                        matrix[[i, j]] = (matrix[[i, j]] + edge_weight) / 2.0;
                    }
                }
                _ => {
                    return Err(PyValueError::new_err("Parallel edges can currently only be dealt with using \"sum\", \"min\", \"max\", or \"avg\"."));
                }
            }
        }
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Return the adjacency matrix for a PyGraph class
///
/// In the case where there are multiple edges between nodes the value in the
/// output matrix will be assigned based on a given parameter. Currently, the minimum, maximum, average, and default sum are supported.
///
/// :param PyGraph graph: The graph used to generate the adjacency matrix from
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells rustworkx/rust how to extract a numerical weight as a ``float``
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
/// :param float null_value: An optional float that will treated as a null
///     value. This is the default value in the output matrix and it is used
///     to indicate the absence of an edge between 2 nodes. By default this is
///     ``0.0``.
/// :param String parallel_edge: Optional argument that determines how the function handles parallel edges.
///     ``"min"`` causes the value in the output matrix to be the minimum of the edges' weights, and similar behavior can be expected for ``"max"`` and ``"avg"``.
///     The function defaults to ``"sum"`` behavior, where the value in the output matrix is the sum of all parallel edge weights.
///
/// :return: The adjacency matrix for the input graph as a numpy array
/// :rtype: numpy.ndarray
#[pyfunction]
#[pyo3(
    signature=(graph, weight_fn=None, default_weight=1.0, null_value=0.0, parallel_edge="sum"),
    text_signature = "(graph, /, weight_fn=None, default_weight=1.0, null_value=0.0, parallel_edge=\"sum\")"
)]
pub fn graph_adjacency_matrix(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    null_value: f64,
    parallel_edge: &str,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::from_elem((n, n), null_value);
    let mut parallel_edge_count = HashMap::new();
    for (i, j, weight) in get_edge_iter_with_weights(&graph.graph) {
        let edge_weight = weight_callable(py, &weight_fn, &weight, default_weight)?;
        if matrix[[i, j]] == null_value || (null_value.is_nan() && matrix[[i, j]].is_nan()) {
            matrix[[i, j]] = edge_weight;
            matrix[[j, i]] = edge_weight;
        } else {
            match parallel_edge {
                "sum" => {
                    matrix[[i, j]] += edge_weight;
                    matrix[[j, i]] += edge_weight;
                }
                "min" => {
                    let weight_min = matrix[[i, j]].min(edge_weight);
                    matrix[[i, j]] = weight_min;
                    matrix[[j, i]] = weight_min;
                }
                "max" => {
                    let weight_max = matrix[[i, j]].max(edge_weight);
                    matrix[[i, j]] = weight_max;
                    matrix[[j, i]] = weight_max;
                }
                "avg" => {
                    if parallel_edge_count.contains_key(&[i, j]) {
                        matrix[[i, j]] = (matrix[[i, j]] * parallel_edge_count[&[i, j]] as f64
                            + edge_weight)
                            / ((parallel_edge_count[&[i, j]] + 1) as f64);
                        matrix[[j, i]] = (matrix[[j, i]] * parallel_edge_count[&[i, j]] as f64
                            + edge_weight)
                            / ((parallel_edge_count[&[i, j]] + 1) as f64);
                        *parallel_edge_count.get_mut(&[i, j]).unwrap() += 1;
                    } else {
                        parallel_edge_count.insert([i, j], 2);
                        matrix[[i, j]] = (matrix[[i, j]] + edge_weight) / 2.0;
                        matrix[[j, i]] = (matrix[[j, i]] + edge_weight) / 2.0;
                    }
                }
                _ => {
                    return Err(PyValueError::new_err("Parallel edges can currently only be dealt with using \"sum\", \"min\", \"max\", or \"avg\"."));
                }
            }
        }
    }
    Ok(matrix.into_pyarray(py).into())
}

/// Compute the complement of an undirected graph.
///
/// :param PyGraph graph: The graph to be used.
///
/// :returns: The complement of the graph.
/// :rtype: PyGraph
///
/// .. note::
///
///     Parallel edges and self-loops are never created,
///     even if the :attr:`~rustworkx.PyGraph.multigraph`
///     attribute is set to ``True``
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_complement(py: Python, graph: &graph::PyGraph) -> PyResult<graph::PyGraph> {
    let mut complement_graph = graph.clone(); // keep same node indices
    complement_graph.graph.clear_edges();

    for node_a in graph.graph.node_indices() {
        let old_neighbors: HashSet<NodeIndex> = graph.graph.neighbors(node_a).collect();
        for node_b in graph.graph.node_indices() {
            if node_a != node_b
                && !old_neighbors.contains(&node_b)
                && (!complement_graph.multigraph
                    || !complement_graph.has_edge(node_a.index(), node_b.index()))
            {
                // avoid creating parallel edges in multigraph
                complement_graph.graph.add_edge(node_a, node_b, py.None());
            }
        }
    }
    Ok(complement_graph)
}

/// Compute the complement of a directed graph.
///
/// :param PyDiGraph graph: The graph to be used.
///
/// :returns: The complement of the graph.
/// :rtype: :class:`~rustworkx.PyDiGraph`
///
/// .. note::
///
///     Parallel edges and self-loops are never created,
///     even if the :attr:`~rustworkx.PyDiGraph.multigraph`
///     attribute is set to ``True``
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn digraph_complement(py: Python, graph: &digraph::PyDiGraph) -> PyResult<digraph::PyDiGraph> {
    let mut complement_graph = graph.clone(); // keep same node indices
    complement_graph.graph.clear_edges();

    for node_a in graph.graph.node_indices() {
        let old_neighbors: HashSet<NodeIndex> = graph
            .graph
            .neighbors_directed(node_a, petgraph::Direction::Outgoing)
            .collect();
        for node_b in graph.graph.node_indices() {
            if node_a != node_b && !old_neighbors.contains(&node_b) {
                complement_graph.add_edge(node_a.index(), node_b.index(), py.None())?;
            }
        }
    }

    Ok(complement_graph)
}

/// Return all simple paths between 2 nodes in a PyGraph object
///
/// A simple path is a path with no repeated nodes.
///
/// :param PyGraph graph: The graph to find the path in
/// :param int origin: The node index to find the paths from
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
#[pyo3(text_signature = "(graph, origin, to, /, min_depth=None, cutoff=None)")]
pub fn graph_all_simple_paths(
    graph: &graph::PyGraph,
    origin: usize,
    to: usize,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let from_index = NodeIndex::new(origin);
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
        Some(0) | None => 0,
        Some(depth) => depth - 2,
    };
    let cutoff_petgraph: Option<usize> = cutoff.map(|depth| depth - 2);
    let result: Vec<Vec<usize>> = algo::all_simple_paths(
        &graph.graph,
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
/// :param int origin: The node index to find the paths from
/// :param int to: The node index to find the paths to
/// :param int min_depth: The minimum depth of the path to include in the output
///     list of paths. By default all paths are included regardless of depth,
///     setting to 0 will behave like the default.
/// :param int cutoff: The maximum depth of path to include in the output list
///     of paths. By default includes all paths regardless of depth, setting to
///     0 will behave like default.
///
/// :returns: A list of lists where each inner list is a path
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, origin, to, /, min_depth=None, cutoff=None)")]
pub fn digraph_all_simple_paths(
    graph: &digraph::PyDiGraph,
    origin: usize,
    to: usize,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let from_index = NodeIndex::new(origin);
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
        Some(0) | None => 0,
        Some(depth) => depth - 2,
    };
    let cutoff_petgraph: Option<usize> = cutoff.map(|depth| depth - 2);
    let result: Vec<Vec<usize>> = algo::all_simple_paths(
        &graph.graph,
        from_index,
        to_index,
        min_intermediate_nodes,
        cutoff_petgraph,
    )
    .map(|v: Vec<NodeIndex>| v.into_iter().map(|i| i.index()).collect())
    .collect();
    Ok(result)
}

/// Return all the simple paths between all pairs of nodes in the graph
///
/// This function is multithreaded and will launch a thread pool with threads
/// equal to the number of CPUs by default. You can tune the number of threads
/// with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// :param PyDiGraph graph: The graph to find all simple paths in
/// :param int min_depth: The minimum depth of the path to include in the output
///     list of paths. By default all paths are included regardless of depth,
///     setting to 0 will behave like the default.
/// :param int cutoff: The maximum depth of path to include in the output list
///     of paths. By default includes all paths regardless of depth, setting to
///     0 will behave like default.
///
/// :returns: A mapping of source node indices to a mapping of target node
///     indices to a list of paths between the source and target nodes.
/// :rtype: AllPairsMultiplePathMapping
///
/// :raises ValueError: If ``min_depth`` or ``cutoff`` are < 2
#[pyfunction]
#[pyo3(text_signature = "(graph, /, min_depth=None, cutoff=None)")]
pub fn digraph_all_pairs_all_simple_paths(
    graph: &digraph::PyDiGraph,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<AllPairsMultiplePathMapping> {
    if min_depth.is_some() && min_depth < Some(2) {
        return Err(PyValueError::new_err("Value for min_depth must be >= 2"));
    }
    if cutoff.is_some() && cutoff < Some(2) {
        return Err(PyValueError::new_err("Value for cutoff must be >= 2"));
    }

    Ok(all_pairs_all_simple_paths::all_pairs_all_simple_paths(
        &graph.graph,
        min_depth,
        cutoff,
    ))
}

/// Return all the simple paths between all pairs of nodes in the graph
///
/// This function is multithreaded and will launch a thread pool with threads
/// equal to the number of CPUs by default. You can tune the number of threads
/// with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// :param PyGraph graph: The graph to find all simple paths in
/// :param int min_depth: The minimum depth of the path to include in the output
///     list of paths. By default all paths are included regardless of depth,
///     setting to 0 will behave like the default.
/// :param int cutoff: The maximum depth of path to include in the output list
///     of paths. By default includes all paths regardless of depth, setting to
///     0 will behave like default.
///
/// :returns: A mapping of node indices to to a mapping of target node
///     indices to a list of paths between the source and target nodes.
/// :rtype: AllPairsMultiplePathMapping
///
/// :raises ValueError: If ``min_depth`` or ``cutoff`` are < 2.
#[pyfunction]
#[pyo3(text_signature = "(graph, /, min_depth=None, cutoff=None)")]
pub fn graph_all_pairs_all_simple_paths(
    graph: &graph::PyGraph,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<AllPairsMultiplePathMapping> {
    if min_depth.is_some() && min_depth < Some(2) {
        return Err(PyValueError::new_err("Value for min_depth must be >= 2"));
    }
    if cutoff.is_some() && cutoff < Some(2) {
        return Err(PyValueError::new_err("Value for cutoff must be >= 2"));
    }

    Ok(all_pairs_all_simple_paths::all_pairs_all_simple_paths(
        &graph.graph,
        min_depth,
        cutoff,
    ))
}

fn longest_simple_path<Ty: EdgeType + Sync + Send>(
    graph: &StablePyGraph<Ty>,
) -> Option<NodeIndices> {
    if graph.node_count() == 0 {
        return None;
    } else if graph.edge_count() == 0 {
        return Some(NodeIndices {
            nodes: vec![graph.node_indices().next()?.index()],
        });
    }
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    let node_index_set = node_indices.iter().copied().collect();
    Some(NodeIndices {
        nodes: node_indices
            .par_iter()
            .filter_map(|u| {
                connectivity::longest_simple_path_multiple_targets(graph, *u, &node_index_set)
            })
            .max_by_key(|x| x.len())
            .unwrap()
            .into_iter()
            .map(|x| x.index())
            .collect(),
    })
}

/// Return a longest simple path in the graph
///
/// This function searches computes all pairs of all simple paths and returns
/// a path of the longest length from that set. It is roughly equivalent to
/// running something like::
///
///     from rustworkx import all_pairs_all_simple_paths
///
///     max((y.values for y in all_pairs_all_simple_paths(graph).values()), key=lambda x: len(x))
///
/// but this function will be more efficient than using ``max()`` as the search
/// is evaluated in parallel before returning to Python. In the case of multiple
/// paths of the same maximum length being present in the graph only one will be
/// provided. There are no guarantees on which of the multiple longest paths
/// will be returned (as it is determined by the parallel execution order). This
/// is a tradeoff to improve runtime performance. If a stable return is required
/// in such case consider using the ``max()`` equivalent above instead.
///
/// This function is multithreaded and will launch a thread pool with threads
/// equal to the number of CPUs by default. You can tune the number of threads
/// with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// :param PyDiGraph graph: The graph to find the longest path in
///
/// :returns: A sequence of node indices that represent the longest simple graph
///     found in the graph. If the graph is empty ``None`` will be returned instead.
/// :rtype: NodeIndices
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn digraph_longest_simple_path(graph: &digraph::PyDiGraph) -> Option<NodeIndices> {
    longest_simple_path(&graph.graph)
}

/// Return a longest simple path in the graph
///
/// This function searches computes all pairs of all simple paths and returns
/// a path of the longest length from that set. It is roughly equivalent to
/// running something like::
///
///     from rustworkx import all_pairs_all_simple_paths
///
///     simple_path_pairs = rx.all_pairs_all_simple_paths(graph)
///     longest_path = max(
///         (u for y in simple_path_pairs.values() for z in y.values() for u in z),
///         key=lambda x: len(x),
///     )
///
/// but this function will be more efficient than using ``max()`` as the search
/// is evaluated in parallel before returning to Python. In the case of multiple
/// paths of the same maximum length being present in the graph only one will be
/// provided. There are no guarantees on which of the multiple longest paths
/// will be returned (as it is determined by the parallel execution order). This
/// is a tradeoff to improve runtime performance. If a stable return is required
/// in such case consider using the ``max()`` equivalent above instead.
///
/// This function is multithreaded and will launch a thread pool with threads
/// equal to the number of CPUs by default. You can tune the number of threads
/// with the ``RAYON_NUM_THREADS`` environment variable. For example, setting
/// ``RAYON_NUM_THREADS=4`` would limit the thread pool to 4 threads.
///
/// :param PyGraph graph: The graph to find the longest path in
///
/// :returns: A sequence of node indices that represent the longest simple graph
///     found in the graph. If the graph is empty ``None`` will be returned instead.
/// :rtype: NodeIndices
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_longest_simple_path(graph: &graph::PyGraph) -> Option<NodeIndices> {
    longest_simple_path(&graph.graph)
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
pub fn graph_core_number(py: Python, graph: &graph::PyGraph) -> PyResult<PyObject> {
    let cores = connectivity::core_number(&graph.graph);
    let out_dict = PyDict::new(py);
    for (k, v) in cores {
        out_dict.set_item(k.index(), v)?;
    }
    Ok(out_dict.into())
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
pub fn digraph_core_number(py: Python, graph: &digraph::PyDiGraph) -> PyResult<PyObject> {
    let cores = connectivity::core_number(&graph.graph);
    let out_dict = PyDict::new(py);
    for (k, v) in cores {
        out_dict.set_item(k.index(), v)?;
    }
    Ok(out_dict.into())
}

/// Compute a weighted minimum cut using the Stoer-Wagner algorithm.
///
/// Determine the minimum cut of a graph using the Stoer-Wagner algorithm [stoer_simple_1997]_.
/// All weights must be nonnegative. If the input graph is disconnected,
/// a cut with zero value will be returned. For graphs with less than
/// two nodes, this function returns ``None``.
///
/// :param PyGraph: The graph to be used
/// :param Callable weight_fn:  An optional callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``.
///     Edges with ``NaN`` weights will be ignored, i.e it's conidered to have zero weight.
///     If ``weight_fn`` is not specified a default value of ``1.0`` will be used for all edges.
///
/// :returns: A tuple with the minimum cut value and a list of all
///     the node indexes contained in one part of the partition
///     that defines a minimum cut.
/// :rtype: (usize, NodeIndices)
///
/// .. [stoer_simple_1997] Stoer, Mechthild and Frank Wagner, "A simple min-cut
///     algorithm". Journal of the ACM 44 (4), 585-591, 1997.
#[pyfunction]
#[pyo3(text_signature = "(graph, /, weight_fn=None)")]
pub fn stoer_wagner_min_cut(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
) -> PyResult<Option<(f64, NodeIndices)>> {
    let cut = connectivity::stoer_wagner_min_cut(&graph.graph, |edge| -> PyResult<_> {
        let val: f64 = weight_callable(py, &weight_fn, edge.weight(), 1.0)?;
        if val.is_nan() {
            Ok(score::Score(0.0))
        } else {
            Ok(score::Score(val))
        }
    })?;

    Ok(cut.map(|(value, partition)| {
        (
            value.0,
            NodeIndices {
                nodes: partition.iter().map(|&nx| nx.index()).collect(),
            },
        )
    }))
}

/// Return the articulation points of an undirected graph.
///
/// An articulation point or cut vertex is any node whose removal (along with
/// all its incident edges) increases the number of connected components of
/// a graph. An undirected connected graph without articulation points is
/// biconnected.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyGraph: The undirected graph to be used.
///
/// :returns: A set with node indices of the articulation points in the graph.
/// :rtype: set
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn articulation_points(graph: &graph::PyGraph) -> HashSet<usize> {
    connectivity::articulation_points(&graph.graph, None)
        .into_iter()
        .map(|nx| nx.index())
        .collect()
}

/// Return the bridges of an undirected graph.
///
/// A bridge is any edge whose removal increases the number of connected
/// components of a graph.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyGraph: The undirected graph to be used.
///
/// :returns: A set with edges of the bridges in the graph, each edge is 
///     represented by a pair of node index.
/// :rtype: set
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn bridges(graph: &graph::PyGraph) -> HashSet<(usize, usize)> {
    let bridges = connectivity::bridges(&graph.graph);
    bridges
        .into_iter()
        .map(|(a, b)| (a.index(), b.index()))
        .collect()
}

/// Return the biconnected components of an undirected graph.
///
/// Biconnected components are maximal subgraphs such that the removal
/// of a node (and all edges incident on that node) will not disconnect
/// the subgraph. Note that nodes may be part of more than one biconnected
/// component. Those nodes are articulation points, or cut vertices. The
/// algorithm computes how many biconnected components are in the graph,
/// and assigning each component an integer label.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyGraph: The undirected graph to be used.
///
/// :returns: A dictionary with keys the edge endpoints and value the biconnected
///     component number that the edge belongs.
/// :rtype: dict
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn biconnected_components(graph: &graph::PyGraph) -> BiconnectedComponents {
    let mut bicomp = HashMap::new();
    connectivity::articulation_points(&graph.graph, Some(&mut bicomp));

    BiconnectedComponents {
        bicon_comp: bicomp
            .into_iter()
            .map(|((v, w), comp)| ((v.index(), w.index()), comp))
            .collect(),
    }
}

/// Returns the chain decomposition of a graph.
///
/// The *chain decomposition* of a graph with respect to a depth-first
/// search tree is a set of cycles or paths derived from the set of
/// fundamental cycles of the tree in the following manner. Consider
/// each fundamental cycle with respect to the given tree, represented
/// as a list of edges beginning with the nontree edge oriented away
/// from the root of the tree. For each fundamental cycle, if it
/// overlaps with any previous fundamental cycle, just take the initial
/// non-overlapping segment, which is a path instead of a cycle. Each
/// cycle or path is called a *chain*. For more information, see [Schmidt]_.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyGraph: The undirected graph to be used
/// :param int source: An optional node index in the graph. If specified,
///     only the chain decomposition for the connected component containing
///     this node will be returned. This node indicates the root of the depth-first
///     search tree. If this is not specified then a source will be chosen
///     arbitrarly and repeated until all components of the graph are searched.
/// :returns: A list of list of edges where each inner list is a chain.
/// :rtype: list of EdgeList
///
/// .. [Schmidt] Jens M. Schmidt (2013). "A simple test on 2-vertex-
///       and 2-edge-connectivity." *Information Processing Letters*,
///       113, 241–244. Elsevier. <https://doi.org/10.1016/j.ipl.2013.01.016>
#[pyfunction]
#[pyo3(text_signature = "(graph, /, source=None)")]
pub fn chain_decomposition(graph: graph::PyGraph, source: Option<usize>) -> Chains {
    let chains = connectivity::chain_decomposition(&graph.graph, source.map(NodeIndex::new));
    Chains {
        chains: chains
            .into_iter()
            .map(|chain| EdgeList {
                edges: chain
                    .into_iter()
                    .map(|(a, b)| (a.index(), b.index()))
                    .collect(),
            })
            .collect(),
    }
}

/// Return a list of isolates in a :class:`~.PyGraph` object
///
/// An isolate is a node without any neighbors meaning it has a degree of 0.
///
/// :param PyGraph graph: The input graph to find isolates in
/// :returns: A list of node indices for isolates in the graph
/// :rtype: NodeIndices
#[pyfunction]
pub fn graph_isolates(graph: graph::PyGraph) -> NodeIndices {
    NodeIndices {
        nodes: connectivity::isolates(&graph.graph)
            .into_iter()
            .map(|x| x.index())
            .collect(),
    }
}

/// Return a list of isolates in a :class:`~.PyGraph` object
///
/// An isolate is a node without any neighbors meaning it has an in-degree
/// and out-degree of 0.
///
/// :param PyGraph graph: The input graph to find isolates in
/// :returns: A list of node indices for isolates in the graph
/// :rtype: NodeIndices
#[pyfunction]
pub fn digraph_isolates(graph: digraph::PyDiGraph) -> NodeIndices {
    NodeIndices {
        nodes: connectivity::isolates(&graph.graph)
            .into_iter()
            .map(|x| x.index())
            .collect(),
    }
}

/// Determine if a given graph is bipartite
///
/// :param PyGraph graph: The graph to check if it's bipartite
/// :returns: ``True`` if the graph is bipartite and ``False`` if it is not
/// :rtype: bool
#[pyfunction]
pub fn graph_is_bipartite(graph: graph::PyGraph) -> bool {
    two_color(&graph.graph).is_some()
}

/// Determine if a given graph is bipartite
///
/// :param PyDiGraph graph: The graph to check if it's bipartite
/// :returns: ``True`` if the graph is bipartite and ``False`` if it is not
/// :rtype: bool
#[pyfunction]
pub fn digraph_is_bipartite(graph: digraph::PyDiGraph) -> bool {
    two_color(&graph.graph).is_some()
}
