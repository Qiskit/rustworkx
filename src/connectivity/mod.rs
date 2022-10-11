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
mod core_number;
mod johnson_simple_cycles;

use super::{
    digraph, get_edge_iter_with_weights, graph, iterators::NodeIndices, score, weight_callable,
    InvalidNode, NullGraph,
};

use hashbrown::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Python;

use petgraph::algo;
use petgraph::stable_graph::NodeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeCount, NodeIndexable, Visitable};

use ndarray::prelude::*;
use numpy::IntoPyArray;

use crate::iterators::{AllPairsMultiplePathMapping, BiconnectedComponents, Chains, EdgeList};
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
    let mut root_node = root;
    let mut graph_nodes: HashSet<NodeIndex> = graph.graph.node_indices().collect();
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
    let mut graph_nodes: HashSet<NodeIndex> = graph.graph.node_indices().collect();
    let mut cycle: Vec<(usize, usize)> = Vec::with_capacity(graph.graph.edge_count());
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
/// output matrix will be the sum of the edges' weights.
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
///
///  :return: The adjacency matrix for the input directed graph as a numpy array
///  :rtype: numpy.ndarray
#[pyfunction(default_weight = "1.0", null_value = "0.0")]
#[pyo3(text_signature = "(graph, /, weight_fn=None, default_weight=1.0, null_value=0.0)")]
pub fn digraph_adjacency_matrix(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    null_value: f64,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::from_elem((n, n), null_value);
    for (i, j, weight) in get_edge_iter_with_weights(&graph.graph) {
        let edge_weight = weight_callable(py, &weight_fn, &weight, default_weight)?;
        if matrix[[i, j]] == null_value || (null_value.is_nan() && matrix[[i, j]].is_nan()) {
            matrix[[i, j]] = edge_weight;
        } else {
            matrix[[i, j]] += edge_weight;
        }
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
///
/// :return: The adjacency matrix for the input graph as a numpy array
/// :rtype: numpy.ndarray
#[pyfunction(default_weight = "1.0", null_value = "0.0")]
#[pyo3(text_signature = "(graph, /, weight_fn=None, default_weight=1.0, null_value=0.0)")]
pub fn graph_adjacency_matrix(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    null_value: f64,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::from_elem((n, n), null_value);
    for (i, j, weight) in get_edge_iter_with_weights(&graph.graph) {
        let edge_weight = weight_callable(py, &weight_fn, &weight, default_weight)?;
        if matrix[[i, j]] == null_value || (null_value.is_nan() && matrix[[i, j]].is_nan()) {
            matrix[[i, j]] = edge_weight;
            matrix[[j, i]] = edge_weight;
        } else {
            matrix[[i, j]] += edge_weight;
            matrix[[j, i]] += edge_weight;
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
                complement_graph.add_edge(node_a.index(), node_b.index(), py.None());
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
#[pyo3(text_signature = "(graph, from, to, /, min_depth=None, cutoff=None)")]
pub fn graph_all_simple_paths(
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
pub fn digraph_all_simple_paths(
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
    core_number::core_number(py, &graph.graph)
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
    core_number::core_number(py, &graph.graph)
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
