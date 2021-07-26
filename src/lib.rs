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

mod connectivity;
mod core_number;
mod dag_algo;
mod digraph;
mod dot_utils;
mod generators;
mod graph;
mod isomorphism_algo;
mod iterators;
mod layout_algo;
mod matching;
mod random_circuit;
mod shortest_path;
mod simple_path;
mod transitivity;
mod traversal;
mod tree;
mod union;

use std::cmp::Reverse;

use connectivity::*;
use core_number::*;
use dag_algo::*;
use isomorphism_algo::*;
use layout_algo::*;
use matching::*;
use random_circuit::*;
use shortest_path::*;
use simple_path::*;
use transitivity::*;
use traversal::*;
use tree::*;

use hashbrown::{HashMap, HashSet};

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use pyo3::wrap_pymodule;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{
    Data, GraphBase, GraphProp, IntoEdgeReferences, IntoNodeIdentifiers,
    NodeCount, NodeIndexable,
};

use ndarray::prelude::*;
use numpy::IntoPyArray;
use rayon::prelude::*;

use crate::generators::PyInit_generators;

pub trait NodesRemoved {
    fn nodes_removed(&self) -> bool;
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
