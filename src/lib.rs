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

mod centrality;
mod coloring;
mod connectivity;
mod dag_algo;
mod digraph;
mod dot_utils;
mod generators;
mod graph;
mod isomorphism;
mod iterators;
mod layout;
mod matching;
mod random_graph;
mod shortest_path;
mod steiner_tree;
mod transitivity;
mod traversal;
mod tree;
mod union;

use centrality::*;
use coloring::*;
use connectivity::*;
use dag_algo::*;
use isomorphism::*;
use layout::*;
use matching::*;
use random_graph::*;
use shortest_path::*;
use steiner_tree::*;
use transitivity::*;
use traversal::*;
use tree::*;
use union::*;

use hashbrown::HashMap;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::wrap_pymodule;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{
    Data, GraphBase, GraphProp, IntoEdgeReferences, IntoNodeIdentifiers,
    NodeCount, NodeIndexable,
};
use petgraph::EdgeType;

use crate::generators::PyInit_generators;

type StablePyGraph<Ty> = StableGraph<PyObject, PyObject, Ty>;

pub trait NodesRemoved {
    fn nodes_removed(&self) -> bool;
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

fn find_node_by_weight<Ty: EdgeType>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    obj: &PyObject,
) -> PyResult<Option<NodeIndex>> {
    let mut index = None;
    for node in graph.node_indices() {
        let weight = graph.node_weight(node).unwrap();
        if obj
            .as_ref(py)
            .rich_compare(weight, pyo3::basic::CompareOp::Eq)?
            .is_true()?
        {
            index = Some(node);
            break;
        }
    }
    Ok(index)
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
    m.add_wrapped(wrap_pyfunction!(digraph_vf2_mapping))?;
    m.add_wrapped(wrap_pyfunction!(graph_vf2_mapping))?;
    m.add_wrapped(wrap_pyfunction!(digraph_union))?;
    m.add_wrapped(wrap_pyfunction!(graph_union))?;
    m.add_wrapped(wrap_pyfunction!(topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(descendants))?;
    m.add_wrapped(wrap_pyfunction!(ancestors))?;
    m.add_wrapped(wrap_pyfunction!(lexicographical_topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(graph_floyd_warshall))?;
    m.add_wrapped(wrap_pyfunction!(digraph_floyd_warshall))?;
    m.add_wrapped(wrap_pyfunction!(graph_floyd_warshall_numpy))?;
    m.add_wrapped(wrap_pyfunction!(digraph_floyd_warshall_numpy))?;
    m.add_wrapped(wrap_pyfunction!(collect_runs))?;
    m.add_wrapped(wrap_pyfunction!(collect_bicolor_runs))?;
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
    m.add_wrapped(wrap_pyfunction!(graph_betweenness_centrality))?;
    m.add_wrapped(wrap_pyfunction!(digraph_betweenness_centrality))?;
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
    m.add_wrapped(wrap_pyfunction!(
        digraph_unweighted_average_shortest_path_length
    ))?;
    m.add_wrapped(wrap_pyfunction!(
        graph_unweighted_average_shortest_path_length
    ))?;
    m.add_wrapped(wrap_pyfunction!(metric_closure))?;
    m.add_wrapped(wrap_pyfunction!(steiner_tree))?;
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
    m.add_class::<iterators::CentralityMapping>()?;
    m.add_class::<iterators::Pos2DMapping>()?;
    m.add_class::<iterators::AllPairsPathLengthMapping>()?;
    m.add_class::<iterators::AllPairsPathMapping>()?;
    m.add_class::<iterators::NodesCountMapping>()?;
    m.add_class::<iterators::NodeMap>()?;
    m.add_wrapped(wrap_pymodule!(generators))?;
    Ok(())
}
