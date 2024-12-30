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

mod bisimulation;
mod cartesian_product;
mod centrality;
mod coloring;
mod connectivity;
mod dag_algo;
mod digraph;
mod dominance;
mod dot_utils;
mod generators;
mod graph;
mod graphml;
mod isomorphism;
mod iterators;
mod json;
mod layout;
mod line_graph;
mod link_analysis;
mod matching;
mod planar;
mod random_graph;
mod score;
mod shortest_path;
mod steiner_tree;
mod tensor_product;
mod token_swapper;
mod toposort;
mod transitivity;
mod traversal;
mod tree;
mod union;

use bisimulation::*;
use cartesian_product::*;
use centrality::*;
use coloring::*;
use connectivity::*;
use dag_algo::*;
use dominance::*;
use graphml::*;
use isomorphism::*;
use json::*;
use layout::*;
use line_graph::*;
use link_analysis::*;

use matching::*;
use planar::*;
use random_graph::*;
use shortest_path::*;
use steiner_tree::*;
use tensor_product::*;
use token_swapper::*;
use transitivity::*;
use traversal::*;
use tree::*;
use union::*;

use hashbrown::HashMap;
use numpy::Complex64;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyValueError;
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::wrap_pymodule;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{
    Data, EdgeIndexable, GraphBase, GraphProp, IntoEdgeReferences, IntoNodeIdentifiers, NodeCount,
    NodeIndexable,
};
use petgraph::EdgeType;

use rustworkx_core::dag_algo::TopologicalSortError;
use std::convert::TryFrom;

use rustworkx_core::dictmap::*;
use rustworkx_core::err::{ContractError, ContractSimpleError};

/// An ergonomic error type used to map Rustworkx core errors to
/// [PyErr] automatically, via [From::from].
///
/// It is constructable from both [PyErr] and core errors and implements
/// [IntoPy], so it can be returned directly from PyO3 methods and
/// functions. Additionally, a [PyErr] can be constructed from this
/// type, since it's just a wrapper around one, so you can even go
/// from a core error => [RxPyErr] => [PyErr].
///
/// # Usage
/// When calling Rustworkx core functions from PyO3 code, use
/// [RxPyResult] as the return type of the calling function and use
/// the `?` operator to unwrap the result with error propagation.
/// Since Rust automatically applies [From::from] to unwrapped error
/// values, a core error will be automatically converted to a
/// Python-friendly error and stored in [RxPyErr], assuming you've
/// added an implementation of [From] for it below. The standard
/// [PyErr] type will be converted to [RxPyErr] using the same
/// mechanism, allowing Rustworkx core and PyO3 API usage to be
/// intermixed within the same calling function.
pub struct RxPyErr {
    pyerr: PyErr,
}

/// Type alias for a [Result] with error type [RxPyErr].
pub type RxPyResult<T> = Result<T, RxPyErr>;

fn map_dag_would_cycle<E: std::error::Error>(value: E) -> PyErr {
    DAGWouldCycle::new_err(format!("{}", value))
}

impl From<ContractError> for RxPyErr {
    fn from(value: ContractError) -> Self {
        RxPyErr {
            pyerr: match value {
                ContractError::DAGWouldCycle => map_dag_would_cycle(value),
            },
        }
    }
}

impl From<ContractSimpleError<PyErr>> for RxPyErr {
    fn from(value: ContractSimpleError<PyErr>) -> Self {
        RxPyErr {
            pyerr: match value {
                ContractSimpleError::DAGWouldCycle => map_dag_would_cycle(value),
                ContractSimpleError::MergeError(e) => e,
            },
        }
    }
}

impl From<TopologicalSortError<PyErr>> for RxPyErr {
    fn from(value: TopologicalSortError<PyErr>) -> Self {
        RxPyErr {
            pyerr: match value {
                TopologicalSortError::CycleOrBadInitialState => {
                    PyValueError::new_err(format!("{}", value))
                }
                TopologicalSortError::KeyError(e) => e,
            },
        }
    }
}

impl IntoPy<PyObject> for RxPyErr {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.pyerr.into_value(py).into()
    }
}

impl From<RxPyErr> for PyErr {
    fn from(value: RxPyErr) -> Self {
        value.pyerr
    }
}

impl From<PyErr> for RxPyErr {
    fn from(value: PyErr) -> Self {
        RxPyErr { pyerr: value }
    }
}

trait IsNan {
    fn is_nan(&self) -> bool;
}

/// https://doc.rust-lang.org/nightly/src/core/num/f64.rs.html#441
impl IsNan for f64 {
    #[inline]
    #[allow(clippy::eq_op)]
    fn is_nan(&self) -> bool {
        self != self
    }
}

/// https://docs.rs/num-complex/0.4.0/src/num_complex/lib.rs.html#572-574
impl IsNan for Complex64 {
    #[inline]
    fn is_nan(&self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}
pub type StablePyGraph<Ty> = StableGraph<PyObject, PyObject, Ty>;

pub trait NodesRemoved {
    fn nodes_removed(&self) -> bool;
}

impl<Ty> NodesRemoved for &StablePyGraph<Ty>
where
    Ty: EdgeType,
{
    fn nodes_removed(&self) -> bool {
        self.node_bound() != self.node_count()
    }
}

pub fn get_edge_iter_with_weights<G>(graph: G) -> impl Iterator<Item = (usize, usize, PyObject)>
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
                let source_index = NodeIndex::new(graph.to_index(edge.source()));
                let target_index = NodeIndex::new(graph.to_index(edge.target()));
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

fn weight_callable<'p, T>(
    py: Python<'p>,
    weight_fn: &'p Option<PyObject>,
    weight: &PyObject,
    default: T,
) -> PyResult<T>
where
    T: FromPyObject<'p>,
{
    match weight_fn {
        Some(weight_fn) => {
            let res = weight_fn.bind(py).call1((weight,))?;
            res.extract()
        }
        None => Ok(default),
    }
}

pub fn edge_weights_from_callable<'p, T, Ty: EdgeType>(
    py: Python<'p>,
    graph: &StablePyGraph<Ty>,
    weight_fn: &'p Option<PyObject>,
    default_weight: T,
) -> PyResult<Vec<Option<T>>>
where
    T: FromPyObject<'p> + Copy,
{
    let mut edge_weights: Vec<Option<T>> = Vec::with_capacity(graph.edge_bound());
    for index in 0..=graph.edge_bound() {
        let raw_weight = graph.edge_weight(EdgeIndex::new(index));
        match raw_weight {
            Some(weight) => edge_weights.push(Some(weight_callable(
                py,
                weight_fn,
                weight,
                default_weight,
            )?)),
            None => edge_weights.push(None),
        };
    }

    Ok(edge_weights)
}

#[inline]
fn is_valid_weight(val: f64) -> PyResult<f64> {
    if val.is_sign_negative() {
        return Err(PyValueError::new_err("Negative weights not supported."));
    }

    if val.is_nan() {
        return Err(PyValueError::new_err("NaN weights not supported."));
    }

    Ok(val)
}

pub enum CostFn {
    Default(f64),
    PyFunction(PyObject),
}

impl From<PyObject> for CostFn {
    fn from(obj: PyObject) -> Self {
        CostFn::PyFunction(obj)
    }
}

impl TryFrom<f64> for CostFn {
    type Error = PyErr;

    fn try_from(val: f64) -> Result<Self, Self::Error> {
        let val = is_valid_weight(val)?;
        Ok(CostFn::Default(val))
    }
}

impl TryFrom<(Option<PyObject>, f64)> for CostFn {
    type Error = PyErr;

    fn try_from(func_or_default: (Option<PyObject>, f64)) -> Result<Self, Self::Error> {
        let (obj, val) = func_or_default;
        match obj {
            Some(obj) => Ok(CostFn::PyFunction(obj)),
            None => CostFn::try_from(val),
        }
    }
}

impl CostFn {
    fn call(&self, py: Python, arg: &PyObject) -> PyResult<f64> {
        match self {
            CostFn::Default(val) => Ok(*val),
            CostFn::PyFunction(obj) => {
                let raw = obj.call1(py, (arg,))?;
                let val: f64 = raw.extract(py)?;
                is_valid_weight(val)
            }
        }
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
            .bind(py)
            .rich_compare(weight, pyo3::basic::CompareOp::Eq)?
            .is_truthy()?
        {
            index = Some(node);
            break;
        }
    }
    Ok(index)
}

fn generic_class_getitem(
    cls: &Bound<'_, pyo3::types::PyType>,
    key: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| -> PyResult<PyObject> {
        let types_mod = py.import_bound("types")?;
        let types_generic_alias = types_mod.getattr("GenericAlias")?;
        let args = (cls, key);
        let generic_alias = types_generic_alias.call1(args)?;
        Ok(generic_alias.into())
    })
}

// The provided node is invalid.
create_exception!(rustworkx, InvalidNode, PyException);
// Performing this operation would result in trying to add a cycle to a DAG.
create_exception!(rustworkx, DAGWouldCycle, PyException);
// There is no edge present between the provided nodes.
create_exception!(rustworkx, NoEdgeBetweenNodes, PyException);
// The specified Directed Graph has a cycle and can't be treated as a DAG.
create_exception!(rustworkx, DAGHasCycle, PyException);
// No neighbors found matching the provided predicate.
create_exception!(rustworkx, NoSuitableNeighbors, PyException);
// Invalid operation on a null graph
create_exception!(rustworkx, NullGraph, PyException);
// No path was found between the specified nodes.
create_exception!(rustworkx, NoPathFound, PyException);
// No mapping was found for the request swapping
create_exception!(rustworkx, InvalidMapping, PyException);
// Prune part of the search tree while traversing a graph.
import_exception!(rustworkx.visit, PruneSearch);
// Stop graph traversal.
import_exception!(rustworkx.visit, StopSearch);
// JSON Error
create_exception!(rustworkx, JSONSerializationError, PyException);
// JSON Error
create_exception!(rustworkx, JSONDeserializationError, PyException);
// Negative Cycle found on shortest-path algorithm
create_exception!(rustworkx, NegativeCycle, PyException);
// Failed to Converge on a solution
create_exception!(rustworkx, FailedToConverge, PyException);
// Graph is not bipartite
create_exception!(rustworkx, GraphNotBipartite, PyException);

#[pymodule]
fn rustworkx(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("InvalidNode", py.get_type_bound::<InvalidNode>())?;
    m.add("DAGWouldCycle", py.get_type_bound::<DAGWouldCycle>())?;
    m.add(
        "NoEdgeBetweenNodes",
        py.get_type_bound::<NoEdgeBetweenNodes>(),
    )?;
    m.add("DAGHasCycle", py.get_type_bound::<DAGHasCycle>())?;
    m.add(
        "NoSuitableNeighbors",
        py.get_type_bound::<NoSuitableNeighbors>(),
    )?;
    m.add("NoPathFound", py.get_type_bound::<NoPathFound>())?;
    m.add("InvalidMapping", py.get_type_bound::<InvalidMapping>())?;
    m.add("NullGraph", py.get_type_bound::<NullGraph>())?;
    m.add("NegativeCycle", py.get_type_bound::<NegativeCycle>())?;
    m.add(
        "JSONSerializationError",
        py.get_type_bound::<JSONSerializationError>(),
    )?;
    m.add("FailedToConverge", py.get_type_bound::<FailedToConverge>())?;
    m.add(
        "GraphNotBipartite",
        py.get_type_bound::<GraphNotBipartite>(),
    )?;
    m.add(
        "JSONDeserializationError",
        py.get_type_bound::<JSONDeserializationError>(),
    )?;
    m.add_wrapped(wrap_pyfunction!(bfs_successors))?;
    m.add_wrapped(wrap_pyfunction!(bfs_predecessors))?;
    m.add_wrapped(wrap_pyfunction!(graph_bfs_search))?;
    m.add_wrapped(wrap_pyfunction!(digraph_bfs_search))?;
    m.add_wrapped(wrap_pyfunction!(graph_dijkstra_search))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dijkstra_search))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path))?;
    m.add_wrapped(wrap_pyfunction!(dag_longest_path_length))?;
    m.add_wrapped(wrap_pyfunction!(dag_weighted_longest_path))?;
    m.add_wrapped(wrap_pyfunction!(dag_weighted_longest_path_length))?;
    m.add_wrapped(wrap_pyfunction!(transitive_reduction))?;
    m.add_wrapped(wrap_pyfunction!(number_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(connected_components))?;
    m.add_wrapped(wrap_pyfunction!(is_connected))?;
    m.add_wrapped(wrap_pyfunction!(node_connected_component))?;
    m.add_wrapped(wrap_pyfunction!(number_weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(weakly_connected_components))?;
    m.add_wrapped(wrap_pyfunction!(is_weakly_connected))?;
    m.add_wrapped(wrap_pyfunction!(is_semi_connected))?;
    m.add_wrapped(wrap_pyfunction!(is_directed_acyclic_graph))?;
    m.add_wrapped(wrap_pyfunction!(digraph_is_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(graph_is_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(digraph_is_subgraph_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(graph_is_subgraph_isomorphic))?;
    m.add_wrapped(wrap_pyfunction!(digraph_vf2_mapping))?;
    m.add_wrapped(wrap_pyfunction!(graph_vf2_mapping))?;
    m.add_wrapped(wrap_pyfunction!(digraph_union))?;
    m.add_wrapped(wrap_pyfunction!(graph_union))?;
    m.add_wrapped(wrap_pyfunction!(immediate_dominators))?;
    m.add_wrapped(wrap_pyfunction!(dominance_frontiers))?;
    m.add_wrapped(wrap_pyfunction!(digraph_maximum_bisimulation))?;
    m.add_wrapped(wrap_pyfunction!(digraph_cartesian_product))?;
    m.add_wrapped(wrap_pyfunction!(graph_cartesian_product))?;
    m.add_wrapped(wrap_pyfunction!(topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(topological_generations))?;
    m.add_wrapped(wrap_pyfunction!(descendants))?;
    m.add_wrapped(wrap_pyfunction!(ancestors))?;
    m.add_wrapped(wrap_pyfunction!(lexicographical_topological_sort))?;
    m.add_wrapped(wrap_pyfunction!(graph_floyd_warshall))?;
    m.add_wrapped(wrap_pyfunction!(digraph_floyd_warshall))?;
    m.add_wrapped(wrap_pyfunction!(graph_floyd_warshall_numpy))?;
    m.add_wrapped(wrap_pyfunction!(digraph_floyd_warshall_numpy))?;
    m.add_wrapped(wrap_pyfunction!(
        graph_floyd_warshall_successor_and_distance
    ))?;
    m.add_wrapped(wrap_pyfunction!(
        digraph_floyd_warshall_successor_and_distance
    ))?;
    m.add_wrapped(wrap_pyfunction!(collect_runs))?;
    m.add_wrapped(wrap_pyfunction!(collect_bicolor_runs))?;
    m.add_wrapped(wrap_pyfunction!(layers))?;
    m.add_wrapped(wrap_pyfunction!(graph_distance_matrix))?;
    m.add_wrapped(wrap_pyfunction!(digraph_distance_matrix))?;
    m.add_wrapped(wrap_pyfunction!(digraph_adjacency_matrix))?;
    m.add_wrapped(wrap_pyfunction!(graph_adjacency_matrix))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_pairs_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_pairs_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_longest_simple_path))?;
    m.add_wrapped(wrap_pyfunction!(digraph_longest_simple_path))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_simple_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_has_path))?;
    m.add_wrapped(wrap_pyfunction!(digraph_has_path))?;
    m.add_wrapped(wrap_pyfunction!(graph_dijkstra_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dijkstra_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(graph_bellman_ford_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_bellman_ford_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_bellman_ford_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_bellman_ford_shortest_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(negative_edge_cycle))?;
    m.add_wrapped(wrap_pyfunction!(find_negative_cycle))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_pairs_dijkstra_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(digraph_all_pairs_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_pairs_dijkstra_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_pairs_dijkstra_shortest_paths))?;
    m.add_wrapped(wrap_pyfunction!(
        digraph_all_pairs_bellman_ford_path_lengths
    ))?;
    m.add_wrapped(wrap_pyfunction!(
        digraph_all_pairs_bellman_ford_shortest_paths
    ))?;
    m.add_wrapped(wrap_pyfunction!(graph_all_pairs_bellman_ford_path_lengths))?;
    m.add_wrapped(wrap_pyfunction!(
        graph_all_pairs_bellman_ford_shortest_paths
    ))?;
    m.add_wrapped(wrap_pyfunction!(graph_betweenness_centrality))?;
    m.add_wrapped(wrap_pyfunction!(digraph_betweenness_centrality))?;
    m.add_wrapped(wrap_pyfunction!(graph_closeness_centrality))?;
    m.add_wrapped(wrap_pyfunction!(digraph_closeness_centrality))?;
    m.add_wrapped(wrap_pyfunction!(graph_edge_betweenness_centrality))?;
    m.add_wrapped(wrap_pyfunction!(digraph_edge_betweenness_centrality))?;
    m.add_wrapped(wrap_pyfunction!(graph_eigenvector_centrality))?;
    m.add_wrapped(wrap_pyfunction!(digraph_eigenvector_centrality))?;
    m.add_wrapped(wrap_pyfunction!(graph_katz_centrality))?;
    m.add_wrapped(wrap_pyfunction!(digraph_katz_centrality))?;
    m.add_wrapped(wrap_pyfunction!(graph_degree_centrality))?;
    m.add_wrapped(wrap_pyfunction!(digraph_degree_centrality))?;
    m.add_wrapped(wrap_pyfunction!(in_degree_centrality))?;
    m.add_wrapped(wrap_pyfunction!(out_degree_centrality))?;
    m.add_wrapped(wrap_pyfunction!(graph_astar_shortest_path))?;
    m.add_wrapped(wrap_pyfunction!(digraph_astar_shortest_path))?;
    m.add_wrapped(wrap_pyfunction!(graph_greedy_color))?;
    m.add_wrapped(wrap_pyfunction!(graph_misra_gries_edge_color))?;
    m.add_wrapped(wrap_pyfunction!(graph_greedy_edge_color))?;
    m.add_wrapped(wrap_pyfunction!(graph_bipartite_edge_color))?;
    m.add_wrapped(wrap_pyfunction!(graph_two_color))?;
    m.add_wrapped(wrap_pyfunction!(digraph_two_color))?;
    m.add_wrapped(wrap_pyfunction!(graph_is_bipartite))?;
    m.add_wrapped(wrap_pyfunction!(digraph_is_bipartite))?;
    m.add_wrapped(wrap_pyfunction!(graph_line_graph))?;
    m.add_wrapped(wrap_pyfunction!(graph_tensor_product))?;
    m.add_wrapped(wrap_pyfunction!(digraph_tensor_product))?;
    m.add_wrapped(wrap_pyfunction!(directed_gnp_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(undirected_gnp_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_gnm_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(undirected_gnm_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(undirected_sbm_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_sbm_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(random_geometric_graph))?;
    m.add_wrapped(wrap_pyfunction!(hyperbolic_random_graph))?;
    m.add_wrapped(wrap_pyfunction!(barabasi_albert_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_barabasi_albert_graph))?;
    m.add_wrapped(wrap_pyfunction!(directed_random_bipartite_graph))?;
    m.add_wrapped(wrap_pyfunction!(undirected_random_bipartite_graph))?;
    m.add_wrapped(wrap_pyfunction!(cycle_basis))?;
    m.add_wrapped(wrap_pyfunction!(simple_cycles))?;
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
    m.add_wrapped(wrap_pyfunction!(graph_token_swapper))?;
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
    m.add_wrapped(wrap_pyfunction!(stoer_wagner_min_cut))?;
    m.add_wrapped(wrap_pyfunction!(steiner_tree::steiner_tree))?;
    m.add_wrapped(wrap_pyfunction!(digraph_dfs_search))?;
    m.add_wrapped(wrap_pyfunction!(graph_dfs_search))?;
    m.add_wrapped(wrap_pyfunction!(articulation_points))?;
    m.add_wrapped(wrap_pyfunction!(bridges))?;
    m.add_wrapped(wrap_pyfunction!(biconnected_components))?;
    m.add_wrapped(wrap_pyfunction!(chain_decomposition))?;
    m.add_wrapped(wrap_pyfunction!(graph_isolates))?;
    m.add_wrapped(wrap_pyfunction!(digraph_isolates))?;
    m.add_wrapped(wrap_pyfunction!(connected_subgraphs))?;
    m.add_wrapped(wrap_pyfunction!(is_planar))?;
    m.add_wrapped(wrap_pyfunction!(read_graphml))?;
    m.add_wrapped(wrap_pyfunction!(digraph_node_link_json))?;
    m.add_wrapped(wrap_pyfunction!(graph_node_link_json))?;
    m.add_wrapped(wrap_pyfunction!(from_node_link_json_file))?;
    m.add_wrapped(wrap_pyfunction!(parse_node_link_json))?;
    m.add_wrapped(wrap_pyfunction!(pagerank))?;
    m.add_wrapped(wrap_pyfunction!(hits))?;
    m.add_class::<digraph::PyDiGraph>()?;
    m.add_class::<graph::PyGraph>()?;
    m.add_class::<toposort::TopologicalSorter>()?;
    m.add_class::<iterators::RelationalCoarsestPartition>()?;
    m.add_class::<iterators::IndexPartitionBlock>()?;
    m.add_class::<iterators::BFSSuccessors>()?;
    m.add_class::<iterators::BFSPredecessors>()?;
    m.add_class::<iterators::Chains>()?;
    m.add_class::<iterators::NodeIndices>()?;
    m.add_class::<iterators::EdgeIndices>()?;
    m.add_class::<iterators::EdgeList>()?;
    m.add_class::<iterators::EdgeIndexMap>()?;
    m.add_class::<iterators::WeightedEdgeList>()?;
    m.add_class::<iterators::PathMapping>()?;
    m.add_class::<iterators::PathLengthMapping>()?;
    m.add_class::<iterators::CentralityMapping>()?;
    m.add_class::<iterators::EdgeCentralityMapping>()?;
    m.add_class::<iterators::Pos2DMapping>()?;
    m.add_class::<iterators::MultiplePathMapping>()?;
    m.add_class::<iterators::AllPairsMultiplePathMapping>()?;
    m.add_class::<iterators::AllPairsPathLengthMapping>()?;
    m.add_class::<iterators::AllPairsPathMapping>()?;
    m.add_class::<iterators::NodesCountMapping>()?;
    m.add_class::<iterators::NodeMap>()?;
    m.add_class::<iterators::ProductNodeMap>()?;
    m.add_class::<iterators::BiconnectedComponents>()?;
    m.add_class::<ColoringStrategy>()?;
    m.add_wrapped(wrap_pymodule!(generators::generators))?;
    Ok(())
}
