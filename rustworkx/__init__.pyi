# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/lib.rs

import numpy as np
import rustworkx.visit as visit

from .rustworkx import *
from typing import Generic, TypeVar, Any, Callable, Iterator, overload

from .graph import PyGraph as PyGraph
from .digraph import PyDiGraph as PyDiGraph

from .cartesian_product import digraph_cartesian_product as digraph_cartesian_product
from .cartesian_product import graph_cartesian_product as graph_cartesian_product

from .centrality import digraph_eigenvector_centrality as digraph_eigenvector_centrality
from .centrality import graph_eigenvector_centrality as graph_eigenvector_centrality
from .centrality import digraph_betweenness_centrality as digraph_betweenness_centrality
from .centrality import graph_betweenness_centrality as graph_betweenness_centrality
from .centrality import digraph_edge_betweenness_centrality as digraph_edge_betweenness_centrality
from .centrality import graph_edge_betweenness_centrality as graph_edge_betweenness_centrality
from .centrality import digraph_closeness_centrality as digraph_closeness_centrality
from .centrality import graph_closeness_centrality as graph_closeness_centrality
from .centrality import digraph_katz_centrality as digraph_katz_centrality
from .centrality import graph_katz_centrality as graph_katz_centrality

from .coloring import graph_greedy_color as graph_greedy_color
from .coloring import graph_greedy_edge_color as graph_greedy_edge_color
from .coloring import graph_is_bipartite as graph_is_bipartite
from .coloring import digraph_is_bipartite as digraph_is_bipartite
from .coloring import graph_two_color as graph_two_color
from .coloring import digraph_two_color as digraph_two_color

from .connectivity import connected_components as connected_components
from .connectivity import is_connected as is_connected
from .connectivity import is_weakly_connected as is_weakly_connected
from .connectivity import number_connected_components as number_connected_components
from .connectivity import number_weakly_connected_components as number_weakly_connected_components
from .connectivity import node_connected_component as node_connected_component
from .connectivity import strongly_connected_components as strongly_connected_components
from .connectivity import weakly_connected_components as weakly_connected_components
from .connectivity import digraph_adjacency_matrix as digraph_adjacency_matrix
from .connectivity import graph_adjacency_matrix as graph_adjacency_matrix
from .connectivity import cycle_basis as cycle_basis
from .connectivity import articulation_points as articulation_points
from .connectivity import biconnected_components as biconnected_components
from .connectivity import chain_decomposition as chain_decomposition
from .connectivity import digraph_find_cycle as digraph_find_cycle
from .connectivity import digraph_complement as digraph_complement
from .connectivity import graph_complement as graph_complement
from .connectivity import digraph_all_simple_paths as digraph_all_simple_paths
from .connectivity import graph_all_simple_paths as graph_all_simple_paths
from .connectivity import digraph_all_pairs_all_simple_paths as digraph_all_pairs_all_simple_paths
from .connectivity import graph_all_pairs_all_simple_paths as graph_all_pairs_all_simple_paths
from .connectivity import digraph_longest_simple_path as digraph_longest_simple_path
from .connectivity import graph_longest_simple_path as graph_longest_simple_path
from .connectivity import digraph_core_number as digraph_core_number
from .connectivity import graph_core_number as graph_core_number
from .connectivity import stoer_wagner_min_cut as stoer_wagner_min_cut
from .connectivity import simple_cycles as simple_cycles
from .connectivity import graph_isolates as graph_isolates
from .connectivity import digraph_isolates as digraph_isolates

from .dag_algo import collect_runs as collect_runs
from .dag_algo import collect_bicolor_runs as collect_bicolor_runs
from .dag_algo import dag_longest_path as dag_longest_path
from .dag_algo import dag_longest_path_length as dag_longest_path_length
from .dag_algo import dag_weighted_longest_path as dag_weighted_longest_path
from .dag_algo import dag_weighted_longest_path_length as dag_weighted_longest_path_length
from .dag_algo import is_directed_acyclic_graph as is_directed_acyclic_graph
from .dag_algo import topological_sort as topological_sort
from .dag_algo import lexicographical_topological_sort as lexicographical_topological_sort
from .dag_algo import transitive_reduction as transitive_reduction
from .dag_algo import layers as layers

from .isomorphism import digraph_is_isomorphic as digraph_is_isomorphic
from .isomorphism import graph_is_isomorphic as graph_is_isomorphic
from .isomorphism import digraph_is_subgraph_isomorphic as digraph_is_subgraph_isomorphic
from .isomorphism import graph_is_subgraph_isomorphic as graph_is_subgraph_isomorphic
from .isomorphism import digraph_vf2_mapping as digraph_vf2_mapping
from .isomorphism import graph_vf2_mapping as graph_vf2_mapping

from .layout import digraph_bipartite_layout as digraph_bipartite_layout
from .layout import graph_bipartite_layout as graph_bipartite_layout
from .layout import digraph_circular_layout as digraph_circular_layout
from .layout import graph_circular_layout as graph_circular_layout
from .layout import digraph_random_layout as digraph_random_layout
from .layout import graph_random_layout as graph_random_layout
from .layout import graph_shell_layout as graph_shell_layout
from .layout import digraph_spiral_layout as digraph_spiral_layout
from .layout import graph_spiral_layout as graph_spiral_layout
from .layout import digraph_spring_layout as digraph_spring_layout
from .layout import graph_spring_layout as graph_spring_layout

from .line_graph import graph_line_graph as graph_line_graph

from .link_analysis import hits as hits
from .link_analysis import pagerank as pagerank

from .matching import max_weight_matching as max_weight_matching
from .matching import is_matching as is_matching
from .matching import is_maximal_matching as is_maximal_matching

from .planar import is_planar as is_planar

from .random_graph import directed_gnm_random_graph as directed_gnm_random_graph
from .random_graph import undirected_gnm_random_graph as undirected_gnm_random_graph
from .random_graph import directed_gnp_random_graph as directed_gnp_random_graph
from .random_graph import undirected_gnp_random_graph as undirected_gnp_random_graph
from .random_graph import random_geometric_graph as random_geometric_graph

from .read_write import read_graphml as read_graphml
from .read_write import digraph_node_link_json as digraph_node_link_json
from .read_write import graph_node_link_json as graph_node_link_json

from .shortest_path import (
    digraph_bellman_ford_shortest_paths as digraph_bellman_ford_shortest_paths,
)
from .shortest_path import graph_bellman_ford_shortest_paths as graph_bellman_ford_shortest_paths
from .shortest_path import (
    digraph_bellman_ford_shortest_path_lengths as digraph_bellman_ford_shortest_path_lengths,
)
from .shortest_path import (
    graph_bellman_ford_shortest_path_lengths as graph_bellman_ford_shortest_path_lengths,
)
from .shortest_path import digraph_dijkstra_shortest_paths as digraph_dijkstra_shortest_paths
from .shortest_path import graph_dijkstra_shortest_paths as graph_dijkstra_shortest_paths
from .shortest_path import (
    digraph_dijkstra_shortest_path_lengths as digraph_dijkstra_shortest_path_lengths,
)
from .shortest_path import (
    graph_dijkstra_shortest_path_lengths as graph_dijkstra_shortest_path_lengths,
)
from .shortest_path import (
    digraph_all_pairs_bellman_ford_path_lengths as digraph_all_pairs_bellman_ford_path_lengths,
)
from .shortest_path import (
    graph_all_pairs_bellman_ford_path_lengths as graph_all_pairs_bellman_ford_path_lengths,
)
from .shortest_path import (
    digraph_all_pairs_bellman_ford_shortest_paths as digraph_all_pairs_bellman_ford_shortest_paths,
)
from .shortest_path import (
    graph_all_pairs_bellman_ford_shortest_paths as graph_all_pairs_bellman_ford_shortest_paths,
)
from .shortest_path import (
    digraph_all_pairs_dijkstra_path_lengths as digraph_all_pairs_dijkstra_path_lengths,
)
from .shortest_path import (
    graph_all_pairs_dijkstra_path_lengths as graph_all_pairs_dijkstra_path_lengths,
)
from .shortest_path import (
    digraph_all_pairs_dijkstra_shortest_paths as digraph_all_pairs_dijkstra_shortest_paths,
)
from .shortest_path import (
    graph_all_pairs_dijkstra_shortest_paths as graph_all_pairs_dijkstra_shortest_paths,
)
from .shortest_path import digraph_astar_shortest_path as digraph_astar_shortest_path
from .shortest_path import graph_astar_shortest_path as graph_astar_shortest_path
from .shortest_path import digraph_k_shortest_path_lengths as digraph_k_shortest_path_lengths
from .shortest_path import graph_k_shortest_path_lengths as graph_k_shortest_path_lengths
from .shortest_path import digraph_has_path as digraph_has_path
from .shortest_path import graph_has_path as graph_has_path
from .shortest_path import (
    digraph_num_shortest_paths_unweighted as digraph_num_shortest_paths_unweighted,
)
from .shortest_path import (
    graph_num_shortest_paths_unweighted as graph_num_shortest_paths_unweighted,
)
from .shortest_path import (
    digraph_unweighted_average_shortest_path_length as digraph_unweighted_average_shortest_path_length,
)
from .shortest_path import digraph_distance_matrix as digraph_distance_matrix
from .shortest_path import graph_distance_matrix as graph_distance_matrix
from .shortest_path import digraph_floyd_warshall as digraph_floyd_warshall
from .shortest_path import graph_floyd_warshall as graph_floyd_warshall
from .shortest_path import digraph_floyd_warshall_numpy as digraph_floyd_warshall_numpy
from .shortest_path import graph_floyd_warshall_numpy as graph_floyd_warshall_numpy
from .shortest_path import (
    digraph_floyd_warshall_successor_and_distance as digraph_floyd_warshall_successor_and_distance,
)
from .shortest_path import (
    graph_floyd_warshall_successor_and_distance as graph_floyd_warshall_successor_and_distance,
)
from .shortest_path import find_negative_cycle as find_negative_cycle
from .shortest_path import negative_edge_cycle as negative_edge_cycle

from .traversal import digraph_bfs_search as digraph_bfs_search
from .traversal import graph_bfs_search as graph_bfs_search
from .traversal import digraph_dfs_search as digraph_dfs_search
from .traversal import graph_dfs_search as graph_dfs_search
from .traversal import digraph_dijkstra_search as digraph_dijkstra_search
from .traversal import graph_dijkstra_search as graph_dijkstra_search
from .traversal import digraph_dfs_edges as digraph_dfs_edges
from .traversal import graph_dfs_edges as graph_dfs_edges
from .traversal import ancestors as ancestors
from .traversal import bfs_predecessors as bfs_predecessors
from .traversal import bfs_successors as bfs_successors
from .traversal import descendants as descendants

from .transitivity import digraph_transitivity as digraph_transitivity
from .transitivity import graph_transitivity as graph_transitivity

from .tree import minimum_spanning_edges as minimum_spanning_edges
from .tree import minimum_spanning_tree as minimum_spanning_tree
from .tree import steiner_tree as steiner_tree
from .tree import metric_closure as metric_closure

from .tensor_product import digraph_tensor_product as digraph_tensor_product
from .tensor_product import graph_tensor_product as graph_tensor_product

from .token_swapper import graph_token_swapper as graph_token_swapper

from .union import digraph_union as digraph_union
from .union import graph_union as graph_union

_S = TypeVar("_S")
_T = TypeVar("_T")
_BFSVisitor = TypeVar("_BFSVisitor", bound=visit.BFSVisitor)
_DFSVisitor = TypeVar("_DFSVisitor", bound=visit.DFSVisitor)
_DijkstraVisitor = TypeVar("_DijkstraVisitor", bound=visit.DijkstraVisitor)

class PyDAG(Generic[_S, _T], PyDiGraph[_S, _T]): ...

def distance_matrix(
    graph: PyGraph | PyDiGraph,
    parallel_threshold: int = ...,
    as_undirected: bool = ...,
    null_value: float = ...,
) -> np.ndarray: ...
def unweighted_average_shortest_path_length(
    graph: PyGraph | PyDiGraph,
    parallel_threshold: int = ...,
    disconnected: bool = ...,
) -> float: ...
def adjacency_matrix(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    null_value: float = ...,
) -> np.ndarray: ...
def all_simple_paths(
    graph: PyGraph | PyDiGraph,
    from_: int,
    to: int,
    min_depth: int | None = ...,
    cutoff: int | None = ...,
) -> list[list[int]]: ...
def floyd_warshall(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    parallel_threshold: int = ...,
) -> AllPairsPathLengthMapping: ...
def floyd_warshall_numpy(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    parallel_threshold: int = ...,
) -> np.ndarray: ...
def floyd_warshall_successor_and_distance(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float | None = ...,
    parallel_threshold: int | None = ...,
) -> tuple[np.ndarray, np.ndarray]: ...
def astar_shortest_path(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    node: int,
    goal_fn: Callable[[_S], bool],
    edge_cost_fn: Callable[[_T], float],
    estimate_cost_fn: Callable[[_S], float],
) -> NodeIndices: ...
def dijkstra_shortest_paths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    source: int,
    target: int | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> PathMapping: ...
def has_path(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T], source: int, target: int, as_undirected: bool = ...
) -> bool: ...
def all_pairs_dijkstra_shortest_paths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float] | None,
) -> AllPairsPathMapping: ...
def all_pairs_all_simple_paths(
    graph: PyGraph | PyDiGraph,
    min_depth: int | None = ...,
    cutoff: int | None = ...,
) -> AllPairsMultiplePathMapping: ...
def all_pairs_dijkstra_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float] | None,
) -> AllPairsPathLengthMapping: ...
def dijkstra_shortest_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    node: int,
    edge_cost_fn: Callable[[_T], float] | None,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def k_shortest_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    start: int,
    k: int,
    edge_cost: Callable[[_T], float],
    goal: int | None = ...,
) -> PathLengthMapping: ...
def dfs_edges(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T], source: int | None = ...) -> EdgeList: ...
@overload
def is_isomorphic(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
@overload
def is_isomorphic(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
@overload
def is_isomorphic_node_match(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    matcher: Callable[[_S, _S], bool],
    id_order: bool = ...,
) -> bool: ...
@overload
def is_isomorphic_node_match(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    matcher: Callable[[_S, _S], bool],
    id_order: bool = ...,
) -> bool: ...
@overload
def is_subgraph_isomorphic(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
@overload
def is_subgraph_isomorphic(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
def transitivity(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> float: ...
def core_number(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> int: ...
def complement(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]
) -> PyGraph[_S, _T | None] | PyDiGraph[_S, _T | None]: ...
def random_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
) -> Pos2DMapping: ...
def spring_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    pos: dict[int, tuple[float, float]] | None = ...,
    fixed: set[int] | None = ...,
    k: float | None = ...,
    repulsive_exponent: int = ...,
    adaptive_cooling: bool = ...,
    num_iter: int = ...,
    tol: float = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: int = ...,
    scale: int = ...,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
) -> Pos2DMapping: ...
def networkx_converter(graph: Any, keep_attributes: bool = ...) -> PyGraph | PyDiGraph: ...
def bipartite_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    first_nodes,
    horizontal: bool = ...,
    scale: int = ...,
    center: tuple[float, float] | None = ...,
    aspect_ratio=...,
) -> Pos2DMapping: ...
def circular_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    scale: int = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def shell_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    nlist: list[list[int]] | None = ...,
    rotate: float | None = ...,
    scale: int = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def spiral_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    scale: int = ...,
    center: tuple[float, float] | None = ...,
    resolution: float = ...,
    equidistant: bool = ...,
) -> Pos2DMapping: ...
def num_shortest_paths_unweighted(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T], source: int
) -> NodesCountMapping: ...
def betweenness_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    normalized: bool = ...,
    endpoints: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def closeness_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T], wf_improved: bool = ...
) -> CentralityMapping: ...
def edge_betweenness_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    normalized: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def eigenvector_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
def katz_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    alpha: float = ...,
    beta: float = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
@overload
def vf2_mapping(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    subgraph: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> Iterator[NodeMap]: ...
@overload
def vf2_mapping(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    subgraph: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> Iterator[NodeMap]: ...
@overload
def union(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyGraph[_S, _T]: ...
@overload
def union(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyDiGraph[_S, _T]: ...
@overload
def tensor_product(
    first: PyGraph,
    second: PyGraph,
) -> tuple[PyGraph, ProductNodeMap]: ...
@overload
def tensor_product(
    first: PyDiGraph,
    second: PyDiGraph,
) -> tuple[PyDiGraph, ProductNodeMap]: ...
@overload
def cartesian_product(
    first: PyGraph,
    second: PyGraph,
) -> tuple[PyGraph, ProductNodeMap]: ...
@overload
def cartesian_product(
    first: PyDiGraph,
    second: PyDiGraph,
) -> tuple[PyDiGraph, ProductNodeMap]: ...
def bfs_search(
    graph: PyGraph | PyDiGraph,
    source: int | None = ...,
    visitor: _BFSVisitor | None = ...,
) -> None: ...
def dfs_search(
    graph: PyGraph | PyDiGraph,
    source: int | None = ...,
    visitor: _DFSVisitor | None = ...,
) -> None: ...
def dijkstra_search(
    graph: PyGraph | PyDiGraph,
    source: int | None = ...,
    weight_fn: Callable[[Any], float] | None = ...,
    visitor: _DijkstraVisitor | None = ...,
) -> None: ...
def bellman_ford_shortest_paths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    source,
    target: int | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> PathMapping: ...
def bellman_ford_shortest_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    node: int,
    edge_cost_fn: Callable[[_T], float] | None,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def all_pairs_bellman_ford_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float] | None,
) -> AllPairsPathLengthMapping: ...
def all_pairs_bellman_ford_shortest_paths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float] | None,
) -> AllPairsPathMapping: ...
def node_link_json(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    path: str | None = ...,
    graph_attrs: Callable[[Any], dict[str, str]] | None = ...,
    node_attrs: Callable[[_S], str] | None = ...,
    edge_attrs: Callable[[_T], str] | None = ...,
) -> str | None: ...
def longest_simple_path(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> NodeIndices | None: ...
def isolates(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> NodeIndices: ...
def two_color(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> dict[int, int]: ...
def is_bipartite(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> bool: ...
