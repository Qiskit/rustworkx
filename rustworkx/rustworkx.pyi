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

from .iterators import *
from .graph import PyGraph as PyGraph
from .digraph import PyDiGraph as PyDiGraph

from typing import Optional, Set, List, Dict, TypeVar, Tuple, Callable, Union

# Centrality functions
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

# Layout functions
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

# Shortest path functions
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

# Traversal functions
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

_S = TypeVar("_S")
_T = TypeVar("_T")

class DAGHasCycle(Exception): ...
class DAGWouldCycle(Exception): ...
class InvalidNode(Exception): ...
class NoEdgeBetweenNodes(Exception): ...
class NoPathFound(Exception): ...
class NoSuitableNeighbors(Exception): ...
class NullGraph(Exception): ...
class NegativeCycle(Exception): ...
class JSONSerializationError(Exception): ...
class FailedToConverge(Exception): ...

def articulation_points(graph: PyGraph, /) -> Set[int]: ...
def biconnected_components(graph: PyGraph, /) -> BiconnectedComponents: ...
def chain_decomposition(graph: PyGraph, /, source: Optional[int] = ...) -> Chains: ...
def connected_components(graph: PyGraph, /) -> List[Set[int]]: ...
def cycle_basis(graph: PyGraph, /, root: Optional[int] = ...) -> List[List[int]]: ...
def collect_runs(
    graph: PyDiGraph[_S, _T],
    filter_fn: Callable[[_S], bool],
) -> List[List[_S]]: ...
def collect_bicolor_runs(
    graph: PyDiGraph[_S, _T],
    filter_fn: Callable[[_S], bool],
    color_fn: Callable[[_T], int],
) -> List[List[_S]]: ...
def dag_longest_path(
    graph: PyDiGraph[_S, _T], /, weight_fn: Optional[Callable[[int, int, _T], int]] = ...
) -> NodeIndices: ...
def dag_longest_path_length(
    graph: PyDiGraph[_S, _T], /, weight_fn: Optional[Callable[[int, int, _T], int]] = ...
) -> int: ...
def dag_weighted_longest_path(
    graph: PyDiGraph[_S, _T],
    weight_fn: Callable[[int, int, _T], float],
    /,
) -> NodeIndices: ...
def dag_weighted_longest_path_length(
    graph: PyDiGraph[_S, _T],
    weight_fn: Callable[[int, int, _T], float],
    /,
) -> float: ...
def digraph_adjacency_matrix(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
    null_value: float = ...,
    parallel_edge: str = ...,
) -> np.ndarray: ...
def graph_adjacency_matrix(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
    null_value: float = ...,
    parallel_edge: str = ...,
) -> np.ndarray: ...
def digraph_core_number(
    graph: PyDiGraph,
    /,
) -> int: ...
def graph_core_number(
    graph: PyGraph,
    /,
) -> int: ...
def digraph_complement(graph: PyDiGraph[_S, _T], /) -> PyDiGraph[_S, Optional[_T]]: ...
def graph_complement(
    graph: PyGraph[_S, _T],
    /,
) -> PyGraph[_S, Optional[_T]]: ...
def minimum_spanning_edges(
    graph: PyGraph[_S, _T],
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
) -> WeightedEdgeList: ...
def minimum_spanning_tree(
    graph: PyGraph[_S, _T],
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
) -> PyGraph[_S, _T]: ...
def digraph_all_simple_paths(
    graph: PyDiGraph,
    origin: int,
    to: int,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> List[List[int]]: ...
def graph_all_simple_paths(
    graph: PyGraph,
    origin: int,
    to: int,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> List[List[int]]: ...
def digraph_all_pairs_all_simple_paths(
    graph: PyDiGraph,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> AllPairsMultiplePathMapping: ...
def graph_all_pairs_all_simple_paths(
    graph: PyGraph,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> AllPairsMultiplePathMapping: ...
def digraph_is_isomorphic(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    /,
    node_matcher: Optional[Callable[[_S, _S], bool]] = ...,
    edge_matcher: Optional[Callable[[_T, _T], bool]] = ...,
    id_order: bool = ...,
    call_limit: Optional[int] = ...,
) -> bool: ...
def graph_is_isomorphic(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    /,
    node_matcher: Optional[Callable[[_S, _S], bool]] = ...,
    edge_matcher: Optional[Callable[[_T, _T], bool]] = ...,
    id_order: bool = ...,
    call_limit: Optional[int] = ...,
) -> bool: ...
def digraph_is_subgraph_isomorphic(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    /,
    node_matcher: Optional[Callable[[_S, _S], bool]] = ...,
    edge_matcher: Optional[Callable[[_T, _T], bool]] = ...,
    id_order: bool = ...,
    induced: bool = ...,
    call_limit: Optional[int] = ...,
) -> bool: ...
def graph_is_subgraph_isomorphic(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    /,
    node_matcher: Optional[Callable[[_S, _S], bool]] = ...,
    edge_matcher: Optional[Callable[[_T, _T], bool]] = ...,
    id_order: bool = ...,
    induced: bool = ...,
    call_limit: Optional[int] = ...,
) -> bool: ...
def find_negative_cycle(
    graph: PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float],
    /,
) -> NodeIndices: ...
def negative_edge_cycle(
    graph: PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float],
    /,
) -> bool: ...
def digraph_find_cycle(
    graph: PyDiGraph[_S, _T],
    /,
    source: Optional[int] = ...,
) -> EdgeList: ...
def digraph_union(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    /,
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyDiGraph[_S, _T]: ...
def graph_union(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    /,
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyGraph[_S, _T]: ...
def is_planar(graph: PyGraph, /) -> bool: ...
def is_connected(graph: PyGraph, /) -> bool: ...
def is_directed_acyclic_graph(graph: PyDiGraph, /) -> bool: ...
def is_weakly_connected(graph: PyDiGraph, /) -> bool: ...
def number_connected_components(graph: PyGraph, /) -> int: ...
def number_weakly_connected_components(graph: PyDiGraph, /) -> bool: ...
def node_connected_component(graph: PyGraph, node: int, /) -> Set[int]: ...
def strongly_connected_components(graph: PyDiGraph, /) -> List[List[int]]: ...
def weakly_connected_components(graph: PyDiGraph, /) -> List[Set[int]]: ...
def steiner_tree(
    graph: PyGraph[_S, _T],
    terminal_nodes: List[int],
    weight_fn: Callable[[_T], float],
    /,
) -> PyGraph[_S, _T]: ...
def topological_sort(graph: PyDiGraph, /) -> NodeIndices: ...
def lexicographical_topological_sort(
    dag: PyDiGraph[_S, _T],
    key: Callable[[_S], str],
    /,
) -> List[_S]: ...
def directed_gnm_random_graph(
    num_nodes: int,
    num_edges: int,
    /,
    seed: Optional[int] = ...,
) -> PyDiGraph: ...
def undirected_gnm_random_graph(
    num_nodes: int,
    num_edges: int,
    /,
    seed: Optional[int] = ...,
) -> PyGraph: ...
def directed_gnp_random_graph(
    num_nodes: int,
    probability: float,
    /,
    seed: Optional[int] = ...,
) -> PyDiGraph: ...
def undirected_gnp_random_graph(
    num_nodes: int,
    probability: float,
    /,
    seed: Optional[int] = ...,
) -> PyGraph: ...
def read_graphml(path: str, /) -> List[Union[PyGraph, PyDiGraph]]: ...
def hits(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    nstart: Optional[Dict[int, float]] = ...,
    tol: Optional[float] = ...,
    max_iter: Optional[int] = ...,
    normalized: Optional[bool] = ...,
) -> Tuple[CentralityMapping, CentralityMapping]: ...
def pagerank(
    graph: PyDiGraph[_S, _T],
    /,
    alpha: Optional[float] = ...,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    nstart: Optional[Dict[int, float]] = ...,
    personalization: Optional[Dict[int, float]] = ...,
    tol: Optional[float] = ...,
    max_iter: Optional[int] = ...,
    dangling: Optional[Dict[int, float]] = ...,
) -> CentralityMapping: ...
def transitive_reduction(graph: PyDiGraph, /) -> Tuple[PyDiGraph, Dict[int, int]]: ...
def digraph_longest_simple_path(graph: PyDiGraph, /) -> Optional[NodeIndices]: ...
def graph_longest_simple_path(graph: PyGraph, /) -> Optional[NodeIndices]: ...
def digraph_transitivity(graph: PyGraph, /) -> float: ...
def graph_transitivity(graph: PyGraph, /) -> float: ...
def graph_token_swapper(
    graph: PyGraph,
    mapping: Dict[int, int],
    /,
    trials: Optional[int] = ...,
    seed: Optional[int] = ...,
    parallel_threshold: Optional[int] = ...,
) -> EdgeList: ...
def graph_greedy_color(graph: PyGraph, /) -> Dict[int, int]: ...
def graph_greedy_edge_color(graph: PyGraph, /) -> Dict[int, int]: ...
def max_weight_matching(
    graph: PyGraph[_S, _T],
    /,
    max_cardinality: bool = ...,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: int = ...,
    verify_optimum: bool = ...,
) -> Set[Tuple[int, int]]: ...
def is_matching(
    graph: PyGraph,
    matching: Set[Tuple[int, int]],
    /,
) -> bool: ...
def is_maximal_matching(
    graph: PyGraph,
    matching: Set[Tuple[int, int]],
    /,
) -> bool: ...

"""
TopologicalSorter is not present in stub
digraph_cartesian_product is not present in stub
digraph_node_link_json is not present in stub
digraph_tensor_product is not present in stub
digraph_vf2_mapping is not present in stub
generators is not present in stub
graph_cartesian_product is not present in stub
graph_line_graph is not present in stub
graph_node_link_json is not present in stub
graph_tensor_product is not present in stub
graph_vf2_mapping is not present in stub
layers is not present in stub
metric_closure is not present in stub
random_geometric_graph is not present in stub
simple_cycles is not present in stub
stoer_wagner_min_cut is not present in stub
"""
