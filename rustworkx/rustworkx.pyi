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

from typing import Optional, Set, List, Dict, TypeVar, Tuple, Callable

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

def ancestors(graph: PyDiGraph, node: int, /) -> Set[int]: ...
def articulation_points(graph: PyGraph, /) -> Set[int]: ...
def biconnected_components(graph: PyGraph, /) -> BiconnectedComponents: ...
def bfs_predecessors(graph: PyDiGraph, node: int, /) -> BFSPredecessors: ...
def bfs_successors(graph: PyDiGraph, node: int, /) -> BFSSuccessors: ...
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
def descendants(graph: PyDiGraph, node: int, /) -> Set[int]: ...
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
def digraph_bellman_ford_shortest_paths(
    graph: PyDiGraph[_S, _T],
    source: int,
    /,
    target: Optional[int] = ...,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> PathMapping: ...
def graph_bellman_ford_shortest_paths(
    graph: PyDiGraph[_S, _T],
    source: int,
    /,
    target: Optional[int] = ...,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
) -> PathMapping: ...
def digraph_bellman_ford_shortest_path_lengths(
    graph: PyDiGraph[_S, _T],
    node: int,
    edge_cost_fn: Optional[Callable[[_T], float]],
    /,
    goal: Optional[int] = ...,
) -> PathLengthMapping: ...
def graph_bellman_ford_shortest_path_lengths(
    graph: PyGraph[_S, _T],
    node: int,
    edge_cost_fn: Optional[Callable[[_T], float]],
    /,
    goal: Optional[int] = ...,
) -> PathLengthMapping: ...
def digraph_dijkstra_shortest_paths(
    graph: PyDiGraph[_S, _T],
    source: int,
    /,
    target: Optional[int],
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> PathMapping: ...
def graph_dijkstra_shortest_paths(
    graph: PyDiGraph[_S, _T],
    source: int,
    /,
    target: Optional[int],
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
) -> PathMapping: ...
def digraph_dijkstra_shortest_path_lengths(
    graph: PyDiGraph[_S, _T],
    node: int,
    edge_cost_fn: Optional[Callable[[_T], float]],
    /,
    goal: Optional[int] = ...,
) -> PathLengthMapping: ...
def graph_dijkstra_shortest_path_lengths(
    graph: PyGraph[_S, _T],
    node: int,
    edge_cost_fn: Optional[Callable[[_T], float]],
    /,
    goal: Optional[int] = ...,
) -> PathLengthMapping: ...
def digraph_all_pairs_bellman_ford_path_lengths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathLengthMapping: ...
def graph_all_pairs_bellman_ford_path_lengths(
    graph: PyGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathLengthMapping: ...
def digraph_all_pairs_bellman_ford_shortest_paths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathMapping: ...
def graph_all_pairs_bellman_ford_shortest_paths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathMapping: ...
def digraph_all_pairs_dijkstra_path_lengths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathLengthMapping: ...
def graph_all_pairs_dijkstra_path_lengths(
    graph: PyGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathLengthMapping: ...
def digraph_all_pairs_dijkstra_shortest_paths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathMapping: ...
def graph_all_pairs_dijkstra_shortest_paths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathMapping: ...
def digraph_eigenvector_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
def graph_eigenvector_centrality(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
def digraph_betweenness_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    normalized: bool = ...,
    endpoints: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def graph_betweenness_centrality(
    graph: PyGraph[_S, _T],
    /,
    normalized: bool = ...,
    endpoints: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def digraph_edge_betweenness_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    normalized: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def graph_edge_betweenness_centrality(
    graph: PyGraph[_S, _T],
    /,
    normalized: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def digraph_closeness_centrality(
    graph: PyDiGraph[_S, _T],
    wf_improved: bool = ...,
) -> CentralityMapping: ...
def graph_closeness_centrality(
    graph: PyGraph[_S, _T],
    wf_improved: bool = ...,
) -> CentralityMapping: ...
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
def digraph_bipartite_layout(
    graph: PyDiGraph,
    first_nodes: Set[int],
    /,
    horizontal: Optional[bool] = ...,
    scale: Optional[float] = ...,
    center: Optional[Tuple[float, float]] = ...,
    aspect_ratio: Optional[float] = ...,
) -> Pos2DMapping: ...
def graph_bipartite_layout(
    graph: PyGraph,
    first_nodes: Set[int],
    /,
    horizontal: Optional[bool] = ...,
    scale: Optional[float] = ...,
    center: Optional[Tuple[float, float]] = ...,
    aspect_ratio: Optional[float] = ...,
) -> Pos2DMapping: ...
def digraph_circular_layout(
    graph: PyDiGraph,
    /,
    scale: Optional[float] = ...,
    center: Optional[Tuple[float, float]] = ...,
) -> Pos2DMapping: ...
def graph_circular_layout(
    graph: PyGraph,
    /,
    scale: Optional[float] = ...,
    center: Optional[Tuple[float, float]] = ...,
) -> Pos2DMapping: ...
def digraph_random_layout(
    graph: PyDiGraph,
    /,
    center: Optional[Tuple[float, float]] = ...,
    seed: Optional[int] = ...,
) -> Pos2DMapping: ...
def graph_random_layout(
    graph: PyGraph,
    /,
    center: Optional[Tuple[float, float]] = ...,
    seed: Optional[int] = ...,
) -> Pos2DMapping: ...
def digraph_shell_layout(
    graph: PyDiGraph,
    /,
    nlist: Optional[List[List[int]]] = ...,
    rotate: Optional[float] = ...,
    scale: Optional[float] = ...,
    center: Optional[Tuple[float, float]] = ...,
) -> Pos2DMapping: ...
def graph_shell_layout(
    graph: PyGraph,
    /,
    nlist: Optional[List[List[int]]] = ...,
    rotate: Optional[float] = ...,
    scale: Optional[float] = ...,
    center: Optional[Tuple[float, float]] = ...,
) -> Pos2DMapping: ...
def digraph_spiral_layout(
    graph: PyDiGraph,
    /,
    scale: Optional[float] = ...,
    center: Optional[Tuple[float, float]] = ...,
    resolution: Optional[float] = ...,
    equidistant: Optional[bool] = ...,
) -> Pos2DMapping: ...
def graph_spiral_layout(
    graph: PyGraph,
    /,
    scale: Optional[float] = ...,
    center: Optional[Tuple[float, float]] = ...,
    resolution: Optional[float] = ...,
    equidistant: Optional[bool] = ...,
) -> Pos2DMapping: ...
def digraph_spring_layout(
    graph: PyDiGraph[_S, _T],
    pos: Optional[Dict[int, Tuple[float, float]]] = ...,
    fixed: Optional[Set[int]] = ...,
    k: Optional[float] = ...,
    repulsive_exponent: int = ...,
    adaptive_cooling: bool = ...,
    num_iter: int = ...,
    tol: Optional[float] = ...,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
    scale: float = ...,
    center: Optional[Tuple[float, float]] = ...,
    seed: Optional[int] = ...,
    /,
) -> Pos2DMapping: ...
def graph_spring_layout(
    graph: PyGraph[_S, _T],
    pos: Optional[Dict[int, Tuple[float, float]]] = ...,
    fixed: Optional[Set[int]] = ...,
    k: Optional[float] = ...,
    repulsive_exponent: int = ...,
    adaptive_cooling: bool = ...,
    num_iter: int = ...,
    tol: Optional[float] = ...,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: float = ...,
    scale: float = ...,
    center: Optional[Tuple[float, float]] = ...,
    seed: Optional[int] = ...,
    /,
) -> Pos2DMapping: ...
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
)-> AllPairsMultiplePathMapping: ...
def graph_all_pairs_all_simple_paths(
    graph: PyGraph, 
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
)-> AllPairsMultiplePathMapping: ...
def digraph_astar_shortest_path(
    graph: PyDiGraph[_S, _T],
    node: int,
    goal_fn: Callable[[_S], bool],
    edge_cost_fn: Callable[[_T], float],
    estimate_cost_fn: Callable[[_S], float],
    /,
) -> NodeIndices: ...
def graph_astar_shortest_path(
    graph: PyGraph[_S, _T],
    node: int,
    goal_fn: Callable[[_S], bool],
    edge_cost_fn: Callable[[_T], float],
    estimate_cost_fn: Callable[[_S], float],
    /,
) -> NodeIndices: ...
def digraph_k_shortest_path_lengths(
    graph: PyDiGraph[_S, _T],
    start: int, 
    k: int, 
    edge_cost: Callable[[_T], float],
    /, 
    goal: Optional[int] = ...,
) -> PathLengthMapping: ...
def graph_k_shortest_path_lengths(
    graph: PyGraph[_S, _T],
    start: int, 
    k: int, 
    edge_cost: Callable[[_T], float],
    /, 
    goal: Optional[int] = ...,
) -> PathLengthMapping: ...
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

"""
def digraph_bfs_search(*args, **kwargs) -> Any: ...
def digraph_cartesian_product(graph_1, graph_2) -> Any: ...
def digraph_dfs_edges(*args, **kwargs) -> Any: ...
def digraph_dfs_search(*args, **kwargs) -> Any: ...
def digraph_dijkstra_search(*args, **kwargs) -> Any: ...
def digraph_distance_matrix(*args, **kwargs) -> Any: ...
def digraph_find_cycle(*args, **kwargs) -> Any: ...
@overload
def digraph_floyd_warshall(graph, weight_fn = ...) -> Any: ...
@overload
def digraph_floyd_warshall(graph, weight_fn = ...) -> Any: ...
def digraph_floyd_warshall_numpy(*args, **kwargs) -> Any: ...
def digraph_node_link_json(*args, **kwargs) -> Any: ...
def digraph_num_shortest_paths_unweighted(*args, **kwargs) -> Any: ...
def digraph_tensor_product(graph_1, graph_2) -> Any: ...
def digraph_transitivity(*args, **kwargs) -> Any: ...
def digraph_union(*args, **kwargs) -> Any: ...
def digraph_unweighted_average_shortest_path_length(*args, **kwargs) -> Any: ...
def digraph_vf2_mapping(graph_a, graph_b, subgraph = ...) -> Any: ...
def directed_gnm_random_graph(*args, **kwargs) -> Any: ...
def directed_gnp_random_graph(*args, **kwargs) -> Any: ...
def find_negative_cycle(*args, **kwargs) -> Any: ...
def graph_bfs_search(*args, **kwargs) -> Any: ...
def graph_cartesian_product(graph_1, graph_2) -> Any: ...
def graph_dfs_edges(*args, **kwargs) -> Any: ...
def graph_dfs_search(*args, **kwargs) -> Any: ...
def graph_dijkstra_search(*args, **kwargs) -> Any: ...
def graph_distance_matrix(*args, **kwargs) -> Any: ...
@overload
def graph_floyd_warshall(graph, weight_fn = ...) -> Any: ...
@overload
def graph_floyd_warshall(graph, weight_fn = ...) -> Any: ...
@overload
def graph_floyd_warshall_numpy(graph, weight_fn) -> Any: ...
@overload
def graph_floyd_warshall_numpy(graph, weight_fn) -> Any: ...
def graph_greedy_color(graph) -> Any: ...
def graph_node_link_json(*args, **kwargs) -> Any: ...
def graph_num_shortest_paths_unweighted(*args, **kwargs) -> Any: ...
def graph_tensor_product(graph_1, graph_2) -> Any: ...
def graph_token_swapper(*args, **kwargs) -> Any: ...
def graph_transitivity(*args, **kwargs) -> Any: ...
def graph_union(*args, **kwargs) -> Any: ...
def graph_unweighted_average_shortest_path_length(*args, **kwargs) -> Any: ...
def graph_vf2_mapping(graph_a, graph_b, subgraph = ...) -> Any: ...
def is_connected(*args, **kwargs) -> Any: ...
def is_directed_acyclic_graph(*args, **kwargs) -> Any: ...
def is_matching(*args, **kwargs) -> Any: ...
def is_maximal_matching(*args, **kwargs) -> Any: ...
def is_planar(*args, **kwargs) -> Any: ...
def is_weakly_connected(*args, **kwargs) -> Any: ...
def layers(*args, **kwargs) -> Any: ...
def lexicographical_topological_sort(*args, **kwargs) -> Any: ...
def max_weight_matching(*args, **kwargs) -> Any: ...
def metric_closure(*args, **kwargs) -> Any: ...
def negative_edge_cycle(*args, **kwargs) -> Any: ...
def node_connected_component(*args, **kwargs) -> Any: ...
def number_connected_components(*args, **kwargs) -> Any: ...
def number_weakly_connected_components(*args, **kwargs) -> Any: ...
def pagerank() -> Any: ...
def random_geometric_graph(*args, **kwargs) -> Any: ...
def read_graphml(*args, **kwargs) -> Any: ...
def simple_cycles(*args, **kwargs) -> Any: ...
def steiner_tree(*args, **kwargs) -> Any: ...
def stoer_wagner_min_cut(*args, **kwargs) -> Any: ...
def strongly_connected_components(*args, **kwargs) -> Any: ...
def topological_sort(*args, **kwargs) -> Any: ...
def undirected_gnm_random_graph(*args, **kwargs) -> Any: ...
def undirected_gnp_random_graph(*args, **kwargs) -> Any: ...
def weakly_connected_components(*args, **kwargs) -> Any: ...
"""
