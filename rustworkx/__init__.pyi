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
    source: int,
    visitor: _BFSVisitor,
) -> None: ...
def dfs_search(
    graph: PyGraph | PyDiGraph,
    source: int,
    visitor: _DFSVisitor,
) -> None: ...
def dijkstra_search(
    graph: PyGraph | PyDiGraph,
    source: int,
    weight_fn: Callable[[Any], float],
    visitor: _DijkstraVisitor,
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
