# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/shortest_path/mod.rs

import numpy as np

from .iterators import *
from .graph import PyGraph
from .digraph import PyDiGraph as PyDiGraph
from .visit import BFSVisitor, DFSVisitor, DijkstraVisitor

from typing import Optional, TypeVar, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

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
def digraph_has_path(
    graph: PyDiGraph,
    source: int,
    target: int,
    /,
    as_undirected: Optional[bool] = ...,
) -> bool: ...
def graph_has_path(
    graph: PyGraph,
    source: int,
    target: int,
) -> bool: ...
def digraph_num_shortest_paths_unweighted(
    graph: PyDiGraph,
    source: int,
    /,
) -> NodesCountMapping: ...
def graph_num_shortest_paths_unweighted(
    graph: PyGraph,
    source: int,
    /,
) -> NodesCountMapping: ...
def digraph_unweighted_average_shortest_path_length(
    graph: PyDiGraph,
    /,
    parallel_threshold: Optional[int] = ...,
    as_undirected: Optional[bool] = ...,
    disconnected: Optional[bool] = ...,
) -> float: ...
def graph_unweighted_average_shortest_path_length(
    graph: PyGraph,
    /,
    parallel_threshold: Optional[int] = ...,
    disconnected: Optional[bool] = ...,
) -> float: ...
def digraph_distance_matrix(
    graph: PyDiGraph,
    /,
    parallel_threshold: Optional[int] = ...,
    as_undirected: Optional[bool] = ...,
    null_value: Optional[float] = ...,
) -> np.ndarray: ...
def graph_distance_matrix(
    graph: PyGraph,
    /,
    parallel_threshold: Optional[int] = ...,
    null_value: Optional[float] = ...,
) -> np.ndarray: ...
def digraph_floyd_warshall(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    as_undirected: Optional[bool] = ...,
    default_weight: Optional[float] = ...,
    parallel_threshold: Optional[int] = ...,
) -> AllPairsPathLengthMapping: ...
def graph_floyd_warshall(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: Optional[float] = ...,
    parallel_threshold: Optional[int] = ...,
) -> AllPairsPathLengthMapping: ...
def digraph_floyd_warshall_numpy(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    as_undirected: Optional[bool] = ...,
    default_weight: Optional[float] = ...,
    parallel_threshold: Optional[int] = ...,
) -> np.ndarray: ...
def graph_floyd_warshall_numpy(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: Optional[float] = ...,
    parallel_threshold: Optional[int] = ...,
) -> np.ndarray: ...
