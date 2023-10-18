# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/connectivity/mod.rs

import numpy as np

from .iterators import *
from .graph import PyGraph
from .digraph import PyDiGraph

from typing import Optional, TypeVar, Callable, Iterator

_S = TypeVar("_S")
_T = TypeVar("_T")

def connected_components(graph: PyGraph, /) -> list[set[int]]: ...
def is_connected(graph: PyGraph, /) -> bool: ...
def is_weakly_connected(graph: PyDiGraph, /) -> bool: ...
def number_connected_components(graph: PyGraph, /) -> int: ...
def number_weakly_connected_components(graph: PyDiGraph, /) -> bool: ...
def node_connected_component(graph: PyGraph, node: int, /) -> set[int]: ...
def strongly_connected_components(graph: PyDiGraph, /) -> list[list[int]]: ...
def weakly_connected_components(graph: PyDiGraph, /) -> list[set[int]]: ...
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
def cycle_basis(graph: PyGraph, /, root: Optional[int] = ...) -> list[list[int]]: ...
def articulation_points(graph: PyGraph, /) -> set[int]: ...
def biconnected_components(graph: PyGraph, /) -> BiconnectedComponents: ...
def chain_decomposition(graph: PyGraph, /, source: Optional[int] = ...) -> Chains: ...
def digraph_find_cycle(
    graph: PyDiGraph[_S, _T],
    /,
    source: Optional[int] = ...,
) -> Edgelist: ...
def digraph_complement(graph: PyDiGraph[_S, _T], /) -> PyDiGraph[_S, Optional[_T]]: ...
def graph_complement(
    graph: PyGraph[_S, _T],
    /,
) -> PyGraph[_S, Optional[_T]]: ...
def digraph_all_simple_paths(
    graph: PyDiGraph,
    origin: int,
    to: int,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> list[list[int]]: ...
def graph_all_simple_paths(
    graph: PyGraph,
    origin: int,
    to: int,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> list[list[int]]: ...
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
def digraph_longest_simple_path(graph: PyDiGraph, /) -> Optional[NodeIndices]: ...
def graph_longest_simple_path(graph: PyGraph, /) -> Optional[NodeIndices]: ...
def digraph_core_number(
    graph: PyDiGraph,
    /,
) -> int: ...
def graph_core_number(
    graph: PyGraph,
    /,
) -> int: ...
def stoer_wagner_min_cut(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
) -> Optional[tuple[float, NodeIndices]]: ...
def simple_cycles(graph: PyDiGraph, /) -> Iterator[NodeIndices]: ...
