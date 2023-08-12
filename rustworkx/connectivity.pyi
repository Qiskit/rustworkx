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
from .graph import PyGraph as PyGraph
from .digraph import PyDiGraph as PyDiGraph

from typing import Optional, Set, List, TypeVar, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

def connected_components(graph: PyGraph, /) -> List[Set[int]]: ...
def is_connected(graph: PyGraph, /) -> bool: ...
def is_weakly_connected(graph: PyDiGraph, /) -> bool: ...
def number_connected_components(graph: PyGraph, /) -> int: ...
def number_weakly_connected_components(graph: PyDiGraph, /) -> bool: ...
def node_connected_component(graph: PyGraph, node: int, /) -> Set[int]: ...
def strongly_connected_components(graph: PyDiGraph, /) -> List[List[int]]: ...
def weakly_connected_components(graph: PyDiGraph, /) -> List[Set[int]]: ...
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
def cycle_basis(graph: PyGraph, /, root: Optional[int] = ...) -> List[List[int]]: ...
def articulation_points(graph: PyGraph, /) -> Set[int]: ...
def biconnected_components(graph: PyGraph, /) -> BiconnectedComponents: ...
def chain_decomposition(graph: PyGraph, /, source: Optional[int] = ...) -> Chains: ...
