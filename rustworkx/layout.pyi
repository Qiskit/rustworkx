# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/layout/mod.rs

from .iterators import *
from .graph import PyGraph
from .digraph import PyDiGraph

from typing import Optional, Set, List, Dict, TypeVar, Tuple, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

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
