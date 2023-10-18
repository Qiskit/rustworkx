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

from typing import TypeVar, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

def digraph_bipartite_layout(
    graph: PyDiGraph,
    first_nodes: set[int],
    /,
    horizontal: bool | None = ...,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
    aspect_ratio: float | None = ...,
) -> Pos2DMapping: ...
def graph_bipartite_layout(
    graph: PyGraph,
    first_nodes: set[int],
    /,
    horizontal: bool | None = ...,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
    aspect_ratio: float | None = ...,
) -> Pos2DMapping: ...
def digraph_circular_layout(
    graph: PyDiGraph,
    /,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def graph_circular_layout(
    graph: PyGraph,
    /,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def digraph_random_layout(
    graph: PyDiGraph,
    /,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
) -> Pos2DMapping: ...
def graph_random_layout(
    graph: PyGraph,
    /,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
) -> Pos2DMapping: ...
def digraph_shell_layout(
    graph: PyDiGraph,
    /,
    nlist: list[list[int]] | None = ...,
    rotate: float | None = ...,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def graph_shell_layout(
    graph: PyGraph,
    /,
    nlist: list[list[int]] | None = ...,
    rotate: float | None = ...,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def digraph_spiral_layout(
    graph: PyDiGraph,
    /,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
    resolution: float | None = ...,
    equidistant: bool | None = ...,
) -> Pos2DMapping: ...
def graph_spiral_layout(
    graph: PyGraph,
    /,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
    resolution: float | None = ...,
    equidistant: bool | None = ...,
) -> Pos2DMapping: ...
def digraph_spring_layout(
    graph: PyDiGraph[_S, _T],
    pos: dict[int, tuple[float, float]] | None = ...,
    fixed: set[int] | None = ...,
    k: float | None = ...,
    repulsive_exponent: int = ...,
    adaptive_cooling: bool = ...,
    num_iter: int = ...,
    tol: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    scale: float = ...,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
    /,
) -> Pos2DMapping: ...
def graph_spring_layout(
    graph: PyGraph[_S, _T],
    pos: dict[int, tuple[float, float]] | None = ...,
    fixed: set[int] | None = ...,
    k: float | None = ...,
    repulsive_exponent: int = ...,
    adaptive_cooling: bool = ...,
    num_iter: int = ...,
    tol: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    scale: float = ...,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
    /,
) -> Pos2DMapping: ...
