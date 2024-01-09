# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py, src/tree.rs, and src/steiner_tree.rs

from .iterators import *
from .graph import PyGraph

from typing import TypeVar, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

def minimum_spanning_edges(
    graph: PyGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
) -> WeightedEdgeList: ...
def minimum_spanning_tree(
    graph: PyGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
) -> PyGraph[_S, _T]: ...
def steiner_tree(
    graph: PyGraph[_S, _T],
    terminal_nodes: list[int],
    weight_fn: Callable[[_T], float],
    /,
) -> PyGraph[_S, _T]: ...
def metric_closure(
    graph: PyGraph[_S, _T],
    weight_fn: Callable[[_T], float],
    /,
) -> PyGraph: ...
