# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/dag_algo/mod.rs

from .iterators import *
from .digraph import PyDiGraph

from typing import Optional, TypeVar, Callable, Union

_S = TypeVar("_S")
_T = TypeVar("_T")

def collect_runs(
    graph: PyDiGraph[_S, _T],
    filter_fn: Callable[[_S], bool],
) -> list[list[_S]]: ...
def collect_bicolor_runs(
    graph: PyDiGraph[_S, _T],
    filter_fn: Callable[[_S], bool],
    color_fn: Callable[[_T], int],
) -> list[list[_S]]: ...
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
def is_directed_acyclic_graph(graph: PyDiGraph, /) -> bool: ...
def topological_sort(graph: PyDiGraph, /) -> NodeIndices: ...
def lexicographical_topological_sort(
    dag: PyDiGraph[_S, _T],
    key: Callable[[_S], str],
    /,
) -> list[_S]: ...
def transitive_reduction(graph: PyDiGraph, /) -> tuple[PyDiGraph, dict[int, int]]: ...
def layers(
    dag: PyDiGraph[_S, _T],
    first_layer: list[int],
    /,
    index_output: bool = ...,
) -> Union[list[_S], list[int]]: ...
