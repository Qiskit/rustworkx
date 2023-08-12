# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/lib.rs

from .iterators import *
from .graph import PyGraph as PyGraph
from .digraph import PyDiGraph as PyDiGraph

from typing import Optional, TypeVar, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

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
