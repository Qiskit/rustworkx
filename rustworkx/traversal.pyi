# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/traversal/mod.rs

from .iterators import *
from .graph import PyGraph
from .digraph import PyDiGraph
from .visit import BFSVisitor, DFSVisitor, DijkstraVisitor

from typing import Any, Optional, TypeVar, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")
_BFSVisitor = TypeVar("_BFSVisitor", bound=BFSVisitor)
_DFSVisitor = TypeVar("_DFSVisitor", bound=DFSVisitor)
_DijkstraVisitor = TypeVar("_DijkstraVisitor", bound=DijkstraVisitor)

def digraph_bfs_search(
    graph: PyDiGraph,
    source: Optional[int] = ...,
    visitor: Optional[_BFSVisitor] = ...,
) -> None: ...
def graph_bfs_search(
    graph: PyGraph,
    source: Optional[int] = ...,
    visitor: Optional[_BFSVisitor] = ...,
) -> None: ...
def digraph_dfs_search(
    graph: PyDiGraph,
    source: Optional[int] = ...,
    visitor: Optional[_DFSVisitor] = ...,
) -> None: ...
def graph_dfs_search(
    graph: PyGraph,
    source: Optional[int] = ...,
    visitor: Optional[_DFSVisitor] = ...,
) -> None: ...
def digraph_dijkstra_search(
    graph: PyDiGraph,
    source: Optional[int] = ...,
    weight_fn: Optional[Callable[[Any], float]] = ...,
    visitor: Optional[_DijkstraVisitor] = ...,
) -> None: ...
def graph_dijkstra_search(
    graph: PyGraph,
    source: Optional[int] = ...,
    weight_fn: Optional[Callable[[Any], float]] = ...,
    visitor: Optional[_DijkstraVisitor] = ...,
) -> None: ...
def digraph_dfs_edges(graph: PyDiGraph[_S, _T], /, source: Optional[int] = ...) -> EdgeList: ...
def graph_dfs_edges(graph: PyGraph[_S, _T], /, source: Optional[int] = ...) -> EdgeList: ...
def ancestors(graph: PyDiGraph, node: int, /) -> set[int]: ...
def bfs_predecessors(graph: PyDiGraph, node: int, /) -> BFSPredecessors: ...
def bfs_successors(graph: PyDiGraph, node: int, /) -> BFSSuccessors: ...
def descendants(graph: PyDiGraph, node: int, /) -> set[int]: ...
