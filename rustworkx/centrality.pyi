# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/centrality.rs

from .iterators import *
from .graph import PyGraph
from .digraph import PyDiGraph

from typing import TypeVar, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

def digraph_eigenvector_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
def graph_eigenvector_centrality(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
def digraph_betweenness_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    normalized: bool = ...,
    endpoints: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def graph_betweenness_centrality(
    graph: PyGraph[_S, _T],
    /,
    normalized: bool = ...,
    endpoints: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def digraph_edge_betweenness_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    normalized: bool = ...,
    parallel_threshold: int = ...,
) -> EdgeCentralityMapping: ...
def graph_edge_betweenness_centrality(
    graph: PyGraph[_S, _T],
    /,
    normalized: bool = ...,
    parallel_threshold: int = ...,
) -> EdgeCentralityMapping: ...
def digraph_closeness_centrality(
    graph: PyDiGraph[_S, _T],
    wf_improved: bool = ...,
) -> CentralityMapping: ...
def graph_closeness_centrality(
    graph: PyGraph[_S, _T],
    wf_improved: bool = ...,
) -> CentralityMapping: ...
def digraph_katz_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    alpha: float | None = ...,
    beta: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float | None = ...,
    max_iter: int | None = ...,
    tol: float | None = ...,
) -> CentralityMapping: ...
def graph_katz_centrality(
    graph: PyGraph[_S, _T],
    /,
    alpha: float | None = ...,
    beta: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float | None = ...,
    max_iter: int | None = ...,
    tol: float | None = ...,
) -> CentralityMapping: ...
