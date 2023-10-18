# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/link_analysis.rs

from .iterators import *
from .digraph import PyDiGraph

from typing import TypeVar, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

def hits(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    nstart: dict[int, float] | None = ...,
    tol: float | None = ...,
    max_iter: int | None = ...,
    normalized: bool | None = ...,
) -> tuple[CentralityMapping, CentralityMapping]: ...
def pagerank(
    graph: PyDiGraph[_S, _T],
    /,
    alpha: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    nstart: dict[int, float] | None = ...,
    personalization: dict[int, float] | None = ...,
    tol: float | None = ...,
    max_iter: int | None = ...,
    dangling: dict[int, float] | None = ...,
) -> CentralityMapping: ...
