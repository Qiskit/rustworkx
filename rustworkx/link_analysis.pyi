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

from typing import Optional, TypeVar, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

def hits(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    nstart: Optional[dict[int, float]] = ...,
    tol: Optional[float] = ...,
    max_iter: Optional[int] = ...,
    normalized: Optional[bool] = ...,
) -> tuple[CentralityMapping, CentralityMapping]: ...
def pagerank(
    graph: PyDiGraph[_S, _T],
    /,
    alpha: Optional[float] = ...,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    nstart: Optional[dict[int, float]] = ...,
    personalization: Optional[dict[int, float]] = ...,
    tol: Optional[float] = ...,
    max_iter: Optional[int] = ...,
    dangling: Optional[dict[int, float]] = ...,
) -> CentralityMapping: ...
