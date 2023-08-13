# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/matcing/mod.rs

from .iterators import *
from .graph import PyGraph
from .digraph import PyDiGraph

from typing import Optional, Set, TypeVar, Tuple, Callable

_S = TypeVar("_S")
_T = TypeVar("_T")

def max_weight_matching(
    graph: PyGraph[_S, _T],
    /,
    max_cardinality: bool = ...,
    weight_fn: Optional[Callable[[_T], float]] = ...,
    default_weight: int = ...,
    verify_optimum: bool = ...,
) -> Set[Tuple[int, int]]: ...
def is_matching(
    graph: PyGraph,
    matching: Set[Tuple[int, int]],
    /,
) -> bool: ...
def is_maximal_matching(
    graph: PyGraph,
    matching: Set[Tuple[int, int]],
    /,
) -> bool: ...