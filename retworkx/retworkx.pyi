# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and lib.rs

import numpy as np
from .custom_return_types import *
from .pygraph import PyGraph as PyGraph
from .pydigraph import PyDiGraph as PyDiGraph

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    TypeVar,
    Optional,
    List,
    Tuple,
)

S = TypeVar("S")
T = TypeVar("T")

class DAGHasCycle(Exception): ...
class DAGWouldCycle(Exception): ...
class InvalidNode(Exception): ...
class NoEdgeBetweenNodes(Exception): ...
class NoPathFound(Exception): ...
class NoSuitableNeighbors(Exception): ...
class NullGraph(Exception): ...
