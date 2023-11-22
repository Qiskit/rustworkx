# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/cartesian_product.rs

from .iterators import *
from .graph import PyGraph
from .digraph import PyDiGraph

def digraph_cartesian_product(
    first: PyDiGraph,
    second: PyDiGraph,
    /,
) -> tuple[PyDiGraph, ProductNodeMap]: ...
def graph_cartesian_product(
    first: PyGraph,
    second: PyGraph,
    /,
) -> tuple[PyGraph, ProductNodeMap]: ...
