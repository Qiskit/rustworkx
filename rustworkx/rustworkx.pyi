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

from typing import Optional, List, Dict, TypeVar, Union

from .cartesian_product import *
from .centrality import *
from .coloring import *
from .connectivity import *
from .dag_algo import *
from .isomorphism import *
from .layout import *
from .link_analysis import *
from .matching import *
from .random_graph import *
from .shortest_path import *
from .traversal import *
from .transitivity import *
from .tree import *
from .tensor_product import *
from .union import *

_S = TypeVar("_S")
_T = TypeVar("_T")

class DAGHasCycle(Exception): ...
class DAGWouldCycle(Exception): ...
class InvalidNode(Exception): ...
class NoEdgeBetweenNodes(Exception): ...
class NoPathFound(Exception): ...
class NoSuitableNeighbors(Exception): ...
class NullGraph(Exception): ...
class NegativeCycle(Exception): ...
class JSONSerializationError(Exception): ...
class FailedToConverge(Exception): ...

def is_planar(graph: PyGraph, /) -> bool: ...
def read_graphml(path: str, /) -> List[Union[PyGraph, PyDiGraph]]: ...
def graph_token_swapper(
    graph: PyGraph,
    mapping: Dict[int, int],
    /,
    trials: Optional[int] = ...,
    seed: Optional[int] = ...,
    parallel_threshold: Optional[int] = ...,
) -> EdgeList: ...

"""
digraph_node_link_json is not present in stub
digraph_vf2_mapping is not present in stub
graph_line_graph is not present in stub
graph_node_link_json is not present in stub
graph_vf2_mapping is not present in stub
simple_cycles is not present in stub
"""
