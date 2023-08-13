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

from .centrality import *
from .connectivity import *
from .dag_algo import *
from .isomorphism import *
from .layout import *
from .link_analysis import *
from .matching import *
from .random_graph import *
from .shortest_path import *
from .traversal import *
from .tree import *

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

def digraph_union(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    /,
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyDiGraph[_S, _T]: ...
def graph_union(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    /,
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyGraph[_S, _T]: ...
def is_planar(graph: PyGraph, /) -> bool: ...
def read_graphml(path: str, /) -> List[Union[PyGraph, PyDiGraph]]: ...
def digraph_transitivity(graph: PyGraph, /) -> float: ...
def graph_transitivity(graph: PyGraph, /) -> float: ...
def graph_token_swapper(
    graph: PyGraph,
    mapping: Dict[int, int],
    /,
    trials: Optional[int] = ...,
    seed: Optional[int] = ...,
    parallel_threshold: Optional[int] = ...,
) -> EdgeList: ...
def graph_greedy_color(graph: PyGraph, /) -> Dict[int, int]: ...
def graph_greedy_edge_color(graph: PyGraph, /) -> Dict[int, int]: ...

"""
TopologicalSorter is not present in stub
digraph_cartesian_product is not present in stub
digraph_node_link_json is not present in stub
digraph_tensor_product is not present in stub
digraph_vf2_mapping is not present in stub
generators is not present in stub
graph_cartesian_product is not present in stub
graph_line_graph is not present in stub
graph_node_link_json is not present in stub
graph_tensor_product is not present in stub
graph_vf2_mapping is not present in stub
layers is not present in stub
metric_closure is not present in stub
simple_cycles is not present in stub
stoer_wagner_min_cut is not present in stub
"""
