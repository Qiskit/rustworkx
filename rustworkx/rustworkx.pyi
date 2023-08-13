# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/lib.rs

import numpy as np

from .iterators import *
from .graph import PyGraph as PyGraph
from .digraph import PyDiGraph as PyDiGraph

from typing import Optional, Set, List, Dict, TypeVar, Tuple, Callable, Union

from .centrality import *
from .connectivity import *
from .dag_algo import *
from .isomorphism import *
from .layout import *
from .link_analysis import *
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

def digraph_core_number(
    graph: PyDiGraph,
    /,
) -> int: ...
def graph_core_number(
    graph: PyGraph,
    /,
) -> int: ...
def digraph_all_simple_paths(
    graph: PyDiGraph,
    origin: int,
    to: int,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> List[List[int]]: ...
def graph_all_simple_paths(
    graph: PyGraph,
    origin: int,
    to: int,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> List[List[int]]: ...
def digraph_all_pairs_all_simple_paths(
    graph: PyDiGraph,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> AllPairsMultiplePathMapping: ...
def graph_all_pairs_all_simple_paths(
    graph: PyGraph,
    /,
    min_depth: Optional[int] = ...,
    cutoff: Optional[int] = ...,
) -> AllPairsMultiplePathMapping: ...
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
def directed_gnm_random_graph(
    num_nodes: int,
    num_edges: int,
    /,
    seed: Optional[int] = ...,
) -> PyDiGraph: ...
def undirected_gnm_random_graph(
    num_nodes: int,
    num_edges: int,
    /,
    seed: Optional[int] = ...,
) -> PyGraph: ...
def directed_gnp_random_graph(
    num_nodes: int,
    probability: float,
    /,
    seed: Optional[int] = ...,
) -> PyDiGraph: ...
def undirected_gnp_random_graph(
    num_nodes: int,
    probability: float,
    /,
    seed: Optional[int] = ...,
) -> PyGraph: ...
def read_graphml(path: str, /) -> List[Union[PyGraph, PyDiGraph]]: ...
def digraph_longest_simple_path(graph: PyDiGraph, /) -> Optional[NodeIndices]: ...
def graph_longest_simple_path(graph: PyGraph, /) -> Optional[NodeIndices]: ...
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
random_geometric_graph is not present in stub
simple_cycles is not present in stub
stoer_wagner_min_cut is not present in stub
"""
