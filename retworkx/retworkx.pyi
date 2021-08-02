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
import retworkx_generators as generators
from .custom_return_types import *

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    TypeVar,
    Optional,
    List,
    Tuple,
    Union,
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

class PyDiGraph(Generic[S, T]):
    check_cycle: bool = ...
    multigraph: bool = ...
    def __init__(
        self,
        check_cycle: bool = False,
        multigraph: bool = True,
        /,
    ) -> None: ...
    def add_child(self, parent: int, obj: S, edge: T, /) -> int: ...
    def add_edge(self, parent: int, child: int, edge: T, /) -> int: ...
    def add_edges_from(
        self,
        obj_list: List[Union[Tuple[int, int, T], List[Union[int, T]]]],
        /,
    ) -> List[int]: ...
    def add_edges_from_no_data(
        self, obj_list: Union[List[Tuple[int, int]], List[List[int]]], /
    ) -> List[int]: ...
    def add_node(self, obj: S, /) -> int: ...
    def add_nodes_from(self, obj_list: List[S], /) -> NodeIndices: ...
    def add_parent(self, child: int, obj: S, edge: T, /) -> int: ...
    def adj(self, node: int, /) -> Dict[int, T]: ...
    def adj_direction(self, node: int, direction: bool, /) -> Dict[int, T]: ...
    def compose(
        self,
        other: PyDiGraph[S, T],
        node_map: Dict[int, Tuple[int, T]],
        /,
        node_map_func: Optional[Callable[[S], int]] = None,
        edge_map_func: Optional[Callable[[T], int]] = None,
    ) -> Dict[int, int]: ...
    def copy(self) -> PyDiGraph[S, T]: ...
    def edge_index_map(self) -> EdgeIndexMap: ...
    def edge_indices(self) -> EdgeIndices: ...
    def edge_list(self) -> EdgeList: ...
    def edges(self) -> List[T]: ...
    def extend_from_edge_list(
        self, edge_list: Union[List[Tuple[int, int]], List[List[int]]], /
    ) -> None: ...
    def extend_from_weighted_edge_list(
        self,
        edge_list: List[Union[Tuple[int, int, T], List[Union[int, T]]]],
        /,
    ) -> None: ...
    def find_adjacent_node_by_edge(
        self, node: int, predicate: Callable[[T], bool], /
    ) -> S: ...
    def find_node_by_weight(
        self, obj: Callable[[S], bool]
    ) -> Optional[int]: ...
    def find_predecessors_by_edge(
        self, node: int, filter_fn: Callable[[T], bool], /
    ) -> List[S]: ...
    def find_successors_by_edge(
        self, node: int, filter_fn: Callable[[T], bool], /
    ) -> List[S]: ...
    @classmethod
    def from_adjacency_matrix(matrix: np.array, /) -> PyDiGraph: ...
    def get_all_edge_data(self, node_a: int, node_b: int, /) -> List[T]: ...
    def get_edge_data(self, node_a: int, node_b: int, /) -> List[T]: ...
    def get_node_data(self, node: int, /) -> S: ...
    def has_edge(self, node_a: int, node_b: int, /) -> bool: ...
    def in_degree(self, node: int, /) -> int: ...
    def in_edges(self, node: int, /) -> WeightedEdgeList: ...
    def insert_node_on_in_edges(self, node: int, ref_node: int, /) -> None: ...
    def insert_node_on_in_edges_multiple(
        self, node: int, ref_nodes: List[int], /
    ) -> None: ...
    def insert_node_on_out_edges(self, node: int, ref_node: int, /) -> None: ...
    def insert_node_on_out_edges_multiple(
        self, node: int, ref_nodes: List[int], /
    ) -> None: ...
    def is_symmetric(self) -> bool: ...
    def merge_nodes(self, u: int, v: int, /) -> None: ...
    def neighbors(self, node: int, /) -> NodeIndices: ...
    def node_indexes(self) -> NodeIndices: ...
    def nodes(self) -> List[S]: ...
    def num_edges(self) -> int: ...
    def num_nodes(self) -> int: ...
    def out_degree(self, node: int, /) -> int: ...
    def out_edges(self, node: int, /) -> WeightedEdgeList: ...
    def predecessor_indices(self, node: int, /) -> NodeIndices: ...
    def predecessors(self, node: int, /) -> List[S]: ...
    @classmethod
    def read_edge_list(
        path: str,
        /,
        comment: Optional[str] = None,
        deliminator: Optional[str] = None,
    ) -> PyDiGraph: ...
    def remove_edge(self, parent: int, child: int, /) -> None: ...
    def remove_edge_from_index(self, edge: int, /) -> None: ...
    def remove_edges_from(
        self, index_list: List[Union[Tuple[int, int], List[int]]], /
    ) -> None: ...
    def remove_node(self, node: int, /) -> None: ...
    def remove_node_retain_edges(
        self,
        node: int,
        /,
        use_outgoing: Optional[bool] = None,
        condition: Optional[Callable[[S, S], bool]] = None,
    ) -> None: ...
    def remove_nodes_from(self, index_list: List[int], /) -> None: ...
    def subgraph(self, nodes: List[int], /) -> PyDiGraph[S, T]: ...
    def substitute_node_with_subgraph(
        self,
        node: int,
        other: PyDiGraph[S, T],
        edge_map_fn: Callable[[int, int, T], Optional[int]],
        /,
        node_filter: Optional[Callable[[S], bool]] = None,
        edge_weight_map: Optional[Callable[T], T] = None,
    ) -> NodeMap: ...
    def successor_indices(self, node: int, /) -> NodeIndices: ...
    def successors(self, node: int, /) -> List[S]: ...
    def to_dot(
        self,
        /,
        node_attr: Optional[Callable[[S], Dict[str, str]]] = None,
        edge_attr: Optional[Callable[[T], Dict[str, str]]] = None,
        graph_attr: Optional[Dict[str, str]] = None,
        filename: Optional[str] = None,
    ) -> Any: ...
    def to_undirected(
        self,
        /,
        multigraph: bool = True,
        weight_combo_fn: Optional[Callable[[T, T], T]] = None,
    ) -> PyGraph[S, T]: ...
    def update_edge(self, source: int, target: int, edge: T, /) -> None: ...
    def update_edge_by_index(self, edge_index: int, edge: T, /) -> None: ...
    def weighted_edge_list(self, *args, **kwargs) -> Any: ...
    def write_edge_list(
        self,
        path: str,
        /,
        deliminator: Optional[str] = None,
        weight_fn: Optional[Callable[[T]], str] = None,
    ) -> None: ...
    def __delitem__(self, idx: int, /) -> None: ...
    def __getitem__(self, idx: int, /) -> S: ...
    def __getstate__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __setitem__(self, idx: int, value: S, /) -> None: ...
    def __setstate__(self, state, /) -> None: ...

class PyGraph(Generic[S, T]):
    multigraph: bool = ...
    @classmethod
    def __init__(self, *args, **kwargs) -> None: ...
    def add_edge(self, *args, **kwargs) -> Any: ...
    def add_edges_from(self, *args, **kwargs) -> Any: ...
    def add_edges_from_no_data(self, *args, **kwargs) -> Any: ...
    def add_node(self, *args, **kwargs) -> Any: ...
    def add_nodes_from(self, *args, **kwargs) -> Any: ...
    def adj(self, *args, **kwargs) -> Any: ...
    def compose(other_graph, node_map) -> Any: ...
    def copy(self, *args, **kwargs) -> Any: ...
    def degree(self, *args, **kwargs) -> Any: ...
    def edge_index_map(self, *args, **kwargs) -> Any: ...
    def edge_indices(self, *args, **kwargs) -> Any: ...
    def edge_list(self, *args, **kwargs) -> Any: ...
    def edges(self, *args, **kwargs) -> Any: ...
    def extend_from_edge_list(self, *args, **kwargs) -> Any: ...
    def extend_from_weighted_edge_list(self, *args, **kwargs) -> Any: ...
    def from_adjacency_matrix(self, *args, **kwargs) -> Any: ...
    def get_all_edge_data(self, *args, **kwargs) -> Any: ...
    def get_edge_data(self, *args, **kwargs) -> Any: ...
    def get_node_data(self, *args, **kwargs) -> Any: ...
    def has_edge(self, *args, **kwargs) -> Any: ...
    def neighbors(self, *args, **kwargs) -> Any: ...
    def node_indexes(self, *args, **kwargs) -> Any: ...
    def nodes(self, *args, **kwargs) -> Any: ...
    def num_edges(self, *args, **kwargs) -> Any: ...
    def num_nodes(self, *args, **kwargs) -> Any: ...
    def read_edge_list(self, *args, **kwargs) -> Any: ...
    def remove_edge(self, *args, **kwargs) -> Any: ...
    def remove_edge_from_index(self, *args, **kwargs) -> Any: ...
    def remove_edges_from(self, *args, **kwargs) -> Any: ...
    def remove_node(self, *args, **kwargs) -> Any: ...
    def remove_nodes_from(self, *args, **kwargs) -> Any: ...
    def subgraph(self, *args, **kwargs) -> Any: ...
    def to_dot(lambdanode) -> Any: ...
    def update_edge(self, *args, **kwargs) -> Any: ...
    def update_edge_by_index(self, *args, **kwargs) -> Any: ...
    def weighted_edge_list(self, *args, **kwargs) -> Any: ...
    def write_edge_list(self, *args, **kwargs) -> Any: ...
    def __delitem__(self, other) -> Any: ...
    def __getitem__(self, index) -> Any: ...
    def __getstate__(self) -> Any: ...
    def __len__(self) -> Any: ...
    def __setitem__(self, index, object) -> Any: ...
    def __setstate__(self, state) -> Any: ...

def ancestors(*args, **kwargs) -> Any: ...
def bfs_successors(*args, **kwargs) -> Any: ...
def collect_runs(*args, **kwargs) -> Any: ...
def cycle_basis(*args, **kwargs) -> Any: ...
def dag_longest_path(*args, **kwargs) -> Any: ...
def dag_longest_path_length(*args, **kwargs) -> Any: ...
def dag_weighted_longest_path(*args, **kwargs) -> Any: ...
def dag_weighted_longest_path_length(*args, **kwargs) -> Any: ...
def descendants(*args, **kwargs) -> Any: ...
def digraph_adjacency_matrix(*args, **kwargs) -> Any: ...
def digraph_all_pairs_dijkstra_path_lengths(*args, **kwargs) -> Any: ...
def digraph_all_pairs_dijkstra_shortest_paths(*args, **kwargs) -> Any: ...
def digraph_all_simple_paths(*args, **kwargs) -> Any: ...
def digraph_astar_shortest_path(*args, **kwargs) -> Any: ...
def digraph_bipartite_layout(*args, **kwargs) -> Any: ...
def digraph_circular_layout(*args, **kwargs) -> Any: ...
def digraph_complement(*args, **kwargs) -> Any: ...
def digraph_core_number(*args, **kwargs) -> Any: ...
def digraph_dfs_edges(*args, **kwargs) -> Any: ...
def digraph_dijkstra_shortest_path_lengths(*args, **kwargs) -> Any: ...
def digraph_dijkstra_shortest_paths(*args, **kwargs) -> Any: ...
def digraph_distance_matrix(*args, **kwargs) -> Any: ...
def digraph_find_cycle(*args, **kwargs) -> Any: ...
def digraph_floyd_warshall(graph, weight_fn=...) -> Any: ...
def digraph_floyd_warshall_numpy(*args, **kwargs) -> Any: ...
def digraph_is_isomorphic(*args, **kwargs) -> Any: ...
def digraph_is_subgraph_isomorphic(*args, **kwargs) -> Any: ...
def digraph_k_shortest_path_lengths(*args, **kwargs) -> Any: ...
def digraph_num_shortest_paths_unweighted(*args, **kwargs) -> Any: ...
def digraph_random_layout(*args, **kwargs) -> Any: ...
def digraph_shell_layout(*args, **kwargs) -> Any: ...
def digraph_spiral_layout(*args, **kwargs) -> Any: ...
def digraph_spring_layout(*args, **kwargs) -> Any: ...
def digraph_transitivity(*args, **kwargs) -> Any: ...
def digraph_union(*args, **kwargs) -> Any: ...
def directed_gnm_random_graph(*args, **kwargs) -> Any: ...
def directed_gnp_random_graph(*args, **kwargs) -> Any: ...
def graph_adjacency_matrix(graph, weight_fn) -> Any: ...
def graph_all_pairs_dijkstra_path_lengths(*args, **kwargs) -> Any: ...
def graph_all_pairs_dijkstra_shortest_paths(*args, **kwargs) -> Any: ...
def graph_all_simple_paths(*args, **kwargs) -> Any: ...
def graph_astar_shortest_path(*args, **kwargs) -> Any: ...
def graph_bipartite_layout(*args, **kwargs) -> Any: ...
def graph_circular_layout(*args, **kwargs) -> Any: ...
def graph_complement(*args, **kwargs) -> Any: ...
def graph_core_number(*args, **kwargs) -> Any: ...
def graph_dfs_edges(*args, **kwargs) -> Any: ...
def graph_dijkstra_shortest_path_lengths(*args, **kwargs) -> Any: ...
def graph_dijkstra_shortest_paths(*args, **kwargs) -> Any: ...
def graph_distance_matrix(*args, **kwargs) -> Any: ...
def graph_floyd_warshall(graph, weight_fn=...) -> Any: ...
def graph_floyd_warshall_numpy(graph, weight_fn) -> Any: ...
def graph_greedy_color(*args, **kwargs) -> Any: ...
def graph_is_isomorphic(*args, **kwargs) -> Any: ...
def graph_is_subgraph_isomorphic(*args, **kwargs) -> Any: ...
def graph_k_shortest_path_lengths(*args, **kwargs) -> Any: ...
def graph_num_shortest_paths_unweighted(*args, **kwargs) -> Any: ...
def graph_random_layout(*args, **kwargs) -> Any: ...
def graph_shell_layout(*args, **kwargs) -> Any: ...
def graph_spiral_layout(*args, **kwargs) -> Any: ...
def graph_spring_layout(*args, **kwargs) -> Any: ...
def graph_transitivity(*args, **kwargs) -> Any: ...
def is_directed_acyclic_graph(*args, **kwargs) -> Any: ...
def is_matching(*args, **kwargs) -> Any: ...
def is_maximal_matching(*args, **kwargs) -> Any: ...
def is_weakly_connected(*args, **kwargs) -> Any: ...
def layers(*args, **kwargs) -> Any: ...
def lexicographical_topological_sort(*args, **kwargs) -> Any: ...
def max_weight_matching(*args, **kwargs) -> Any: ...
def minimum_spanning_edges(graph, weight_fn) -> Any: ...
def minimum_spanning_tree(graph, weight_fn) -> Any: ...
def number_weakly_connected_components(*args, **kwargs) -> Any: ...
def random_geometric_graph(*args, **kwargs) -> Any: ...
def strongly_connected_components(*args, **kwargs) -> Any: ...
def topological_sort(*args, **kwargs) -> Any: ...
def undirected_gnm_random_graph(*args, **kwargs) -> Any: ...
def undirected_gnp_random_graph(*args, **kwargs) -> Any: ...
def weakly_connected_components(*args, **kwargs) -> Any: ...
