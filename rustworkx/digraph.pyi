# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/digraph.rs

import numpy as np
import sys
from .iterators import *
from .graph import PyGraph

assert sys.version_info >= (3, 8)

from typing import Any, Callable, Dict, Generic, TypeVar, Optional, List, Tuple, Sequence

__all__ = ["PyDiGraph"]

S = TypeVar("S")
T = TypeVar("T")

class PyDiGraph(Generic[S, T]):
    check_cycle: bool
    multigraph: bool
    def __init__(
        self,
        /,
        check_cycle: bool = ...,
        multigraph: bool = ...,
    ) -> None: ...
    def add_child(self, parent: int, obj: S, edge: T, /) -> int: ...
    def add_edge(self, parent: int, child: int, edge: T, /) -> int: ...
    def add_edges_from(
        self,
        obj_list: Sequence[Tuple[int, int, T]],
        /,
    ) -> List[int]: ...
    def add_edges_from_no_data(
        self: PyDiGraph[S, Optional[T]], obj_list: Sequence[Tuple[int, int]], /
    ) -> List[int]: ...
    def add_node(self, obj: S, /) -> int: ...
    def add_nodes_from(self, obj_list: Sequence[S], /) -> NodeIndices: ...
    def add_parent(self, child: int, obj: S, edge: T, /) -> int: ...
    def adj(self, node: int, /) -> Dict[int, T]: ...
    def adj_direction(self, node: int, direction: bool, /) -> Dict[int, T]: ...
    def compose(
        self,
        other: PyDiGraph[S, T],
        node_map: Dict[int, Tuple[int, T]],
        /,
        node_map_func: Optional[Callable[[S], int]] = ...,
        edge_map_func: Optional[Callable[[T], int]] = ...,
    ) -> Dict[int, int]: ...
    def copy(self) -> PyDiGraph[S, T]: ...
    def edge_index_map(self) -> EdgeIndexMap[T]: ...
    def edge_indices(self) -> EdgeIndices: ...
    def edge_list(self) -> EdgeList: ...
    def edges(self) -> List[T]: ...
    def extend_from_edge_list(
        self: PyDiGraph[Optional[S], Optional[T]], edge_list: Sequence[Tuple[int, int]], /
    ) -> None: ...
    def extend_from_weighted_edge_list(
        self: PyDiGraph[Optional[S], T],
        edge_list: Sequence[Tuple[int, int, T]],
        /,
    ) -> None: ...
    def find_adjacent_node_by_edge(self, node: int, predicate: Callable[[T], bool], /) -> S: ...
    def find_node_by_weight(
        self,
        obj: Callable[[S], bool],
        /,
    ) -> Optional[int]: ...
    def find_predecessors_by_edge(
        self, node: int, filter_fn: Callable[[T], bool], /
    ) -> List[S]: ...
    def find_successors_by_edge(self, node: int, filter_fn: Callable[[T], bool], /) -> List[S]: ...
    @staticmethod
    def from_adjacency_matrix(
        matrix: np.ndarray, /, null_value: float = ...
    ) -> PyDiGraph[int, float]: ...
    @staticmethod
    def from_complex_adjacency_matrix(
        matrix: np.ndarray, /, null_value: complex = ...
    ) -> PyDiGraph[int, complex]: ...
    def get_all_edge_data(self, node_a: int, node_b: int, /) -> List[T]: ...
    def get_edge_data(self, node_a: int, node_b: int, /) -> T: ...
    def get_node_data(self, node: int, /) -> S: ...
    def has_edge(self, node_a: int, node_b: int, /) -> bool: ...
    def in_degree(self, node: int, /) -> int: ...
    def in_edges(self, node: int, /) -> WeightedEdgeList[T]: ...
    def insert_node_on_in_edges(self, node: int, ref_node: int, /) -> None: ...
    def insert_node_on_in_edges_multiple(self, node: int, ref_nodes: Sequence[int], /) -> None: ...
    def insert_node_on_out_edges(self, node: int, ref_node: int, /) -> None: ...
    def insert_node_on_out_edges_multiple(self, node: int, ref_nodes: Sequence[int], /) -> None: ...
    def is_symmetric(self) -> bool: ...
    def merge_nodes(self, u: int, v: int, /) -> None: ...
    def neighbors(self, node: int, /) -> NodeIndices: ...
    def node_indexes(self) -> NodeIndices: ...
    def nodes(self) -> List[S]: ...
    def num_edges(self) -> int: ...
    def num_nodes(self) -> int: ...
    def out_degree(self, node: int, /) -> int: ...
    def out_edges(self, node: int, /) -> WeightedEdgeList[T]: ...
    def predecessor_indices(self, node: int, /) -> NodeIndices: ...
    def predecessors(self, node: int, /) -> List[S]: ...
    @staticmethod
    def read_edge_list(
        path: str,
        /,
        comment: Optional[str] = ...,
        deliminator: Optional[str] = ...,
        labels: bool = ...,
    ) -> PyDiGraph: ...
    def remove_edge(self, parent: int, child: int, /) -> None: ...
    def remove_edge_from_index(self, edge: int, /) -> None: ...
    def remove_edges_from(self, index_list: Sequence[Tuple[int, int]], /) -> None: ...
    def remove_node(self, node: int, /) -> None: ...
    def remove_node_retain_edges(
        self,
        node: int,
        /,
        use_outgoing: Optional[bool] = ...,
        condition: Optional[Callable[[S, S], bool]] = ...,
    ) -> None: ...
    def remove_nodes_from(self, index_list: Sequence[int], /) -> None: ...
    def subgraph(self, nodes: Sequence[int], /, preserve_attrs: bool = ...) -> PyDiGraph[S, T]: ...
    def substitute_node_with_subgraph(
        self,
        node: int,
        other: PyDiGraph[S, T],
        edge_map_fn: Callable[[int, int, T], Optional[int]],
        /,
        node_filter: Optional[Callable[[S], bool]] = ...,
        edge_weight_map: Optional[Callable[[T], T]] = ...,
    ) -> NodeMap: ...
    def successor_indices(self, node: int, /) -> NodeIndices: ...
    def successors(self, node: int, /) -> List[S]: ...
    def to_dot(
        self,
        /,
        node_attr: Optional[Callable[[S], Dict[str, str]]] = ...,
        edge_attr: Optional[Callable[[T], Dict[str, str]]] = ...,
        graph_attr: Optional[Dict[str, str]] = ...,
        filename: Optional[str] = ...,
    ) -> Optional[str]: ...
    def to_undirected(
        self,
        /,
        multigraph: bool = ...,
        weight_combo_fn: Optional[Callable[[T, T], T]] = ...,
    ) -> PyGraph[S, T]: ...
    def update_edge(
        self,
        source: int,
        target: int,
        edge: T,
        /,
    ) -> None: ...
    def update_edge_by_index(self, edge_index: int, edge: T, /) -> None: ...
    def weighted_edge_list(self) -> WeightedEdgeList[T]: ...
    def write_edge_list(
        self,
        path: str,
        /,
        deliminator: Optional[str] = ...,
        weight_fn: Optional[Callable[[T], str]] = ...,
    ) -> None: ...
    def reverse(self) -> None: ...
    def __delitem__(self, idx: int, /) -> None: ...
    def __getitem__(self, idx: int, /) -> S: ...
    def __getstate__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __setitem__(self, idx: int, value: S, /) -> None: ...
    def __setstate__(self, state, /) -> None: ...
