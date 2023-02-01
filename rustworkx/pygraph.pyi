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

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    TypeVar,
    Optional,
    List,
    Tuple,
    Sequence,
)

__all__ = ['PyGraph']

S = TypeVar("S")
T = TypeVar("T")

class PyGraph(Generic[S, T]):
    multigraph: bool = ...
    def __init__(self, /, multigraph: bool = ...) -> None: ...
    def add_edge(self, node_a: int, node_b: int, edge: T, /) -> int: ...
    def add_edges_from(self, obj_list: Sequence[Tuple[int, int, T]]) -> List[int]: ...
    def add_edges_from_no_data(
        self: PyGraph[S, Optional[T]], obj_list: Sequence[Tuple[int, int]], /
    ) -> List[int]: ...
    def add_node(self, obj: S, /) -> int: ...
    def add_nodes_from(self, obj_list: Sequence[S], /) -> NodeIndices: ...
    def adj(self, node: int, /) -> Dict[int, T]: ...
    def compose(
        self,
        other: PyGraph[S, T],
        node_map: Dict[int, Tuple[int, T]],
        /,
        node_map_func: Optional[Callable[[S], int]] = ...,
        edge_map_func: Optional[Callable[[T], int]] = ...,
    ) -> Dict[int, int]: ...
    def copy(self) -> PyGraph[S, T]: ...
    def degree(self, node: int, /) -> int: ...
    def edge_index_map(self) -> EdgeIndexMap[T]: ...
    def edge_indices(self) -> EdgeIndices: ...
    def edge_list(self) -> EdgeList: ...
    def edges(self) -> List[T]: ...
    def extend_from_edge_list(
        self: PyGraph[Optional[S], Optional[T]], edge_list: Sequence[Tuple[int, int]], /
    ) -> None: ...
    def extend_from_weighted_edge_list(
        self: PyGraph[Optional[S], T],
        edge_list: Sequence[Tuple[int, int, T]],
        /,
    ) -> None: ...
    @classmethod
    def from_adjacency_matrix(matrix: np.array, /) -> PyGraph[int, float]: ...
    @classmethod
    def from_complex_adjacency_matrix(matrix: np.array, /) -> PyGraph[int, complex]: ...
    def get_all_edge_data(self, node_a: int, node_b: int, /) -> List[T]: ...
    def get_edge_data(self, node_a: int, node_b: int, /) -> T: ...
    def get_node_data(self, node: int, /) -> S: ...
    def has_edge(self, node_a: int, node_b: int, /) -> bool: ...
    def neighbors(self, node: int, /) -> NodeIndices: ...
    def node_indexes(self) -> NodeIndices: ...
    def nodes(self) -> List[S]: ...
    def num_edges(self) -> int: ...
    def num_nodes(self) -> int: ...
    @classmethod
    def read_edge_list(
        path: str,
        /,
        comment: Optional[str] = ...,
        deliminator: Optional[str] = ...,
    ) -> PyGraph: ...
    def remove_edge(self, node_a: int, node_b: int, /) -> None: ...
    def remove_edge_from_index(self, edge: int, /) -> None: ...
    def remove_edges_from(self, index_list: Sequence[Tuple[int, int]], /) -> None: ...
    def remove_node(self, node: int, /) -> None: ...
    def remove_nodes_from(self, index_list: Sequence[int], /) -> None: ...
    def subgraph(self, nodes: Sequence[int], /) -> PyGraph[S, T]: ...
    def to_dot(
        self,
        /,
        node_attr: Optional[Callable[[S], Dict[str, str]]] = ...,
        edge_attr: Optional[Callable[[T], Dict[str, str]]] = ...,
        graph_attr: Optional[Dict[str, str]] = ...,
        filename: Optional[str] = ...,
    ) -> Optional[str]: ...
    def update_edge(self, source: int, target: int, edge: T, /) -> None: ...
    def update_edge_by_index(self, edge_index: int, edge: T, /) -> None: ...
    def weighted_edge_list(self) -> WeightedEdgeList[T]: ...
    def write_edge_list(
        self,
        path: str,
        /,
        deliminator: Optional[str] = ...,
        weight_fn: Optional[Callable[[T], str]] = ...,
    ) -> None: ...
    def __delitem__(self, idx: int, /) -> None: ...
    def __getitem__(self, idx: int, /) -> S: ...
    def __getstate__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __setitem__(self, idx: int, value: S, /) -> None: ...
    def __setstate__(self, state, /) -> None: ...
