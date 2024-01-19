# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/graph.rs

import numpy as np
from .iterators import *

from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    Sequence,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .digraph import PyDiGraph

__all__ = ["PyGraph"]

S = TypeVar("S")
T = TypeVar("T")

class PyGraph(Generic[S, T]):
    attrs: Any
    multigraph: bool = ...
    def __init__(self, /, multigraph: bool = ...) -> None: ...
    def add_edge(self, node_a: int, node_b: int, edge: T, /) -> int: ...
    def add_edges_from(
        self,
        obj_list: Sequence[tuple[int, int, T]],
        /,
    ) -> list[int]: ...
    def add_edges_from_no_data(
        self: PyGraph[S, T | None], obj_list: Sequence[tuple[int, int]], /
    ) -> list[int]: ...
    def add_node(self, obj: S, /) -> int: ...
    def add_nodes_from(self, obj_list: Sequence[S], /) -> NodeIndices: ...
    def adj(self, node: int, /) -> dict[int, T]: ...
    def clear(self) -> None: ...
    def clear_edges(self) -> None: ...
    def compose(
        self,
        other: PyGraph[S, T],
        node_map: dict[int, tuple[int, T]],
        /,
        node_map_func: Callable[[S], int] | None = ...,
        edge_map_func: Callable[[T], int] | None = ...,
    ) -> dict[int, int]: ...
    def contract_nodes(
        self,
        nodes: Sequence[int],
        obj: S,
        /,
        weight_combo_fn: Callable[[T, T], T] | None = ...,
    ) -> int: ...
    def copy(self) -> PyGraph[S, T]: ...
    def degree(self, node: int, /) -> int: ...
    def edge_index_map(self) -> EdgeIndexMap[T]: ...
    def edge_indices(self) -> EdgeIndices: ...
    def edge_list(self) -> EdgeList: ...
    def edges(self) -> list[T]: ...
    def edge_subgraph(self, edge_list: Sequence[tuple[int, int]], /) -> PyGraph[S, T]: ...
    def extend_from_edge_list(
        self: PyGraph[S | None, T | None], edge_list: Sequence[tuple[int, int]], /
    ) -> None: ...
    def extend_from_weighted_edge_list(
        self: PyGraph[S | None, T],
        edge_list: Sequence[tuple[int, int, T]],
        /,
    ) -> None: ...
    def filter_edges(self, filter_function: Callable[[T], bool]) -> EdgeIndices: ...
    def filter_nodes(self, filter_function: Callable[[S], bool]) -> NodeIndices: ...
    def find_node_by_weight(
        self,
        obj: Callable[[S], bool],
        /,
    ) -> int | None: ...
    @staticmethod
    def from_adjacency_matrix(
        matrix: np.ndarray, /, null_value: float = ...
    ) -> PyGraph[int, float]: ...
    @staticmethod
    def from_complex_adjacency_matrix(
        matrix: np.ndarray, /, null_value: complex = ...
    ) -> PyGraph[int, complex]: ...
    def get_all_edge_data(self, node_a: int, node_b: int, /) -> list[T]: ...
    def get_edge_data(self, node_a: int, node_b: int, /) -> T: ...
    def get_edge_data_by_index(self, edge_index: int, /) -> T: ...
    def get_edge_endpoints_by_index(self, edge_index: int, /) -> tuple[int, int]: ...
    def get_node_data(self, node: int, /) -> S: ...
    def has_edge(self, node_a: int, node_b: int, /) -> bool: ...
    def has_parallel_edges(self) -> bool: ...
    def in_edges(self, node: int, /) -> WeightedEdgeList[T]: ...
    def incident_edge_index_map(self, node: int, /) -> EdgeIndexMap: ...
    def incident_edges(self, node: int, /) -> EdgeIndices: ...
    def neighbors(self, node: int, /) -> NodeIndices: ...
    def node_indexes(self) -> NodeIndices: ...
    def node_indices(self) -> NodeIndices: ...
    def nodes(self) -> list[S]: ...
    def num_edges(self) -> int: ...
    def num_nodes(self) -> int: ...
    def out_edges(self, node: int, /) -> WeightedEdgeList[T]: ...
    @staticmethod
    def read_edge_list(
        path: str,
        /,
        comment: str | None = ...,
        deliminator: str | None = ...,
        labels: bool = ...,
    ) -> PyGraph: ...
    def remove_edge(self, node_a: int, node_b: int, /) -> None: ...
    def remove_edge_from_index(self, edge: int, /) -> None: ...
    def remove_edges_from(self, index_list: Sequence[tuple[int, int]], /) -> None: ...
    def remove_node(self, node: int, /) -> None: ...
    def remove_nodes_from(self, index_list: Sequence[int], /) -> None: ...
    def subgraph(self, nodes: Sequence[int], /, preserve_attrs: bool = ...) -> PyGraph[S, T]: ...
    def substitute_node_with_subgraph(
        self,
        node: int,
        other: PyGraph[S, T],
        edge_map_fn: Callable[[int, int, T], int | None],
        /,
        node_filter: Callable[[S], bool] | None = ...,
        edge_weight_map: Callable[[T], T] | None = ...,
    ) -> NodeMap: ...
    def to_dot(
        self,
        /,
        node_attr: Callable[[S], dict[str, str]] | None = ...,
        edge_attr: Callable[[T], dict[str, str]] | None = ...,
        graph_attr: dict[str, str] | None = ...,
        filename: str | None = None,
    ) -> str | None: ...
    def to_directed(self) -> PyDiGraph[S, T]: ...
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
        deliminator: str | None = ...,
        weight_fn: Callable[[T], str] | None = ...,
    ) -> None: ...
    def __delitem__(self, idx: int, /) -> None: ...
    def __getitem__(self, idx: int, /) -> S: ...
    def __getstate__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __setitem__(self, idx: int, value: S, /) -> None: ...
    def __setstate__(self, state, /) -> None: ...
