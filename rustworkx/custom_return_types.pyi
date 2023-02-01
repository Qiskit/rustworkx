# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and lib.rs

from typing import Any, Generic, List, ItemsView, KeysView, ValuesView, Iterator, Mapping, TypeVar, Tuple, overload
from abc import ABC
from collections.abc import  Sequence
from typing_extensions import Self

S = TypeVar("S")
T_co = TypeVar("T_co", covariant=True)

__all__ = [
    'NodeIndices',
    'PathLengthMapping',
    'PathMapping',
    'AllPairsPathLengthMapping',
    'AllPairsPathMapping',
    'BFSSuccessors',
    'EdgeIndexMap',
    'EdgeIndices',
    'Chains',
    'EdgeList',
    'NodeMap',
    'NodesCountMapping',
    'Pos2DMapping',
    'WeightedEdgeList',
]

class RustworkxCustomVecIter(Generic[T_co], Sequence[T_co], ABC):
    def __init__(self) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    @overload
    def __getitem__(self, index: int) -> T_co: ...
    @overload
    def __getitem__(self: Self, index: slice) -> Self: ...
    def __getstate__(self) -> Any: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: Sequence[T_co]) -> None: ...

class RustworkxCustomHashMapIter(Generic[S, T_co], Mapping[S, T_co], ABC):
    def __init__(self) -> None: ...
    def items(self) -> ItemsView[S, T_co]: ...
    def keys(self) -> KeysView[S]: ...
    def values(self) -> ValuesView[T_co]: ...
    def __contains__(self, other: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __getitem__(self, index: S) -> T_co: ...
    def __getstate__(self) -> Any: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __iter__(self) -> Iterator[S]: ...
    def __len__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: Mapping[S, T_co]) -> None: ...

class NodeIndices(RustworkxCustomVecIter[int]): ...
class PathLengthMapping(RustworkxCustomHashMapIter[int, float]): ...
class PathMapping(RustworkxCustomHashMapIter[int, NodeIndices]): ...
class AllPairsPathLengthMapping(RustworkxCustomHashMapIter[int, PathLengthMapping]): ...
class AllPairsPathMapping(RustworkxCustomHashMapIter[int, PathMapping]): ...
class BFSSuccessors(Generic[T_co], RustworkxCustomVecIter[Tuple[T_co, List[T_co]]]): ...
class EdgeIndexMap(Generic[T_co], RustworkxCustomHashMapIter[int, Tuple[int, int, T_co]]): ...
class EdgeIndices(RustworkxCustomVecIter[int]): ...
class Chains(RustworkxCustomVecIter[EdgeIndices]): ...
class EdgeList(RustworkxCustomVecIter[Tuple[int, int]]): ...
class NodeMap(RustworkxCustomHashMapIter[int, int]): ...
class NodesCountMapping(RustworkxCustomHashMapIter[int, int]): ...
class Pos2DMapping(RustworkxCustomHashMapIter[int, Tuple[float, float]]): ...
class WeightedEdgeList(Generic[T_co], RustworkxCustomVecIter[Tuple[int, int, T_co]]): ...
