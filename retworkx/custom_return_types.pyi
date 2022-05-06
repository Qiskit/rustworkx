# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and lib.rs

from typing import Any, Generic, List, Iterable, Mapping, TypeVar, Tuple, overload
from collections.abc import ABC, Sequence

# Until PEP 673 is implemented in Python 3.11, we need to use this hack
Self = TypeVar("Self", bound="RetworkxCustomVecIter")

S = TypeVar("S")
T_co = TypeVar("T_co", covariant=True)

class RetworkxCustomVecIter(Generic[T_co], Sequence[T_co], ABC):
    def __init__(self) -> None: ...
    def __eq__(self, other: Sequence[T_co]) -> bool: ...
    @overload
    def __getitem__(self, index: int) -> T_co: ...
    @overload
    def __getitem__(self: Self, index: slice) -> Self: ...
    def __getstate__(self) -> Any: ...
    def __hash__(self) -> int: ...
    def __len__(self) -> int: ...
    def __ne__(self, other: Sequence[T_co]) -> bool: ...
    def __setstate__(self, state) -> None: ...

class RetworkxCustomHashMapIter(Generic[S], Generic[T_co], Mapping[S, T_co], ABC):
    def __init__(self) -> None: ...
    def items(self) -> Iterable[Tuple[S, T_co]]: ...
    def keys(self) -> Iterable[S]: ...
    def values(self) -> Iterable[T_co]: ...
    def __contains__(self, other: S) -> bool: ...
    def __eq__(self, other: Mapping[S, T_co]) -> bool: ...
    def __getitem__(self, index: S) -> T_co: ...
    def __getstate__(self) -> Any: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterable[S]: ...
    def __len__(self) -> int: ...
    def __ne__(self, other: Mapping[S, T_co]) -> bool: ...
    def __setstate__(self, state) -> None: ...

class NodeIndices(RetworkxCustomVecIter[int]):
    pass

class PathLengthMapping(RetworkxCustomHashMapIter[int, float]):
    pass

class PathMapping(RetworkxCustomHashMapIter[int, NodeIndices]):
    pass

class AllPairsPathLengthMapping(RetworkxCustomHashMapIter[int, PathLengthMapping]):
    pass

class AllPairsPathMapping(RetworkxCustomHashMapIter[int, PathMapping]):
    pass

class BFSSuccessors(Generic[T_co], RetworkxCustomVecIter[Tuple[T_co, List[T_co]]]):
    pass

class EdgeIndexMap(Generic[T_co], RetworkxCustomHashMapIter[int, Tuple[int, int, T_co]]):
    pass

class EdgeIndices(RetworkxCustomVecIter[int]):
    pass

class Chains(RetworkxCustomVecIter[EdgeIndices]):
    pass

class EdgeList(RetworkxCustomVecIter[Tuple[int, int]]):
    pass

class NodeMap(RetworkxCustomHashMapIter[int, int]):
    pass

class NodesCountMapping(RetworkxCustomHashMapIter[int, int]):
    pass

class Pos2DMapping(RetworkxCustomHashMapIter[int, Tuple[float, float]]):
    pass

class WeightedEdgeList(Generic[T_co], RetworkxCustomVecIter[Tuple[int, int, T_co]]):
    pass
