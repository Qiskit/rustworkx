# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and lib.rs

from typing import Any, Generic, List, Iterable, TypeVar, Tuple
from collections.abc import Sequence

T = TypeVar("T")

class NodeIndices(Sequence[int]):
    def __init__(self) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> int: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class PathLengthMapping:
    def __init__(self) -> None: ...
    def items(self) -> Iterable[Tuple[int, float]]: ...
    def keys(self) -> Iterable[int]: ...
    def values(self) -> Iterable[float]: ...
    def __contains__(self, other) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> float: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterable[int]: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class PathMapping:
    def __init__(self) -> None: ...
    def items(self) -> Iterable[Tuple[int, NodeIndices]]: ...
    def keys(self) -> Iterable[int]: ...
    def values(self) -> Iterable[NodeIndices]: ...
    def __contains__(self, other) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> NodeIndices: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterable[int]: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class AllPairsPathLengthMapping:
    def __init__(self) -> None: ...
    def items(self) -> Iterable[Tuple[int, PathLengthMapping]]: ...
    def keys(self) -> Iterable[int]: ...
    def values(self) -> Iterable[PathLengthMapping]: ...
    def __contains__(self, other) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> PathLengthMapping: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterable[int]: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class AllPairsPathMapping:
    def __init__(self) -> None: ...
    def items(self) -> Iterable[Tuple[int, PathMapping]]: ...
    def keys(self) -> Iterable[int]: ...
    def values(self) -> Iterable[PathMapping]: ...
    def __contains__(self, other) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> PathMapping: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterable[int]: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class BFSSuccessors(Generic[T]):
    def __init__(self) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> Tuple[T, List[T]]: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class EdgeIndexMap(Generic[T]):
    def __init__(self) -> None: ...
    def items(self) -> Iterable[Tuple[int, Tuple[int, int, T]]]: ...
    def keys(self) -> Iterable[int]: ...
    def values(self) -> Iterable[Tuple[int, int, T]]: ...
    def __contains__(self, other) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> Tuple[int, int, T]: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterable[int]: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class EdgeIndices:
    def __init__(self) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> int: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class EdgeList(Sequence[Tuple[int, int]]):
    def __init__(self) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> Tuple[int, int]: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class NodeMap:
    def __init__(self) -> None: ...
    def items(self) -> Iterable[Tuple[int, int]]: ...
    def keys(self) -> Iterable[int]: ...
    def values(self) -> Iterable[int]: ...
    def __contains__(self, other) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> int: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterable[int]: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class NodesCountMapping:
    def __init__(self) -> None: ...
    def items(self) -> Iterable[Tuple[int, int]]: ...
    def keys(self) -> Iterable[int]: ...
    def values(self) -> Iterable[int]: ...
    def __contains__(self, other) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> int: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterable[int]: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class Pos2DMapping:
    def __init__(self) -> None: ...
    def items(self) -> Iterable[Tuple[int, Tuple[float, float]]]: ...
    def keys(self) -> Iterable[int]: ...
    def values(self) -> Iterable[Tuple[float, float]]: ...
    def __contains__(self, other) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> Tuple[float, float]: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterable[int]: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...

class WeightedEdgeList(Generic[T], Sequence[Tuple[int, int, T]]):
    def __init__(self) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __getitem__(self, index: int) -> Tuple[int, int, T]: ...
    def __getstate__(self) -> Any: ...
    def __gt__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    def __le__(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __setstate__(self, state) -> None: ...
