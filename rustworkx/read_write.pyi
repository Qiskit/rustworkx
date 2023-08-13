# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/json.rs, and src/graphml.rs
from .graph import PyGraph
from .digraph import PyDiGraph

from typing import Optional, List, Dict, TypeVar, Union, Callable, Any

_S = TypeVar("_S")
_T = TypeVar("_T")

def read_graphml(path: str, /) -> List[Union[PyGraph, PyDiGraph]]: ...
def digraph_node_link_json(
    graph: PyDiGraph[_S, _T],
    /,
    path: Optional[str] = ...,
    graph_attrs: Optional[Callable[[Any], Dict[str, str]]] = ...,
    node_attrs: Optional[Callable[[_S], str]] = ...,
    edge_attrs: Optional[Callable[[_T], str]] = ...,
) -> Optional[str]: ...
def graph_node_link_json(
    graph: PyGraph[_S, _T],
    /,
    path: Optional[str] = ...,
    graph_attrs: Optional[Callable[[Any], Dict[str, str]]] = ...,
    node_attrs: Optional[Callable[[_S], str]] = ...,
    edge_attrs: Optional[Callable[[_T], str]] = ...,
) -> Optional[str]: ...
