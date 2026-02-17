# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import typing
from collections.abc import Callable

from rustworkx.rustworkx import PyGraph, PyDiGraph

if typing.TYPE_CHECKING:
    from plotly.graph_objects import Figure

_S = typing.TypeVar("_S")
_T = typing.TypeVar("_T")

def plotly_draw(
    graph: PyDiGraph[_S, _T] | PyGraph[_S, _T],
    node_attr_fn: Callable[[_S], dict] | None = ...,
    edge_attr_fn: Callable[[_T], dict] | None = ...,
    graph_attr: dict[str, str] | None = ...,
    method: typing.Literal["twopi", "neato", "circo", "fdp", "sfdp", "dot", "spring"] | None = ...,
    show_node_indices: bool = ...,
    show_edge_indices: bool = ...,
    spring_attr: dict | None = ...,
) -> Figure: ...
