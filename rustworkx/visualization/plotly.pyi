# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import typing
from collections.abc import Callable

from rustworkx.rustworkx import PyDiGraph, PyGraph

if typing.TYPE_CHECKING:
    from plotly.graph_objects import Figure

NodeT = typing.TypeVar("NodeT")
EdgeT = typing.TypeVar("EdgeT")

def plotly_draw(
    graph: PyDiGraph[NodeT, EdgeT] | PyGraph[NodeT, EdgeT],
    node_attr_fn: Callable[[NodeT], dict] | None = None,
    edge_attr_fn: Callable[[EdgeT], dict] | None = None,
    method: typing.Literal["twopi", "neato", "circo", "fdp", "sfdp", "dot", "spring"] | None = None,
    show_node_indices: bool = True,
    show_edge_indices: bool = False,
    graph_attr: dict[str, str] | None = None,
    spring_attr: dict | None = None,
) -> Figure: ...
