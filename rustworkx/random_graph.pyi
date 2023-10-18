# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/lib.rs

from .graph import PyGraph
from .digraph import PyDiGraph

def directed_gnm_random_graph(
    num_nodes: int,
    num_edges: int,
    /,
    seed: int | None = ...,
) -> PyDiGraph: ...
def undirected_gnm_random_graph(
    num_nodes: int,
    num_edges: int,
    /,
    seed: int | None = ...,
) -> PyGraph: ...
def directed_gnp_random_graph(
    num_nodes: int,
    probability: float,
    /,
    seed: int | None = ...,
) -> PyDiGraph: ...
def undirected_gnp_random_graph(
    num_nodes: int,
    probability: float,
    /,
    seed: int | None = ...,
) -> PyGraph: ...
def random_geometric_graph(
    num_nodes: int,
    radius: float,
    /,
    dim: int = ...,
    pos: list[list[float]] | None = ...,
    p: float = ...,
    seed: int | None = ...,
) -> PyGraph: ...
