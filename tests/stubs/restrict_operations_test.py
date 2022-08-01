# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from typing import Optional
from rustworkx import PyGraph, PyDiGraph

import pytest


@pytest.mark.mypy_testing
def test_pygraph_add_edges_from_no_data_illegal() -> None:
    graph: PyGraph[str, float] = PyGraph()
    node_a: int = graph.add_node("A")
    node_b: int = graph.add_node("B")

    # fmt: off
    graph.add_edges_from_no_data([(node_a, node_b)])  # E: Invalid self argument "PyGraph[str, float]" to attribute function "add_edges_from_no_data" with type "Callable[[PyGraph[S, Optional[T]], Sequence[Tuple[int, int]]], List[int]]"
    # fmt: on


@pytest.mark.mypy_testing
def test_pygraph_add_edges_from_no_data_legal() -> None:
    graph: PyGraph[str, Optional[float]] = PyGraph()
    node_a: int = graph.add_node("A")
    node_b: int = graph.add_node("B")

    graph.add_edges_from_no_data([(node_a, node_b)])


@pytest.mark.mypy_testing
def test_pygraph_extend_from_edge_list_illegal() -> None:
    graph: PyGraph[str, float] = PyGraph()

    # fmt: off
    graph.extend_from_edge_list([(0, 5)])  # E: Invalid self argument "PyGraph[str, float]" to attribute function "extend_from_edge_list" with type "Callable[[PyGraph[Optional[S], Optional[T]], Sequence[Tuple[int, int]]], None]"
    # fmt: on


@pytest.mark.mypy_testing
def test_pygraph_extend_from_edge_list_legal() -> None:
    graph: PyGraph[Optional[str], Optional[float]] = PyGraph()
    graph.extend_from_edge_list([(0, 5)])


@pytest.mark.mypy_testing
def test_pydigraph_add_edges_from_no_data_illegal() -> None:
    graph: PyDiGraph[str, float] = PyDiGraph()
    node_a: int = graph.add_node("A")
    node_b: int = graph.add_node("B")

    # fmt: off
    graph.add_edges_from_no_data([(node_a, node_b)])  # E: Invalid self argument "PyDiGraph[str, float]" to attribute function "add_edges_from_no_data" with type "Callable[[PyDiGraph[S, Optional[T]], Sequence[Tuple[int, int]]], List[int]]"
    # fmt: on


@pytest.mark.mypy_testing
def test_pydigraph_add_edges_from_no_data_legal() -> None:
    graph: PyDiGraph[str, Optional[float]] = PyDiGraph()
    node_a: int = graph.add_node("A")
    node_b: int = graph.add_node("B")

    graph.add_edges_from_no_data([(node_a, node_b)])


@pytest.mark.mypy_testing
def test_pydigraph_extend_from_edge_list_illegal() -> None:
    graph: PyDiGraph[str, float] = PyDiGraph()

    # fmt: off
    graph.extend_from_edge_list([(0, 5)])  # E: Invalid self argument "PyDiGraph[str, float]" to attribute function "extend_from_edge_list" with type "Callable[[PyDiGraph[Optional[S], Optional[T]], Sequence[Tuple[int, int]]], None]"
    # fmt: on


@pytest.mark.mypy_testing
def test_pydigraph_extend_from_edge_list_legal() -> None:
    graph: PyDiGraph[Optional[str], Optional[float]] = PyDiGraph()
    graph.extend_from_edge_list([(0, 5)])
