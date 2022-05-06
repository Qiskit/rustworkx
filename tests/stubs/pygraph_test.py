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

from typing import List, Tuple
from retworkx import EdgeList, PyGraph, NodeIndices, WeightedEdgeList

import pytest


@pytest.mark.mypy_testing
def test_pygraph_simple() -> None:
    graph: PyGraph[str, int] = PyGraph()
    node_a: int = graph.add_node("A")
    node_b: int = graph.add_node("B")
    edge_ab: int = graph.add_edge(node_a, node_b, 3)
    reveal_type(node_a)  # R: builtins.int
    reveal_type(edge_ab)  # R: builtins.int


@pytest.mark.mypy_testing
def test_custom_return_types() -> None:
    graph: PyGraph[str, int] = PyGraph()
    node_a: int = graph.add_node("A")
    node_b: int = graph.add_node("B")
    graph.add_edge(node_a, node_b, 3)

    edges: EdgeList = graph.edge_list()
    weighted_edges: WeightedEdgeList[int] = graph.weighted_edge_list()
    node_indices: NodeIndices = graph.node_indexes()

    # fmt: off
    reveal_type(edges)  # R: retworkx.custom_return_types.EdgeList
    reveal_type(weighted_edges)  # R: retworkx.custom_return_types.WeightedEdgeList[builtins.int]
    reveal_type(node_indices)  # R: retworkx.custom_return_types.NodeIndices
    # fmt: on

    list_of_edges: List[Tuple[int, int]] = list(edges)
    list_of_weights: List[Tuple[int, int, int]] = list(weighted_edges)
    list_of_nodes: List[int] = list(node_indices)

    # fmt: off
    reveal_type(list_of_edges)  # R: builtins.list[Tuple[builtins.int, builtins.int]]
    reveal_type(list_of_weights)  # R: builtins.list[Tuple[builtins.int, builtins.int, builtins.int]]
    reveal_type(list_of_nodes)  # R: builtins.list[builtins.int]
    # fmt: on
