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

import rustworkx as rx
import types
import unittest

from typing import Optional


class TestAnnotationSubscriptions(unittest.TestCase):
    def test_digraph(self):
        graph: rx.PyDiGraph[int, int] = rx.PyDiGraph()
        self.assertIsInstance(
            graph.__class_getitem__((int, int)),
            types.GenericAlias,
        )

    def test_graph(self):
        graph: rx.PyGraph[int, int] = rx.PyGraph()
        self.assertIsInstance(
            graph.__class_getitem__((int, int)),
            types.GenericAlias,
        )

    def test_dag(self):
        graph: rx.PyDAG[int, int] = rx.PyDAG()
        self.assertIsInstance(
            graph.__class_getitem__((int, int)),
            types.GenericAlias,
        )

    def test_custom_vector_allowed(self):
        graph: rx.PyGraph[Optional[int], Optional[int]] = rx.generators.path_graph(5)
        we_list: rx.WeightedEdgeList[Optional[int]] = graph.weighted_edge_list()
        self.assertIsInstance(
            we_list.__class_getitem__((Optional[int],)),
            types.GenericAlias,
        )

    def test_custom_vector_not_allowed(self):
        graph: rx.PyGraph[Optional[int], Optional[int]] = rx.generators.path_graph(5)
        edge_list: rx.EdgeList = graph.edge_list()
        with self.assertRaises(TypeError):
            self.assertIsInstance(
                edge_list.__class_getitem__((Optional[int],)),
                types.GenericAlias,
            )

    def test_custom_hashmap_allowed(self):
        graph: rx.PyGraph[Optional[int], Optional[int]] = rx.generators.path_graph(5)
        ei_map: rx.WeightedEdgeList[Optional[int]] = graph.edge_index_map()
        self.assertIsInstance(
            ei_map.__class_getitem__((Optional[int],)),
            types.GenericAlias,
        )

    def test_custom_vector_not_allowed(self):
        graph: rx.PyGraph[Optional[int], Optional[int]] = rx.generators.path_graph(5)
        all_pairs_pm: rx.AllPairsPathMapping = rx.all_pairs_dijkstra_shortest_paths(
            graph, lambda _: 1.0
        )
        with self.assertRaises(TypeError):
            self.assertIsInstance(
                all_pairs_pm.__class_getitem__((Optional[int],)),
                types.GenericAlias,
            )
