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

import unittest
import rustworkx


class TestSubstitute(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.graph = rustworkx.generators.directed_path_graph(5)

    def test_empty_replacement(self):
        in_graph = rustworkx.PyDiGraph()
        with self.assertRaises(IndexError):
            self.graph.substitute_subgraph([2], in_graph, {})

    def test_single_node(self):
        in_graph = rustworkx.PyDiGraph()
        in_graph.add_node(0)
        in_graph.add_child(0, 1, "edge")
        res = self.graph.substitute_subgraph([2], in_graph, {2: 0})
        self.assertEqual([(0, 1), (2, 5), (2, 3), (3, 4), (1, 2)], self.graph.edge_list())
        self.assertEqual("edge", self.graph.get_edge_data(2, 5))
        self.assertEqual(res, {0: 2, 1: 5})

    def test_edge_weight_modifier(self):
        in_graph = rustworkx.PyDiGraph()
        in_graph.add_node(0)
        in_graph.add_child(0, 1, "edge")
        res = self.graph.substitute_subgraph(
            [2],
            in_graph,
            {2: 0},
            edge_weight_map=lambda edge: edge + "-migrated",
        )

        self.assertEqual([(0, 1), (2, 5), (2, 3), (3, 4), (1, 2)], self.graph.edge_list())
        self.assertEqual("edge-migrated", self.graph.get_edge_data(2, 5))
        self.assertEqual(res, {0: 2, 1: 5})

    def test_multiple_mapping(self):
        graph = rustworkx.generators.directed_star_graph(5)
        in_graph = rustworkx.generators.directed_star_graph(3, inward=True)
        res = graph.substitute_subgraph([0, 1, 2], in_graph, {0: 0, 1: 1, 2: 2})
        self.assertEqual({0: 2, 1: 1, 2: 0}, res)
        expected = [(1, 2), (0, 2), (2, 4), (2, 3)]
        self.assertEqual(expected, graph.edge_list())
