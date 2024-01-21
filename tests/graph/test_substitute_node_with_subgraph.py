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


class TestSubstituteNodeSubGraph(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.graph = rustworkx.generators.path_graph(5)

    def test_empty_replacement(self):
        in_graph = rustworkx.PyGraph()
        res = self.graph.substitute_node_with_subgraph(3, in_graph, lambda _, __, ___: None)
        self.assertEqual(res, {})
        self.assertEqual([(0, 1), (1, 2)], self.graph.edge_list())

    def test_single_node(self):
        in_graph = rustworkx.generators.path_graph(1)
        res = self.graph.substitute_node_with_subgraph(2, in_graph, lambda _, __, ___: 0)
        self.assertEqual(res, {0: 5})
        self.assertEqual([(0, 1), (1, 5), (3, 4), (5, 3)], sorted(self.graph.edge_list()))

    def test_node_filter(self):
        in_graph = rustworkx.generators.complete_graph(5)
        res = self.graph.substitute_node_with_subgraph(
            0, in_graph, lambda _, __, ___: 2, node_filter=lambda node: node is None
        )
        self.assertEqual(res, {i: i + 5 for i in range(5)})
        self.assertEqual(
            [
                (1, 2),
                (2, 3),
                (3, 4),
                (5, 6),
                (5, 7),
                (5, 8),
                (5, 9),
                (6, 7),
                (6, 8),
                (6, 9),
                (7, 1),
                (7, 8),
                (7, 9),
                (8, 9),
            ],
            sorted(self.graph.edge_list()),
        )

    def test_edge_weight_modifier(self):
        in_graph = rustworkx.PyGraph()
        in_graph.add_node("meep")
        in_graph.add_node("moop")
        in_graph.add_edges_from(
            [
                (
                    0,
                    1,
                    "edge",
                )
            ]
        )
        res = self.graph.substitute_node_with_subgraph(
            2,
            in_graph,
            lambda _, __, ___: 0,
            edge_weight_map=lambda edge: edge + "-migrated",
        )
        self.assertEqual([(0, 1), (3, 4), (5, 6), (1, 5), (5, 3)], self.graph.edge_list())
        self.assertEqual("edge-migrated", self.graph.get_edge_data(5, 6))
        self.assertEqual(res, {0: 5, 1: 6})

    def test_none_mapping(self):
        in_graph = rustworkx.PyGraph()
        in_graph.add_node("boop")
        in_graph.add_node("beep")
        in_graph.add_edges_from([(0, 1, "edge")])
        res = self.graph.substitute_node_with_subgraph(2, in_graph, lambda _, __, ___: None)
        self.assertEqual([(0, 1), (3, 4), (5, 6)], self.graph.edge_list())
        self.assertEqual(res, {0: 5, 1: 6})

    def test_multiple_mapping(self):
        graph = rustworkx.generators.star_graph(5)
        in_graph = rustworkx.generators.star_graph(3)

        def map_function(_source, target, _weight):
            if target > 2:
                return 2
            return 1

        res = graph.substitute_node_with_subgraph(0, in_graph, map_function)
        self.assertEqual({0: 5, 1: 6, 2: 7}, res)
        expected = [(5, 6), (5, 7), (7, 4), (7, 3), (6, 2), (6, 1)]
        self.assertEqual(sorted(expected), sorted(graph.edge_list()))

    def test_multiple_mapping_full(self):
        graph = rustworkx.generators.star_graph(5)
        in_graph = rustworkx.generators.star_graph(weights=list(range(3)))
        in_graph.add_edge(1, 2, None)

        def map_function(source, target, _weight):
            if target > 2:
                return 2
            return 1

        def filter_fn(node):
            return node > 0

        def map_weight(_):
            return "migrated"

        res = graph.substitute_node_with_subgraph(0, in_graph, map_function, filter_fn, map_weight)
        self.assertEqual({1: 5, 2: 6}, res)
        expected = [
            (5, 6, "migrated"),
            (6, 4, None),
            (6, 3, None),
            (5, 2, None),
            (5, 1, None),
        ]
        self.assertEqual(expected, graph.weighted_edge_list())

    def test_invalid_target(self):
        in_graph = rustworkx.generators.grid_graph(5, 5)
        with self.assertRaises(IndexError):
            self.graph.substitute_node_with_subgraph(0, in_graph, lambda *args: 42)

    def test_invalid_node_id(self):
        in_graph = rustworkx.generators.grid_graph(5, 5)
        with self.assertRaises(IndexError):
            self.graph.substitute_node_with_subgraph(16, in_graph, lambda *args: None)
