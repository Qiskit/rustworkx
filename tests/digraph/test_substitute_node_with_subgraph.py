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
        res = self.graph.substitute_node_with_subgraph(2, in_graph, lambda _, __, ___: None)
        self.assertEqual(res, {})
        self.assertEqual([(0, 1), (3, 4)], self.graph.edge_list())

    def test_single_node(self):
        in_graph = rustworkx.PyDiGraph()
        in_graph.add_node(0)
        in_graph.add_child(0, 1, "edge")
        res = self.graph.substitute_node_with_subgraph(2, in_graph, lambda _, __, ___: 0)
        self.assertEqual([(0, 1), (3, 4), (5, 6), (1, 5), (5, 3)], self.graph.edge_list())
        self.assertEqual("edge", self.graph.get_edge_data(5, 6))
        self.assertEqual(res, {0: 5, 1: 6})

    def test_node_filter(self):
        in_graph = rustworkx.PyDiGraph()
        in_graph.add_node(0)
        in_graph.add_child(0, 1, "edge")
        res = self.graph.substitute_node_with_subgraph(
            2,
            in_graph,
            lambda _, __, ___: 0,
            node_filter=lambda node: node == 0,
        )
        self.assertEqual([(0, 1), (3, 4), (1, 5), (5, 3)], self.graph.edge_list())
        self.assertEqual(res, {0: 5})

    def test_edge_weight_modifier(self):
        in_graph = rustworkx.PyDiGraph()
        in_graph.add_node(0)
        in_graph.add_child(0, 1, "edge")
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
        in_graph = rustworkx.PyDiGraph()
        in_graph.add_node(0)
        in_graph.add_child(0, 1, "edge")
        res = self.graph.substitute_node_with_subgraph(2, in_graph, lambda _, __, ___: None)
        self.assertEqual([(0, 1), (3, 4), (5, 6)], self.graph.edge_list())
        self.assertEqual(res, {0: 5, 1: 6})

    def test_multiple_mapping(self):
        graph = rustworkx.generators.directed_star_graph(5)
        in_graph = rustworkx.generators.directed_star_graph(3, inward=True)

        def map_function(source, target, _weight):
            if target > 2:
                return 2
            return 1

        res = graph.substitute_node_with_subgraph(0, in_graph, map_function)
        self.assertEqual({0: 5, 1: 6, 2: 7}, res)
        expected = [(6, 5), (7, 5), (7, 4), (7, 3), (6, 2), (6, 1)]
        self.assertEqual(expected, graph.edge_list())

    def test_multiple_mapping_full(self):
        graph = rustworkx.generators.directed_star_graph(5)
        in_graph = rustworkx.generators.directed_star_graph(weights=list(range(3)), inward=True)
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
        in_graph = rustworkx.generators.directed_grid_graph(5, 5)
        with self.assertRaises(IndexError):
            self.graph.substitute_node_with_subgraph(0, in_graph, lambda *args: 42)

    def test_invalid_target_both_directions(self):
        graph = rustworkx.generators.directed_star_graph(4, inward=True)
        in_graph = rustworkx.generators.directed_grid_graph(5, 5)
        with self.assertRaises(IndexError):
            graph.substitute_node_with_subgraph(0, in_graph, lambda *args: 42)
        graph = rustworkx.generators.directed_star_graph(4, inward=False)
        with self.assertRaises(IndexError):
            graph.substitute_node_with_subgraph(0, in_graph, lambda *args: 42)

    def test_invalid_node_id(self):
        in_graph = rustworkx.generators.directed_grid_graph(5, 5)
        with self.assertRaises(IndexError):
            self.graph.substitute_node_with_subgraph(16, in_graph, lambda *args: None)

    def test_bidirectional(self):
        graph = rustworkx.generators.directed_path_graph(5, bidirectional=True)
        in_graph = rustworkx.generators.directed_star_graph(5, bidirectional=True)

        def map_function(source, target, _weight):
            if source != 2:
                return 0
            else:
                return target

        res = graph.substitute_node_with_subgraph(2, in_graph, map_function)
        expected_node_map = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9}
        self.assertEqual(expected_node_map, res)
        expected_edge_list = [
            (0, 1),  # From graph
            (1, 0),  # From graph
            (3, 4),  # From graph
            (4, 3),  # From graph
            (6, 5),  # From in_graph
            (5, 6),  # From in_graph
            (7, 5),  # From in_graph
            (5, 7),  # From in_graph
            (8, 5),  # From in_graph
            (5, 8),  # From in_graph
            (9, 5),  # From in_graph
            (5, 9),  # From in_graph
            (3, 5),  # output of res[map_function(3, 2, None)] -> 5
            (1, 5),  # output of res[map_function(1, 2, None)] -> 5
            (8, 3),  # output of res[map_function(2, 3, None)] -> 8
            (6, 1),  # output of res[map_function(2, 1, None)] -> 6
        ]
        self.assertEqual(expected_edge_list, graph.edge_list())
