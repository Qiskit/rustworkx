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

import pprint
import unittest

import rustworkx


class TestSteinerTree(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph(multigraph=False)
        self.graph.add_node(None)
        self.graph.extend_from_weighted_edge_list(
            [
                (1, 2, 10),
                (2, 3, 10),
                (3, 4, 10),
                (4, 5, 10),
                (5, 6, 10),
                (2, 7, 1),
                (7, 5, 1),
            ]
        )
        self.graph.remove_node(0)

    def test_metric_closure(self):
        closure_graph = rustworkx.metric_closure(self.graph, weight_fn=float)
        expected_edges = [
            (1, 2, (10.0, [1, 2])),
            (1, 3, (20.0, [1, 2, 3])),
            (1, 4, (22.0, [1, 2, 7, 5, 4])),
            (1, 5, (12.0, [1, 2, 7, 5])),
            (1, 6, (22.0, [1, 2, 7, 5, 6])),
            (1, 7, (11.0, [1, 2, 7])),
            (2, 3, (10.0, [2, 3])),
            (2, 4, (12.0, [2, 7, 5, 4])),
            (2, 5, (2.0, [2, 7, 5])),
            (2, 6, (12, [2, 7, 5, 6])),
            (2, 7, (1.0, [2, 7])),
            (3, 4, (10.0, [3, 4])),
            (3, 5, (12.0, [3, 2, 7, 5])),
            (3, 6, (22.0, [3, 2, 7, 5, 6])),
            (3, 7, (11.0, [3, 2, 7])),
            (4, 5, (10.0, [4, 5])),
            (4, 6, (20.0, [4, 5, 6])),
            (4, 7, (11.0, [4, 5, 7])),
            (5, 6, (10.0, [5, 6])),
            (5, 7, (1.0, [5, 7])),
            (6, 7, (11.0, [6, 5, 7])),
        ]
        edges = list(closure_graph.weighted_edge_list())
        for edge in expected_edges:
            found = False
            if edge in edges:
                found = True
            if not found:

                if (
                    edge[1],
                    edge[0],
                    (edge[2][0], list(reversed(edge[2][1]))),
                ) in edges:
                    found = True
            if not found:
                self.fail(
                    f"edge: {edge} nor it's reverse not found in metric "
                    f"closure output:\n{pprint.pformat(edges)}"
                )

    def test_not_connected_metric_closure(self):
        self.graph.add_node(None)
        with self.assertRaises(ValueError):
            rustworkx.metric_closure(self.graph, weight_fn=float)

    def test_partially_connected_metric_closure(self):
        graph = rustworkx.PyGraph()
        graph.add_node(None)
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 10),
                (2, 3, 10),
                (3, 4, 10),
                (4, 5, 10),
                (5, 6, 10),
                (2, 7, 1),
                (7, 5, 1),
            ]
        )
        graph.extend_from_weighted_edge_list(
            [
                (0, 8, 20),
                (0, 9, 20),
                (0, 10, 20),
                (8, 10, 10),
                (9, 10, 5),
            ]
        )
        with self.assertRaises(ValueError):
            rustworkx.metric_closure(graph, weight_fn=float)

    def test_metric_closure_empty_graph(self):
        graph = rustworkx.PyGraph()
        closure = rustworkx.metric_closure(graph, weight_fn=float)
        self.assertEqual([], closure.weighted_edge_list())

    def test_steiner_graph(self):
        steiner_tree = rustworkx.steiner_tree(self.graph, [1, 2, 3, 4, 5], weight_fn=float)
        expected_steiner_tree = [
            (1, 2, 10),
            (2, 3, 10),
            (2, 7, 1),
            (3, 4, 10),
            (7, 5, 1),
        ]
        steiner_tree_edge_list = steiner_tree.weighted_edge_list()
        for edge in expected_steiner_tree:
            self.assertIn(edge, steiner_tree_edge_list)

    def test_steiner_graph_multigraph(self):
        edge_list = [
            (1, 2, 1),
            (2, 3, 999),
            (2, 3, 1),
            (3, 4, 1),
            (3, 5, 1),
        ]
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list(edge_list)
        graph.remove_node(0)
        terminal_nodes = [2, 4, 5]
        tree = rustworkx.steiner_tree(graph, terminal_nodes, weight_fn=float)
        expected_edges = [
            (2, 3, 1),
            (3, 4, 1),
            (3, 5, 1),
        ]
        steiner_tree_edge_list = tree.weighted_edge_list()
        for edge in expected_edges:
            self.assertIn(edge, steiner_tree_edge_list)

    def test_not_connected_steiner_tree(self):
        self.graph.add_node(None)
        with self.assertRaises(ValueError):
            rustworkx.steiner_tree(self.graph, [1, 2, 0], weight_fn=float)

    def test_steiner_tree_empty_graph(self):
        graph = rustworkx.PyGraph()
        tree = rustworkx.steiner_tree(graph, [], weight_fn=float)
        self.assertEqual([], tree.weighted_edge_list())

    def test_equal_distance_graph(self):
        n = 3
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(range(n + 5))
        graph.add_edges_from(
            [
                (n, n + 1, 0.5),
                (n, n + 2, 0.5),
                (n + 1, n + 2, 0.5),
                (n, n + 3, 0.5),
                (n + 1, n + 4, 0.5),
            ]
        )
        graph.add_edges_from([(i, n + 2, 2) for i in range(n)])
        terminals = list(range(5)) + [n + 3, n + 4]
        tree = rustworkx.steiner_tree(graph, terminals, weight_fn=float)
        # Assert no cycle
        self.assertEqual(rustworkx.cycle_basis(tree), [])
        expected_edges = [
            (3, 4, 0.5),
            (4, 5, 0.5),
            (3, 6, 0.5),
            (4, 7, 0.5),
            (0, 5, 2),
            (1, 5, 2),
            (2, 5, 2),
        ]
        self.assertEqual(tree.weighted_edge_list(), expected_edges)
