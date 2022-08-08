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


class TestBinomialTreeGraph(unittest.TestCase):
    def test_binomial_tree_graph(self):
        expected_edges = {
            0: [],
            1: [(0, 1)],
            2: [(0, 1), (2, 3), (0, 2)],
            3: [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)],
            4: [
                (0, 1),
                (2, 3),
                (0, 2),
                (4, 5),
                (6, 7),
                (4, 6),
                (0, 4),
                (8, 9),
                (10, 11),
                (8, 10),
                (12, 13),
                (14, 15),
                (12, 14),
                (8, 12),
                (0, 8),
            ],
        }
        for n in range(5):
            with self.subTest(n=n):
                graph = rustworkx.generators.binomial_tree_graph(n)
                self.assertEqual(len(graph), 2**n)
                self.assertEqual(len(graph.edges()), 2**n - 1)
                self.assertEqual(list(graph.edge_list()), expected_edges[n])

    def test_binomial_tree_graph_weights(self):
        graph = rustworkx.generators.binomial_tree_graph(2, weights=list(range(4)))
        expected_edges = [(0, 1), (2, 3), (0, 2)]
        self.assertEqual(len(graph), 4)
        self.assertEqual([x for x in range(4)], graph.nodes())
        self.assertEqual(len(graph.edges()), 3)
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_binomial_tree_graph_weight_less_nodes(self):
        graph = rustworkx.generators.binomial_tree_graph(2, weights=list(range(2)))
        self.assertEqual(len(graph), 4)
        expected_weights = [x for x in range(2)]
        expected_weights.extend([None, None])
        self.assertEqual(expected_weights, graph.nodes())
        self.assertEqual(len(graph.edges()), 3)

    def test_binomial_tree_graph_weights_greater_nodes(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.binomial_tree_graph(2, weights=list(range(7)))

    def test_binomial_tree_no_order(self):
        with self.assertRaises(TypeError):
            rustworkx.generators.binomial_tree_graph(weights=list(range(4)))

    def test_directed_binomial_tree_graph(self):
        expected_edges = {
            0: [],
            1: [(0, 1)],
            2: [(0, 1), (2, 3), (0, 2)],
            3: [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)],
            4: [
                (0, 1),
                (2, 3),
                (0, 2),
                (4, 5),
                (6, 7),
                (4, 6),
                (0, 4),
                (8, 9),
                (10, 11),
                (8, 10),
                (12, 13),
                (14, 15),
                (12, 14),
                (8, 12),
                (0, 8),
            ],
        }

        for n in range(5):
            with self.subTest(n=n):
                graph = rustworkx.generators.directed_binomial_tree_graph(n)
                self.assertEqual(len(graph), 2**n)
                self.assertEqual(len(graph.edges()), 2**n - 1)
                self.assertEqual(list(graph.edge_list()), expected_edges[n])

    def test_directed_binomial_tree_graph_weights(self):
        graph = rustworkx.generators.directed_binomial_tree_graph(2, weights=list(range(4)))
        self.assertEqual(len(graph), 4)
        self.assertEqual([x for x in range(4)], graph.nodes())
        self.assertEqual(len(graph.edges()), 3)

    def test_directed_binomial_tree_graph_weight_less_nodes(self):
        graph = rustworkx.generators.directed_binomial_tree_graph(2, weights=list(range(2)))
        self.assertEqual(len(graph), 4)
        expected_weights = [x for x in range(2)]
        expected_weights.extend([None, None])
        self.assertEqual(expected_weights, graph.nodes())
        self.assertEqual(len(graph.edges()), 3)

    def test_directed_binomial_tree_graph_weights_greater_nodes(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.directed_binomial_tree_graph(2, weights=list(range(7)))

    def test_directed_binomial_tree_no_order(self):
        with self.assertRaises(TypeError):
            rustworkx.generators.directed_binomial_tree_graph(weights=list(range(4)))

    def test_directed_binomial_tree_graph_bidirectional(self):
        expected_edges = {
            0: [],
            1: [(0, 1), (1, 0)],
            2: [(0, 1), (1, 0), (2, 3), (3, 2), (0, 2), (2, 0)],
            3: [
                (0, 1),
                (1, 0),
                (2, 3),
                (3, 2),
                (0, 2),
                (2, 0),
                (4, 5),
                (5, 4),
                (6, 7),
                (7, 6),
                (4, 6),
                (6, 4),
                (0, 4),
                (4, 0),
            ],
            4: [
                (0, 1),
                (1, 0),
                (2, 3),
                (3, 2),
                (0, 2),
                (2, 0),
                (4, 5),
                (5, 4),
                (6, 7),
                (7, 6),
                (4, 6),
                (6, 4),
                (0, 4),
                (4, 0),
                (8, 9),
                (9, 8),
                (10, 11),
                (11, 10),
                (8, 10),
                (10, 8),
                (12, 13),
                (13, 12),
                (14, 15),
                (15, 14),
                (12, 14),
                (14, 12),
                (8, 12),
                (12, 8),
                (0, 8),
                (8, 0),
            ],
        }
        for n in range(5):
            with self.subTest(n=n):
                graph = rustworkx.generators.directed_binomial_tree_graph(n, bidirectional=True)
                self.assertEqual(len(graph), 2**n)
                self.assertEqual(len(graph.edges()), 2 * (2**n - 1))
                self.assertEqual(list(graph.edge_list()), expected_edges[n])

    def test_overflow_binomial_tree(self):
        with self.assertRaises(OverflowError):
            rustworkx.generators.binomial_tree_graph(75)

    def test_overflow_directed_binomial_tree(self):
        with self.assertRaises(OverflowError):
            rustworkx.generators.directed_binomial_tree_graph(75)
