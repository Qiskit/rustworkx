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


class TestFullRaryTreeTreeGraph(unittest.TestCase):
    def test_full_rary_tree_graph(self):
        b_factors = {
            0: 0,
            1: 2,
            2: 2,
            3: 5,
        }
        num_nodes = {
            0: 0,
            1: 4,
            2: 10,
            3: 15,
        }
        expected_edges = {
            0: [],
            1: [(0, 1), (0, 2), (1, 3)],
            2: [
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 4),
                (2, 5),
                (2, 6),
                (3, 7),
                (3, 8),
                (4, 9),
            ],
            3: [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 6),
                (1, 7),
                (1, 8),
                (1, 9),
                (1, 10),
                (2, 11),
                (2, 12),
                (2, 13),
                (2, 14),
            ],
        }
        for n in range(4):
            with self.subTest(n=n):
                graph = rustworkx.generators.full_rary_tree(b_factors[n], num_nodes[n])
                self.assertEqual(list(graph.edge_list()), expected_edges[n])

    def test_full_rary_tree_graph_weights(self):
        graph = rustworkx.generators.full_rary_tree(2, 4, weights=list(range(4)))
        expected_edges = [(0, 1), (0, 2), (1, 3)]
        self.assertEqual(len(graph), 4)
        self.assertEqual([x for x in range(4)], graph.nodes())
        self.assertEqual(len(graph.edges()), 3)
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_full_rary_tree_graph_weight_less_nodes(self):
        graph = rustworkx.generators.full_rary_tree(2, 6, weights=list(range(4)))
        self.assertEqual(len(graph), 6)
        expected_weights = [x for x in range(4)]
        expected_weights.extend([None, None])
        self.assertEqual(expected_weights, graph.nodes())
        self.assertEqual(len(graph.edges()), 5)

    def test_full_rary_tree_graph_weights_greater_nodes(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.full_rary_tree(2, 4, weights=list(range(7)))

    def test_full_rary_tree_no_order(self):
        with self.assertRaises(TypeError):
            rustworkx.generators.full_rary_tree(weights=list(range(4)))
