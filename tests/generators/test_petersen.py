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


class TestPetersenGraph(unittest.TestCase):
    def test_petersen_graph_count(self):
        n = 99
        k = 23
        graph = rustworkx.generators.generalized_petersen_graph(n, k)
        self.assertEqual(len(graph), 2 * n)
        self.assertEqual(len(graph.edges()), 3 * n)

    def test_petersen_graph_edge(self):
        graph = rustworkx.generators.generalized_petersen_graph(5, 2)
        edge_list = graph.edge_list()
        expected_edge_list = [
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 0),
            (4, 1),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 5),
            (5, 0),
            (6, 1),
            (7, 2),
            (8, 3),
            (9, 4),
        ]
        self.assertEqual(edge_list, expected_edge_list)

    def test_petersen_invalid_n_k(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.generalized_petersen_graph(2, 1)

        with self.assertRaises(IndexError):
            rustworkx.generators.generalized_petersen_graph(5, 0)

        with self.assertRaises(IndexError):
            rustworkx.generators.generalized_petersen_graph(5, 4)
