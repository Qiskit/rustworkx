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


class TestChainDecomposition(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
        self.graph.extend_from_edge_list(
            [
                (0, 1),
                (0, 2),
                (1, 2),
                (3, 4),
                (3, 5),
                (4, 5),
                (2, 3),
            ]
        )
        return super().setUp()

    def test_graph(self):
        edges = [
            # back edges
            (0, 2),
            (0, 3),
            (1, 4),
            (4, 9),
            (5, 7),
            # tree edges
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (4, 5),
            (4, 8),
            (5, 6),
            (6, 7),
            (8, 9),
        ]

        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list(edges)
        chains = rustworkx.chain_decomposition(graph, source=0)
        expected = [
            [(0, 3), (3, 2), (2, 1), (1, 0)],
            [(0, 2)],
            [(1, 4), (4, 2)],
            [(4, 9), (9, 8), (8, 4)],
            [(5, 7), (7, 6), (6, 5)],
        ]
        self.assertEqual(expected, chains)

    def test_barbell_graph(self):
        chains = rustworkx.chain_decomposition(self.graph, source=0)
        expected = [[(0, 1), (1, 2), (2, 0)], [(3, 4), (4, 5), (5, 3)]]
        self.assertEqual(expected, chains)

    def test_disconnected_graph(self):
        graph = rustworkx.union(self.graph, self.graph)
        chains = rustworkx.chain_decomposition(graph)
        expected = [
            [(0, 1), (1, 2), (2, 0)],
            [(3, 4), (4, 5), (5, 3)],
            [(6, 7), (7, 8), (8, 6)],
            [(9, 10), (10, 11), (11, 9)],
        ]
        self.assertEqual(expected, chains)

    def test_disconnected_graph_root_node(self):
        graph = rustworkx.union(self.graph, self.graph)
        chains = rustworkx.chain_decomposition(graph, source=0)
        expected = [
            [(0, 1), (1, 2), (2, 0)],
            [(3, 4), (4, 5), (5, 3)],
        ]
        self.assertEqual(expected, chains)
