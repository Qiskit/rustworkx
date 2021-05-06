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
import retworkx


class TestHexagonalLatticeGraph(unittest.TestCase):

    def test_directed_hexagonal_graph_2_2(self):
        graph = retworkx.generators.directed_hexagonal_lattice_graph(2, 2)
        expected_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8),
                          (8, 9), (9, 10), (10, 11), (13, 14), (14, 15),
                          (15, 16), (16, 17), (0, 6), (2, 8), (4, 10),
                          (7, 13), (9, 15), (11, 17)]
        self.assertEqual(len(graph), 16)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_3_2(self):
        graph = retworkx.generators.directed_hexagonal_lattice_graph(3, 2)
        expected_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                          (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
                          (13, 14), (14, 15), (17, 18), (18, 19), (19, 20),
                          (20, 21), (21, 22), (22, 23), (0, 8), (2, 10),
                          (4, 12), (6, 14), (9, 17), (11, 19), (13, 21),
                          (15, 23)]
        self.assertEqual(len(graph), 22)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_2_4(self):
        graph = retworkx.generators.directed_hexagonal_lattice_graph(2, 4)
        expected_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8),
                          (8, 9), (9, 10), (10, 11), (12, 13), (13, 14),
                          (14, 15), (15, 16), (16, 17), (18, 19), (19, 20),
                          (20, 21), (21, 22), (22, 23), (25, 26), (26, 27),
                          (27, 28), (28, 29), (0, 6), (2, 8), (4, 10), (7, 13),
                          (9, 15), (11, 17), (12, 18), (14, 20), (16, 22),
                          (19, 25), (21, 27), (23, 29)]
        self.assertEqual(len(graph), 28)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_0_0(self):
        graph = retworkx.generators.directed_hexagonal_lattice_graph(0, 0)
        self.assertEqual(len(graph), 0)
        self.assertEqual(len(graph.edges()), 0)

    def test_directed_hexagonal_graph_0_0_bidirectional(self):
        graph = retworkx.generators.directed_hexagonal_lattice_graph(
            0, 0, bidirectional=True)
        self.assertEqual(len(graph), 0)
        self.assertEqual(len(graph.edges()), 0)

    def test_directed_hexagonal_graph_2_2_bidirectional(self):
        graph = retworkx.generators.directed_hexagonal_lattice_graph(
            2, 2,
            bidirectional=True)
        expected_edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2),
                          (3, 4), (4, 3), (6, 7), (7, 6), (7, 8), (8, 7),
                          (8, 9), (9, 8), (9, 10), (10, 9), (10, 11), (11, 10),
                          (13, 14), (14, 13), (14, 15), (15, 14), (15, 16),
                          (16, 15), (16, 17), (17, 16), (0, 6), (6, 0), (2, 8),
                          (8, 2), (4, 10), (10, 4), (7, 13), (13, 7), (9, 15),
                          (15, 9), (11, 17), (17, 11)]
        self.assertEqual(len(graph), 16)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_3_2_bidirectional(self):
        graph = retworkx.generators.directed_hexagonal_lattice_graph(
            3, 2,
            bidirectional=True)
        expected_edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2),
                          (3, 4), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5),
                          (8, 9), (9, 8), (9, 10), (10, 9), (10, 11), (11, 10),
                          (11, 12), (12, 11), (12, 13), (13, 12), (13, 14),
                          (14, 13), (14, 15), (15, 14), (17, 18), (18, 17),
                          (18, 19), (19, 18), (19, 20), (20, 19), (20, 21),
                          (21, 20), (21, 22), (22, 21), (22, 23), (23, 22),
                          (0, 8), (8, 0), (2, 10), (10, 2), (4, 12), (12, 4),
                          (6, 14), (14, 6), (9, 17), (17, 9), (11, 19),
                          (19, 11), (13, 21), (21, 13), (15, 23), (23, 15)]
        self.assertEqual(len(graph), 22)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_2_4_bidirectional(self):
        graph = retworkx.generators.directed_hexagonal_lattice_graph(
            2, 4,
            bidirectional=True)
        expected_edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2),
                          (3, 4), (4, 3), (6, 7), (7, 6), (7, 8), (8, 7),
                          (8, 9), (9, 8), (9, 10), (10, 9), (10, 11), (11, 10),
                          (12, 13), (13, 12), (13, 14), (14, 13), (14, 15),
                          (15, 14), (15, 16), (16, 15), (16, 17), (17, 16),
                          (18, 19), (19, 18), (19, 20), (20, 19), (20, 21),
                          (21, 20), (21, 22), (22, 21), (22, 23), (23, 22),
                          (25, 26), (26, 25), (26, 27), (27, 26), (27, 28),
                          (28, 27), (28, 29), (29, 28), (0, 6), (6, 0),
                          (2, 8), (8, 2), (4, 10), (10, 4), (7, 13), (13, 7),
                          (9, 15), (15, 9), (11, 17), (17, 11), (12, 18),
                          (18, 12), (14, 20), (20, 14), (16, 22), (22, 16),
                          (19, 25), (25, 19), (21, 27), (27, 21), (23, 29),
                          (29, 23)]
        self.assertEqual(len(graph), 28)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_hexagonal_graph_2_2(self):
        graph = retworkx.generators.hexagonal_lattice_graph(2, 2)
        expected_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8),
                          (8, 9), (9, 10), (10, 11), (13, 14), (14, 15),
                          (15, 16), (16, 17), (0, 6), (2, 8), (4, 10),
                          (7, 13), (9, 15), (11, 17)]
        self.assertEqual(len(graph), 16)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_hexagonal_graph_3_2(self):
        graph = retworkx.generators.hexagonal_lattice_graph(3, 2)
        expected_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                          (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
                          (13, 14), (14, 15), (17, 18), (18, 19), (19, 20),
                          (20, 21), (21, 22), (22, 23), (0, 8), (2, 10),
                          (4, 12), (6, 14), (9, 17), (11, 19), (13, 21),
                          (15, 23)]
        self.assertEqual(len(graph), 22)
        self.assertEqual(len(graph.edges()), 27)
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_hexagonal_graph_2_4(self):
        graph = retworkx.generators.hexagonal_lattice_graph(2, 4)
        expected_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8),
                          (8, 9), (9, 10), (10, 11), (12, 13), (13, 14),
                          (14, 15), (15, 16), (16, 17), (18, 19), (19, 20),
                          (20, 21), (21, 22), (22, 23), (25, 26), (26, 27),
                          (27, 28), (28, 29), (0, 6), (2, 8), (4, 10), (7, 13),
                          (9, 15), (11, 17), (12, 18), (14, 20), (16, 22),
                          (19, 25), (21, 27), (23, 29)]
        self.assertEqual(len(graph), 28)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_hexagonal_graph_0_0(self):
        graph = retworkx.generators.hexagonal_lattice_graph(0, 0)
        self.assertEqual(len(graph), 0)
        self.assertEqual(len(graph.edges()), 0)
