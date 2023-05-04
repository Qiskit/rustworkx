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


class TestHeavyHexGraph(unittest.TestCase):
    def test_directed_heavy_hex_graph_1(self):
        d = 1
        graph = rustworkx.generators.directed_heavy_hex_graph(d)
        self.assertEqual(1, len(graph))
        self.assertEqual(graph.edge_list(), [])

    def test_heavy_hex_graph_1(self):
        d = 1
        graph = rustworkx.generators.heavy_hex_graph(d)
        self.assertEqual(1, len(graph))
        self.assertEqual(graph.edge_list(), [])

    def test_directed_heavy_hex_graph_3(self):
        d = 3
        graph = rustworkx.generators.directed_heavy_hex_graph(d)
        self.assertEqual(len(graph), (5 * d * d - 2 * d - 1) / 2)
        self.assertEqual(len(graph.edges()), 2 * d * (d - 1) + (d + 1) * (d - 1))
        expected_edges = [
            (0, 13),
            (1, 13),
            (1, 14),
            (2, 14),
            (3, 15),
            (4, 15),
            (4, 16),
            (5, 16),
            (6, 17),
            (7, 17),
            (7, 18),
            (8, 18),
            (0, 9),
            (3, 9),
            (5, 12),
            (8, 12),
            (10, 14),
            (10, 16),
            (11, 15),
            (11, 17),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_heavy_hex_graph_3_bidirectional(self):
        d = 3
        graph = rustworkx.generators.directed_heavy_hex_graph(d, bidirectional=True)
        self.assertEqual(len(graph), (5 * d * d - 2 * d - 1) / 2)
        self.assertEqual(len(graph.edges()), 2 * (2 * d * (d - 1) + (d + 1) * (d - 1)))
        expected_edges = [
            (0, 13),
            (1, 13),
            (13, 0),
            (13, 1),
            (1, 14),
            (2, 14),
            (14, 1),
            (14, 2),
            (3, 15),
            (4, 15),
            (15, 3),
            (15, 4),
            (4, 16),
            (5, 16),
            (16, 4),
            (16, 5),
            (6, 17),
            (7, 17),
            (17, 6),
            (17, 7),
            (7, 18),
            (8, 18),
            (18, 7),
            (18, 8),
            (0, 9),
            (3, 9),
            (9, 0),
            (9, 3),
            (5, 12),
            (8, 12),
            (12, 5),
            (12, 8),
            (10, 14),
            (10, 16),
            (14, 10),
            (16, 10),
            (11, 15),
            (11, 17),
            (15, 11),
            (17, 11),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_heavy_hex_graph_3(self):
        d = 3
        graph = rustworkx.generators.heavy_hex_graph(d)
        self.assertEqual(len(graph), (5 * d * d - 2 * d - 1) / 2)
        self.assertEqual(len(graph.edges()), 2 * d * (d - 1) + (d + 1) * (d - 1))
        expected_edges = [
            (0, 13),
            (1, 13),
            (1, 14),
            (2, 14),
            (3, 15),
            (4, 15),
            (4, 16),
            (5, 16),
            (6, 17),
            (7, 17),
            (7, 18),
            (8, 18),
            (0, 9),
            (3, 9),
            (5, 12),
            (8, 12),
            (10, 14),
            (10, 16),
            (11, 15),
            (11, 17),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_heavy_hex_graph_5(self):
        d = 5
        graph = rustworkx.generators.directed_heavy_hex_graph(d)
        self.assertEqual(len(graph), (5 * d * d - 2 * d - 1) / 2)
        self.assertEqual(len(graph.edges()), 2 * d * (d - 1) + (d + 1) * (d - 1))
        expected_edges = [
            (0, 37),
            (1, 37),
            (1, 38),
            (2, 38),
            (2, 39),
            (3, 39),
            (3, 40),
            (4, 40),
            (5, 41),
            (6, 41),
            (6, 42),
            (7, 42),
            (7, 43),
            (8, 43),
            (8, 44),
            (9, 44),
            (10, 45),
            (11, 45),
            (11, 46),
            (12, 46),
            (12, 47),
            (13, 47),
            (13, 48),
            (14, 48),
            (15, 49),
            (16, 49),
            (16, 50),
            (17, 50),
            (17, 51),
            (18, 51),
            (18, 52),
            (19, 52),
            (20, 53),
            (21, 53),
            (21, 54),
            (22, 54),
            (22, 55),
            (23, 55),
            (23, 56),
            (24, 56),
            (0, 25),
            (5, 25),
            (9, 30),
            (14, 30),
            (10, 31),
            (15, 31),
            (19, 36),
            (24, 36),
            (26, 38),
            (26, 42),
            (27, 40),
            (27, 44),
            (28, 41),
            (28, 45),
            (29, 43),
            (29, 47),
            (32, 46),
            (32, 50),
            (33, 48),
            (33, 52),
            (34, 49),
            (34, 53),
            (35, 51),
            (35, 55),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_heavy_hex_graph_5_bidirectional(self):
        d = 5
        graph = rustworkx.generators.directed_heavy_hex_graph(d, bidirectional=True)
        self.assertEqual(len(graph), (5 * d * d - 2 * d - 1) / 2)
        self.assertEqual(len(graph.edges()), 2 * (2 * d * (d - 1) + (d + 1) * (d - 1)))
        expected_edges = [
            (0, 37),
            (1, 37),
            (37, 0),
            (37, 1),
            (1, 38),
            (2, 38),
            (38, 1),
            (38, 2),
            (2, 39),
            (3, 39),
            (39, 2),
            (39, 3),
            (3, 40),
            (4, 40),
            (40, 3),
            (40, 4),
            (5, 41),
            (6, 41),
            (41, 5),
            (41, 6),
            (6, 42),
            (7, 42),
            (42, 6),
            (42, 7),
            (7, 43),
            (8, 43),
            (43, 7),
            (43, 8),
            (8, 44),
            (9, 44),
            (44, 8),
            (44, 9),
            (10, 45),
            (11, 45),
            (45, 10),
            (45, 11),
            (11, 46),
            (12, 46),
            (46, 11),
            (46, 12),
            (12, 47),
            (13, 47),
            (47, 12),
            (47, 13),
            (13, 48),
            (14, 48),
            (48, 13),
            (48, 14),
            (15, 49),
            (16, 49),
            (49, 15),
            (49, 16),
            (16, 50),
            (17, 50),
            (50, 16),
            (50, 17),
            (17, 51),
            (18, 51),
            (51, 17),
            (51, 18),
            (18, 52),
            (19, 52),
            (52, 18),
            (52, 19),
            (20, 53),
            (21, 53),
            (53, 20),
            (53, 21),
            (21, 54),
            (22, 54),
            (54, 21),
            (54, 22),
            (22, 55),
            (23, 55),
            (55, 22),
            (55, 23),
            (23, 56),
            (24, 56),
            (56, 23),
            (56, 24),
            (0, 25),
            (5, 25),
            (25, 0),
            (25, 5),
            (9, 30),
            (14, 30),
            (30, 9),
            (30, 14),
            (10, 31),
            (15, 31),
            (31, 10),
            (31, 15),
            (19, 36),
            (24, 36),
            (36, 19),
            (36, 24),
            (26, 38),
            (26, 42),
            (38, 26),
            (42, 26),
            (27, 40),
            (27, 44),
            (40, 27),
            (44, 27),
            (28, 41),
            (28, 45),
            (41, 28),
            (45, 28),
            (29, 43),
            (29, 47),
            (43, 29),
            (47, 29),
            (32, 46),
            (32, 50),
            (46, 32),
            (50, 32),
            (33, 48),
            (33, 52),
            (48, 33),
            (52, 33),
            (34, 49),
            (34, 53),
            (49, 34),
            (53, 34),
            (35, 51),
            (35, 55),
            (51, 35),
            (55, 35),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_heavy_hex_graph_5(self):
        d = 5
        graph = rustworkx.generators.heavy_hex_graph(d)
        self.assertEqual(len(graph), (5 * d * d - 2 * d - 1) / 2)
        self.assertEqual(len(graph.edges()), 2 * d * (d - 1) + (d + 1) * (d - 1))
        expected_edges = [
            (0, 37),
            (1, 37),
            (1, 38),
            (2, 38),
            (2, 39),
            (3, 39),
            (3, 40),
            (4, 40),
            (5, 41),
            (6, 41),
            (6, 42),
            (7, 42),
            (7, 43),
            (8, 43),
            (8, 44),
            (9, 44),
            (10, 45),
            (11, 45),
            (11, 46),
            (12, 46),
            (12, 47),
            (13, 47),
            (13, 48),
            (14, 48),
            (15, 49),
            (16, 49),
            (16, 50),
            (17, 50),
            (17, 51),
            (18, 51),
            (18, 52),
            (19, 52),
            (20, 53),
            (21, 53),
            (21, 54),
            (22, 54),
            (22, 55),
            (23, 55),
            (23, 56),
            (24, 56),
            (0, 25),
            (5, 25),
            (9, 30),
            (14, 30),
            (10, 31),
            (15, 31),
            (19, 36),
            (24, 36),
            (26, 38),
            (26, 42),
            (27, 40),
            (27, 44),
            (28, 41),
            (28, 45),
            (29, 43),
            (29, 47),
            (32, 46),
            (32, 50),
            (33, 48),
            (33, 52),
            (34, 49),
            (34, 53),
            (35, 51),
            (35, 55),
        ]

        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_heavy_hex_graph_even_d(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.heavy_hex_graph(2)
