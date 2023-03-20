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
        graph = rustworkx.generators.directed_heavy_square_graph(d)
        self.assertEqual(1, len(graph))
        self.assertEqual(graph.edge_list(), [])

    def test_heavy_hex_graph_1(self):
        d = 1
        graph = rustworkx.generators.heavy_square_graph(d)
        self.assertEqual(1, len(graph))
        self.assertEqual(graph.edge_list(), [])

    def test_directed_heavy_square_graph_5(self):
        d = 5
        graph = rustworkx.generators.directed_heavy_square_graph(d)
        self.assertEqual(len(graph), 3 * d * d - 2 * d)
        self.assertEqual(len(graph.edges()), 2 * d * (d - 1) + 2 * d * (d - 1))
        expected_edges = [
            (0, 45),
            (45, 1),
            (1, 46),
            (46, 2),
            (2, 47),
            (47, 3),
            (3, 48),
            (48, 4),
            (5, 49),
            (49, 6),
            (6, 50),
            (50, 7),
            (7, 51),
            (51, 8),
            (8, 52),
            (52, 9),
            (10, 53),
            (53, 11),
            (11, 54),
            (54, 12),
            (12, 55),
            (55, 13),
            (13, 56),
            (56, 14),
            (15, 57),
            (57, 16),
            (16, 58),
            (58, 17),
            (17, 59),
            (59, 18),
            (18, 60),
            (60, 19),
            (20, 61),
            (61, 21),
            (21, 62),
            (62, 22),
            (22, 63),
            (63, 23),
            (23, 64),
            (64, 24),
            (4, 29),
            (9, 29),
            (5, 30),
            (10, 30),
            (14, 39),
            (19, 39),
            (15, 40),
            (20, 40),
            (25, 45),
            (25, 49),
            (26, 46),
            (26, 50),
            (27, 47),
            (27, 51),
            (28, 48),
            (28, 52),
            (31, 49),
            (31, 53),
            (32, 50),
            (32, 54),
            (33, 51),
            (33, 55),
            (34, 52),
            (34, 56),
            (35, 53),
            (35, 57),
            (36, 54),
            (36, 58),
            (37, 55),
            (37, 59),
            (38, 56),
            (38, 60),
            (41, 57),
            (41, 61),
            (42, 58),
            (42, 62),
            (43, 59),
            (43, 63),
            (44, 60),
            (44, 64),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_heavy_square_graph_5_bidirectional(self):
        d = 5
        graph = rustworkx.generators.directed_heavy_square_graph(d, bidirectional=True)
        self.assertEqual(len(graph), 3 * d * d - 2 * d)
        self.assertEqual(len(graph.edges()), 2 * (2 * d * (d - 1) + 2 * d * (d - 1)))
        expected_edges = [
            (0, 45),
            (45, 1),
            (45, 0),
            (1, 45),
            (1, 46),
            (46, 2),
            (46, 1),
            (2, 46),
            (2, 47),
            (47, 3),
            (47, 2),
            (3, 47),
            (3, 48),
            (48, 4),
            (48, 3),
            (4, 48),
            (5, 49),
            (49, 6),
            (49, 5),
            (6, 49),
            (6, 50),
            (50, 7),
            (50, 6),
            (7, 50),
            (7, 51),
            (51, 8),
            (51, 7),
            (8, 51),
            (8, 52),
            (52, 9),
            (52, 8),
            (9, 52),
            (10, 53),
            (53, 11),
            (53, 10),
            (11, 53),
            (11, 54),
            (54, 12),
            (54, 11),
            (12, 54),
            (12, 55),
            (55, 13),
            (55, 12),
            (13, 55),
            (13, 56),
            (56, 14),
            (56, 13),
            (14, 56),
            (15, 57),
            (57, 16),
            (57, 15),
            (16, 57),
            (16, 58),
            (58, 17),
            (58, 16),
            (17, 58),
            (17, 59),
            (59, 18),
            (59, 17),
            (18, 59),
            (18, 60),
            (60, 19),
            (60, 18),
            (19, 60),
            (20, 61),
            (61, 21),
            (61, 20),
            (21, 61),
            (21, 62),
            (62, 22),
            (62, 21),
            (22, 62),
            (22, 63),
            (63, 23),
            (63, 22),
            (23, 63),
            (23, 64),
            (64, 24),
            (64, 23),
            (24, 64),
            (4, 29),
            (9, 29),
            (29, 4),
            (29, 9),
            (5, 30),
            (10, 30),
            (30, 5),
            (30, 10),
            (14, 39),
            (19, 39),
            (39, 14),
            (39, 19),
            (15, 40),
            (20, 40),
            (40, 15),
            (40, 20),
            (25, 45),
            (25, 49),
            (45, 25),
            (49, 25),
            (26, 46),
            (26, 50),
            (46, 26),
            (50, 26),
            (27, 47),
            (27, 51),
            (47, 27),
            (51, 27),
            (28, 48),
            (28, 52),
            (48, 28),
            (52, 28),
            (31, 49),
            (31, 53),
            (49, 31),
            (53, 31),
            (32, 50),
            (32, 54),
            (50, 32),
            (54, 32),
            (33, 51),
            (33, 55),
            (51, 33),
            (55, 33),
            (34, 52),
            (34, 56),
            (52, 34),
            (56, 34),
            (35, 53),
            (35, 57),
            (53, 35),
            (57, 35),
            (36, 54),
            (36, 58),
            (54, 36),
            (58, 36),
            (37, 55),
            (37, 59),
            (55, 37),
            (59, 37),
            (38, 56),
            (38, 60),
            (56, 38),
            (60, 38),
            (41, 57),
            (41, 61),
            (57, 41),
            (61, 41),
            (42, 58),
            (42, 62),
            (58, 42),
            (62, 42),
            (43, 59),
            (43, 63),
            (59, 43),
            (63, 43),
            (44, 60),
            (44, 64),
            (60, 44),
            (64, 44),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_heavy_square_graph_5(self):
        d = 5
        graph = rustworkx.generators.heavy_square_graph(d)
        self.assertEqual(len(graph), 3 * d * d - 2 * d)
        self.assertEqual(len(graph.edges()), 2 * d * (d - 1) + 2 * d * (d - 1))
        expected_edges = [
            (0, 45),
            (45, 1),
            (1, 46),
            (46, 2),
            (2, 47),
            (47, 3),
            (3, 48),
            (48, 4),
            (5, 49),
            (49, 6),
            (6, 50),
            (50, 7),
            (7, 51),
            (51, 8),
            (8, 52),
            (52, 9),
            (10, 53),
            (53, 11),
            (11, 54),
            (54, 12),
            (12, 55),
            (55, 13),
            (13, 56),
            (56, 14),
            (15, 57),
            (57, 16),
            (16, 58),
            (58, 17),
            (17, 59),
            (59, 18),
            (18, 60),
            (60, 19),
            (20, 61),
            (61, 21),
            (21, 62),
            (62, 22),
            (22, 63),
            (63, 23),
            (23, 64),
            (64, 24),
            (4, 29),
            (9, 29),
            (5, 30),
            (10, 30),
            (14, 39),
            (19, 39),
            (15, 40),
            (20, 40),
            (25, 45),
            (25, 49),
            (26, 46),
            (26, 50),
            (27, 47),
            (27, 51),
            (28, 48),
            (28, 52),
            (31, 49),
            (31, 53),
            (32, 50),
            (32, 54),
            (33, 51),
            (33, 55),
            (34, 52),
            (34, 56),
            (35, 53),
            (35, 57),
            (36, 54),
            (36, 58),
            (37, 55),
            (37, 59),
            (38, 56),
            (38, 60),
            (41, 57),
            (41, 61),
            (42, 58),
            (42, 62),
            (43, 59),
            (43, 63),
            (44, 60),
            (44, 64),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_heavy_square_graph_3(self):
        d = 3
        graph = rustworkx.generators.directed_heavy_square_graph(d)
        self.assertEqual(len(graph), 3 * d * d - 2 * d)
        self.assertEqual(len(graph.edges()), 2 * d * (d - 1) + 2 * d * (d - 1))
        expected_edges = [
            (0, 15),
            (15, 1),
            (1, 16),
            (16, 2),
            (3, 17),
            (17, 4),
            (4, 18),
            (18, 5),
            (6, 19),
            (19, 7),
            (7, 20),
            (20, 8),
            (2, 11),
            (5, 11),
            (3, 12),
            (6, 12),
            (9, 15),
            (9, 17),
            (10, 16),
            (10, 18),
            (13, 17),
            (13, 19),
            (14, 18),
            (14, 20),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_heavy_square_graph_3_bidirectional(self):
        d = 3
        graph = rustworkx.generators.directed_heavy_square_graph(d, bidirectional=True)
        self.assertEqual(len(graph), 3 * d * d - 2 * d)
        self.assertEqual(len(graph.edges()), 2 * (2 * d * (d - 1) + 2 * d * (d - 1)))
        expected_edges = [
            (0, 15),
            (15, 1),
            (15, 0),
            (1, 15),
            (1, 16),
            (16, 2),
            (16, 1),
            (2, 16),
            (3, 17),
            (17, 4),
            (17, 3),
            (4, 17),
            (4, 18),
            (18, 5),
            (18, 4),
            (5, 18),
            (6, 19),
            (19, 7),
            (19, 6),
            (7, 19),
            (7, 20),
            (20, 8),
            (20, 7),
            (8, 20),
            (2, 11),
            (5, 11),
            (11, 2),
            (11, 5),
            (3, 12),
            (6, 12),
            (12, 3),
            (12, 6),
            (9, 15),
            (9, 17),
            (15, 9),
            (17, 9),
            (10, 16),
            (10, 18),
            (16, 10),
            (18, 10),
            (13, 17),
            (13, 19),
            (17, 13),
            (19, 13),
            (14, 18),
            (14, 20),
            (18, 14),
            (20, 14),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_heavy_square_graph_3(self):
        d = 3
        graph = rustworkx.generators.heavy_square_graph(d)
        self.assertEqual(len(graph), 3 * d * d - 2 * d)
        self.assertEqual(len(graph.edges()), 2 * d * (d - 1) + 2 * d * (d - 1))
        expected_edges = [
            (0, 15),
            (15, 1),
            (1, 16),
            (16, 2),
            (3, 17),
            (17, 4),
            (4, 18),
            (18, 5),
            (6, 19),
            (19, 7),
            (7, 20),
            (20, 8),
            (2, 11),
            (5, 11),
            (3, 12),
            (6, 12),
            (9, 15),
            (9, 17),
            (10, 16),
            (10, 18),
            (13, 17),
            (13, 19),
            (14, 18),
            (14, 20),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_heavy_square_graph_no_d(self):
        with self.assertRaises(TypeError):
            rustworkx.generators.heavy_square_graph()
