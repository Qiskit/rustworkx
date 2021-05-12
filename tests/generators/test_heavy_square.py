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


class TestHeavyHexGraph(unittest.TestCase):
    def test_heavy_square_graph_5(self):
        d = 5
        graph = retworkx.generators.heavy_square_graph(d)
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
            (29, 9),
            (5, 30),
            (30, 10),
            (14, 39),
            (39, 19),
            (15, 40),
            (40, 20),
            (45, 25),
            (25, 49),
            (46, 26),
            (26, 50),
            (47, 27),
            (27, 51),
            (48, 28),
            (28, 52),
            (49, 31),
            (31, 53),
            (50, 32),
            (32, 54),
            (51, 33),
            (33, 55),
            (52, 34),
            (34, 56),
            (53, 35),
            (35, 57),
            (54, 36),
            (36, 58),
            (55, 37),
            (37, 59),
            (56, 38),
            (38, 60),
            (57, 41),
            (41, 61),
            (58, 42),
            (42, 62),
            (59, 43),
            (43, 63),
            (60, 44),
            (44, 64),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_heavy_square_graph_3(self):
        d = 3
        graph = retworkx.generators.heavy_square_graph(d)
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
            (11, 5),
            (3, 12),
            (12, 6),
            (15, 9),
            (9, 17),
            (16, 10),
            (10, 18),
            (17, 13),
            (13, 19),
            (18, 14),
            (14, 20),
        ]
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_heavy_square_graph_no_d(self):
        with self.assertRaises(TypeError):
            retworkx.generators.heavy_square_graph()
