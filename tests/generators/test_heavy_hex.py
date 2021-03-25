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

    def test_heavy_hex_graph_1_1(self):
        graph = retworkx.generators.heavy_hex_graph(1,1)
        self.assertEqual(len(graph), 57)
        self.assertEqual(len(graph.edges()), 64)
        
        expected_edges = [
            (0, 1), (0, 9), (1, 2), (2, 3), (3, 4), 
            (3, 10), (4, 5), (5, 6), (6, 7), (7, 8), 
            (7, 11), (12, 9), (12, 13), (13, 14), (13, 21), 
            (14, 15), (15, 10), (15, 16), (16, 17), (17, 18), 
            (17, 22), (18, 19), (19, 11), (19, 20), (20, 23), 
            (24, 25), (24, 33), (25, 21), (25, 26), (26, 27), 
            (27, 28), (27, 34), (28, 29), (29, 22), (29, 30), 
            (30, 31), (31, 32), (31, 35), (32, 23), (36, 33), 
            (36, 37), (37, 38), (37, 45), (38, 39), (39, 34), 
            (39, 40), (40, 41), (41, 42), (41, 46), (42, 43), 
            (43, 35), (43, 44), (44, 47), (48, 49), (49, 45), 
            (49, 50), (50, 51), (51, 52), (52, 53), (53, 46), 
            (53, 54), (54, 55), (55, 56), (56, 47)
            ]

        for got in graph.edge_list():
            self.assertIn(got, expected_edges)

    def test_heavy_hex_graph_0_0(self):
        graph = retworkx.generators.heavy_hex_graph(0,0)
        self.assertEqual(len(graph), 0)
        self.assertEqual(len(graph.edges()), 0)

    def test_heavy_hex_1_1_weights(self):
        graph = retworkx.generators.heavy_hex_graph(1, 1, weights=list(range(57)))
        self.assertEqual(len(graph), 57)
        self.assertEqual([x for x in range(57)], graph.nodes())

    def test_heavy_hex_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            retworkx.generators.star_graph()
