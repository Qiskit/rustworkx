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


class TestCoreNumber(unittest.TestCase):
    def setUp(self):
        # This is the example graph in Figure 1 from Batagelj and
        # Zaversnik's paper titled An O(m) Algorithm for Cores
        # Decomposition of Networks, 2003,
        # http://arXiv.org/abs/cs/0310049.  With nodes labeled as
        # shown, the 3-core is given by nodes 0-7, the 2-core by nodes
        # 8-15, the 1-core by nodes 16-19 and node 20 is in the
        # 0-core.
        self.example_edges = [
            (0, 2),
            (0, 3),
            (0, 5),
            (1, 4),
            (1, 6),
            (1, 7),
            (2, 3),
            (3, 5),
            (2, 5),
            (5, 6),
            (4, 6),
            (4, 7),
            (6, 7),
            (5, 8),
            (6, 8),
            (6, 9),
            (8, 9),
            (0, 10),
            (1, 10),
            (1, 11),
            (10, 11),
            (12, 13),
            (13, 15),
            (14, 15),
            (12, 14),
            (8, 19),
            (11, 16),
            (11, 17),
            (12, 18),
        ]

        example_core = {}
        for i in range(8):
            example_core[i] = 3
        for i in range(8, 16):
            example_core[i] = 2
        for i in range(16, 20):
            example_core[i] = 1
        example_core[20] = 0
        self.example_core = example_core

    def test_undirected_empty(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.core_number(graph)
        self.assertIsInstance(res, dict)
        self.assertEqual(res, {})

    def test_undirected_all_0(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(4)))
        res = rustworkx.core_number(graph)
        self.assertIsInstance(res, dict)
        self.assertEqual(res, {0: 0, 1: 0, 2: 0, 3: 0})

    def test_undirected_all_3(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(4)))
        graph.add_edges_from_no_data([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        res = rustworkx.core_number(graph)
        self.assertIsInstance(res, dict)
        self.assertEqual(res, {0: 3, 1: 3, 2: 3, 3: 3})

    def test_undirected_paper_example(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(21)))
        graph.add_edges_from_no_data(self.example_edges)
        res = rustworkx.core_number(graph)
        self.assertIsInstance(res, dict)
        self.assertEqual(res, self.example_core)
