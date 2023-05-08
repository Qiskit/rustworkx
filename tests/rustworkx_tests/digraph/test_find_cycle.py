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


class TestFindCycle(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyDiGraph()
        self.graph.add_nodes_from(list(range(10)))
        self.graph.add_edges_from_no_data(
            [
                (0, 1),
                (3, 0),
                (0, 5),
                (8, 0),
                (1, 2),
                (1, 6),
                (2, 3),
                (3, 4),
                (4, 5),
                (6, 7),
                (7, 8),
                (8, 9),
            ]
        )

    def test_find_cycle(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(6)))
        graph.add_edges_from_no_data(
            [(0, 1), (0, 3), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (4, 0)]
        )
        res = rustworkx.digraph_find_cycle(graph, 0)
        self.assertEqual([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], res)

    def test_find_cycle_multiple_roots_same_cycles(self):
        res = rustworkx.digraph_find_cycle(self.graph, 0)
        self.assertEqual(res, [(0, 1), (1, 2), (2, 3), (3, 0)])
        res = rustworkx.digraph_find_cycle(self.graph, 1)
        self.assertEqual(res, [(1, 2), (2, 3), (3, 0), (0, 1)])
        res = rustworkx.digraph_find_cycle(self.graph, 5)
        self.assertEqual(res, [])

    def test_find_cycle_disconnected_graphs(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edges_from_no_data([(10, 11), (12, 10), (11, 12)])
        res = rustworkx.digraph_find_cycle(self.graph, 0)
        self.assertEqual(res, [(0, 1), (1, 2), (2, 3), (3, 0)])
        res = rustworkx.digraph_find_cycle(self.graph, 10)
        self.assertEqual(res, [(10, 11), (11, 12), (12, 10)])

    def test_invalid_types(self):
        graph = rustworkx.PyGraph()
        with self.assertRaises(TypeError):
            rustworkx.digraph_find_cycle(graph)

    def test_self_loop(self):
        self.graph.add_edge(1, 1, None)
        res = rustworkx.digraph_find_cycle(self.graph, 0)
        self.assertEqual([(1, 1)], res)
