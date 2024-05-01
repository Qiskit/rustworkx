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
import rustworkx.generators


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

    def assertCycle(self, first_node, graph, res):
        self.assertEqual(first_node, res[0][0])
        for i in range(len(res)):
            s, t = res[i]
            self.assertTrue(graph.has_edge(s, t))
            next_s, _ = res[(i + 1) % len(res)]
            self.assertEqual(t, next_s)

    def test_find_cycle(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(6)))
        graph.add_edges_from_no_data(
            [(0, 1), (0, 3), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (4, 0)]
        )
        res = rustworkx.digraph_find_cycle(graph, 0)
        self.assertCycle(0, graph, res)

    def test_find_cycle_multiple_roots_same_cycles(self):
        res = rustworkx.digraph_find_cycle(self.graph, 0)
        self.assertCycle(0, self.graph, res)
        res = rustworkx.digraph_find_cycle(self.graph, 1)
        self.assertCycle(1, self.graph, res)
        res = rustworkx.digraph_find_cycle(self.graph, 5)
        self.assertEqual(res, [])

    def test_find_cycle_disconnected_graphs(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edges_from_no_data([(10, 11), (12, 10), (11, 12)])
        res = rustworkx.digraph_find_cycle(self.graph, 0)
        self.assertCycle(0, self.graph, res)
        res = rustworkx.digraph_find_cycle(self.graph, 10)
        self.assertCycle(10, self.graph, res)

    def test_invalid_types(self):
        graph = rustworkx.PyGraph()
        with self.assertRaises(TypeError):
            rustworkx.digraph_find_cycle(graph)

    def test_self_loop(self):
        self.graph.add_edge(1, 1, None)
        res = rustworkx.digraph_find_cycle(self.graph, 0)
        self.assertCycle(1, self.graph, res)

    def test_no_cycle_no_source(self):
        g = rustworkx.generators.directed_grid_graph(10, 10)
        res = rustworkx.digraph_find_cycle(g)
        self.assertEqual(res, [])

    def test_cycle_no_source(self):
        g = rustworkx.generators.directed_path_graph(1000)
        a = g.add_node(1000)
        b = g.node_indices()[-1]
        g.add_edge(b, a, None)
        g.add_edge(a, b, None)
        res = rustworkx.digraph_find_cycle(g)
        self.assertEqual(len(res), 2)
        self.assertTrue(a in res[0] and b in res[0])
        self.assertTrue(a in res[1] and b in res[1])
