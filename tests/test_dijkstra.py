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


class TestDijkstraDiGraph(unittest.TestCase):

    def test_dijkstra(self):
        g = retworkx.PyDAG()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        g.add_edge(a, b, 7)
        g.add_edge(c, a, 9)
        g.add_edge(a, d, 14)
        g.add_edge(b, c, 10)
        g.add_edge(d, c, 2)
        g.add_edge(d, e, 9)
        g.add_edge(b, f, 15)
        g.add_edge(c, f, 11)
        g.add_edge(e, f, 6)
        path = retworkx.digraph_dijkstra_shortest_path_lengths(
            g, a, lambda x: float(x), e)
        expected = {4: 23.0}
        self.assertEqual(expected, path)

    def test_dijkstra_with_graph_input(self):
        g = retworkx.PyGraph()
        g.add_node(0)
        with self.assertRaises(TypeError):
            retworkx.digraph_dijkstra_shortest_path_lengths(g, 0, lambda x: x)


class TestDijkstraGraph(unittest.TestCase):

    def test_dijkstra(self):
        g = retworkx.PyGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        g.add_edge(a, b, 7)
        g.add_edge(c, a, 9)
        g.add_edge(a, d, 14)
        g.add_edge(b, c, 10)
        g.add_edge(d, c, 2)
        g.add_edge(d, e, 9)
        g.add_edge(b, f, 15)
        g.add_edge(c, f, 11)
        g.add_edge(e, f, 6)
        path = retworkx.graph_dijkstra_shortest_path_lengths(
            g, a, lambda x: float(x), e)
        expected = {4: 20.0}
        self.assertEqual(expected, path)

    def test_astar_graph_with_digraph_input(self):
        g = retworkx.PyDAG()
        g.add_node(0)
        with self.assertRaises(TypeError):
            retworkx.graph_dijkstra_shortest_path_lengths(
                g, 0, lambda x: x)
