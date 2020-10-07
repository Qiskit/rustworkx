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

    def test_dijkstra_path(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        edge_list = [
            (a, b, 7),
            (c, a, 9),
            (a, d, 14),
            (b, c, 10),
            (d, c, 2),
            (d, e, 9),
            (b, f, 15),
            (c, f, 11),
            (e, f, 6),
        ]
        g.add_edges_from(edge_list)
        paths = retworkx.digraph_dijkstra_shortest_paths(g, a)
        expected = {
            1: [0, 1],
            2: [0, 3, 2],
            3: [0, 3],
            4: [0, 3, 4],
            5: [0, 1, 5],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_with_weight_fn(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        edge_list = [
            (a, b, 7),
            (c, a, 9),
            (a, d, 14),
            (b, c, 10),
            (d, c, 2),
            (d, e, 9),
            (b, f, 15),
            (c, f, 11),
            (e, f, 6),
        ]
        g.add_edges_from(edge_list)
        paths = retworkx.digraph_dijkstra_shortest_paths(
            g, a, weight_fn=lambda x: x)
        expected = {
            1: [0, 1],
            2: [0, 1, 2],
            3: [0, 3],
            4: [0, 3, 4],
            5: [0, 1, 5],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_with_target(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        edge_list = [
            (a, b, 7),
            (c, a, 9),
            (a, d, 14),
            (b, c, 10),
            (d, c, 2),
            (d, e, 9),
            (b, f, 15),
            (c, f, 11),
            (e, f, 6),
        ]
        g.add_edges_from(edge_list)
        paths = retworkx.digraph_dijkstra_shortest_paths(g, a, target=e)
        expected = {
            4: [0, 3, 4],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_with_weight_fn_and_target(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        edge_list = [
            (a, b, 7),
            (c, a, 9),
            (a, d, 14),
            (b, c, 10),
            (d, c, 2),
            (d, e, 9),
            (b, f, 15),
            (c, f, 11),
            (e, f, 6),
        ]
        g.add_edges_from(edge_list)
        paths = retworkx.digraph_dijkstra_shortest_paths(
            g, a, target=e, weight_fn=lambda x: x)
        expected = {
            4: [0, 3, 4],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_undirected(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        edge_list = [
            (a, b, 7),
            (c, a, 9),
            (a, d, 14),
            (b, c, 10),
            (d, c, 2),
            (d, e, 9),
            (b, f, 15),
            (c, f, 11),
            (e, f, 6),
        ]
        g.add_edges_from(edge_list)
        paths = retworkx.digraph_dijkstra_shortest_paths(g, a,
                                                         as_undirected=True)
        expected = {
            1: [0, 1],
            2: [0, 2],
            3: [0, 3],
            4: [0, 3, 4],
            5: [0, 1, 5],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_undirected_with_weight_fn(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        edge_list = [
            (a, b, 7),
            (c, a, 9),
            (a, d, 14),
            (b, c, 10),
            (d, c, 2),
            (d, e, 9),
            (b, f, 15),
            (c, f, 11),
            (e, f, 6),
        ]
        g.add_edges_from(edge_list)
        paths = retworkx.digraph_dijkstra_shortest_paths(g, a,
                                                         weight_fn=lambda x: x,
                                                         as_undirected=True)
        expected = {
            1: [0, 1],
            2: [0, 2],
            3: [0, 3],
            4: [0, 3, 4],
            5: [0, 1, 5],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_undirected_with_target(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        edge_list = [
            (a, b, 7),
            (c, a, 9),
            (a, d, 14),
            (b, c, 10),
            (d, c, 2),
            (d, e, 9),
            (b, f, 15),
            (c, f, 11),
            (e, f, 6),
        ]
        g.add_edges_from(edge_list)
        paths = retworkx.digraph_dijkstra_shortest_paths(g, a,
                                                         target=e,
                                                         as_undirected=True)
        expected = {
            4: [0, 3, 4],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_undirected_with_weight_fn_and_target(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        edge_list = [
            (a, b, 7),
            (c, a, 9),
            (a, d, 14),
            (b, c, 10),
            (d, c, 2),
            (d, e, 9),
            (b, f, 15),
            (c, f, 11),
            (e, f, 6),
        ]
        g.add_edges_from(edge_list)
        paths = retworkx.digraph_dijkstra_shortest_paths(g, a,
                                                         target=e,
                                                         weight_fn=lambda x: x,
                                                         as_undirected=True)
        expected = {
            4: [0, 3, 4],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_with_no_goal_set(self):
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
            g, a, lambda x: 1)
        expected = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0, 5: 2.0}
        self.assertEqual(expected, path)

    def test_dijkstra_with_no_path(self):
        g = retworkx.PyDiGraph()
        a = g.add_node('A')
        g.add_node('B')
        path = retworkx.digraph_dijkstra_shortest_path_lengths(
            g, a, lambda x: float(x))
        expected = {}
        self.assertEqual(expected, path)

    def test_dijkstra_path_with_no_path(self):
        g = retworkx.PyDiGraph()
        a = g.add_node('A')
        g.add_node('B')
        path = retworkx.digraph_dijkstra_shortest_paths(
            g, a)
        expected = {}
        self.assertEqual(expected, path)

    def test_dijkstra_with_disconnected_nodes(self):
        g = retworkx.PyDiGraph()
        a = g.add_node('A')
        b = g.add_child(a, 'B', 1.2)
        g.add_node('C')
        g.add_parent(b, 'D', 2.4)
        path = retworkx.digraph_dijkstra_shortest_path_lengths(
            g, a, lambda x: x)
        expected = {1: 1.2}
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

    def test_dijkstra_path(self):
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
        path = retworkx.graph_dijkstra_shortest_paths(
            g, a, weight_fn=lambda x: float(x), target=e)
        expected = {
            4: [0, 3, 4]
        }
        self.assertEqual(expected, path)

    def test_dijkstra_with_no_goal_set(self):
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
            g, a, lambda x: 1)
        expected = {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 2.0}
        self.assertEqual(expected, path)

    def test_dijkstra_path_with_no_goal_set(self):
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
        path = retworkx.graph_dijkstra_shortest_paths(
            g, a)
        expected = {
            1: [0, 1],
            2: [0, 2],
            3: [0, 3],
            4: [0, 3, 4],
            5: [0, 1, 5],
        }
        self.assertEqual(expected, path)

    def test_dijkstra_with_no_path(self):
        g = retworkx.PyGraph()
        a = g.add_node('A')
        g.add_node('B')
        path = retworkx.graph_dijkstra_shortest_path_lengths(
            g, a, lambda x: float(x))
        expected = {}
        self.assertEqual(expected, path)

    def test_dijkstra_path_with_no_path(self):
        g = retworkx.PyGraph()
        a = g.add_node('A')
        g.add_node('B')
        path = retworkx.graph_dijkstra_shortest_paths(
            g, a, weight_fn=lambda x: float(x))
        expected = {}
        self.assertEqual(expected, path)

    def test_dijkstra_with_disconnected_nodes(self):
        g = retworkx.PyDiGraph()
        a = g.add_node('A')
        b = g.add_node('B')
        g.add_edge(a, b, 1.2)
        g.add_node('C')
        d = g.add_node('D')
        g.add_edge(b, d, 2.4)
        path = retworkx.digraph_dijkstra_shortest_path_lengths(
            g, a, lambda x: round(x, 1))
        # Computers never work:
        expected = {1: 1.2, 3: 3.5999999999999996}
        self.assertEqual(expected, path)

    def test_dijkstra_graph_with_digraph_input(self):
        g = retworkx.PyDAG()
        g.add_node(0)
        with self.assertRaises(TypeError):
            retworkx.graph_dijkstra_shortest_path_lengths(
                g, 0, lambda x: x)
