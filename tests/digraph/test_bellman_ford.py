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


class TestBellmanFordDiGraph(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyDiGraph()
        self.a = self.graph.add_node("A")
        self.b = self.graph.add_node("B")
        self.c = self.graph.add_node("C")
        self.d = self.graph.add_node("D")
        self.e = self.graph.add_node("E")
        self.f = self.graph.add_node("F")
        edge_list = [
            (self.a, self.b, 7),
            (self.c, self.a, 9),
            (self.a, self.d, 14),
            (self.b, self.c, 10),
            (self.d, self.c, 2),
            (self.d, self.e, 9),
            (self.b, self.f, 15),
            (self.c, self.f, 11),
            (self.e, self.f, 6),
        ]
        self.graph.add_edges_from(edge_list)

    def test_bellman_ford(self):
        path = retworkx.digraph_bellman_ford_shortest_path_lengths(
            self.graph, self.a, lambda x: float(x)
        )
        expected = {1: 7.0, 2: 16.0, 3: 14.0, 4: 23.0, 5: 22.0}
        self.assertEqual(expected, path)

    def test_bellman_ford_length_with_no_path(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        g.add_node("B")
        path_lenghts = retworkx.digraph_bellman_ford_shortest_path_lengths(g, a, edge_cost_fn=float)
        expected = {}
        self.assertEqual(expected, path_lenghts)

    def test_bellman_ford_path(self):
        paths = retworkx.digraph_bellman_ford_shortest_paths(self.graph, self.a)
        expected = {
            # a -> b
            1: [0, 1],
            # a -> c: a, d, c
            2: [0, 3, 2],
            # a -> d
            3: [0, 3],
            # a -> e: a, d, e
            4: [0, 3, 4],
            # a -> f: a, b, f
            5: [0, 1, 5],
        }
        self.assertEqual(expected, paths)

    def test_bellman_ford_path_with_weight_fn(self):
        paths = retworkx.digraph_bellman_ford_shortest_paths(
            self.graph, self.a, weight_fn=lambda x: x
        )
        expected = {
            1: [0, 1],
            2: [0, 3, 2],
            3: [0, 3],
            4: [0, 3, 4],
            5: [0, 1, 5],
        }
        self.assertEqual(expected, paths)

    def test_bellman_ford_path_undirected(self):
        paths = retworkx.digraph_bellman_ford_shortest_paths(self.graph, self.a, as_undirected=True)
        expected = {
            1: [0, 1],
            2: [0, 2],
            3: [0, 3],
            4: [0, 3, 4],
            5: [0, 1, 5],
        }
        self.assertEqual(expected, paths)

    def test_bellman_ford_path_undirected_with_weight_fn(self):
        paths = retworkx.digraph_bellman_ford_shortest_paths(
            self.graph, self.a, weight_fn=lambda x: x, as_undirected=True
        )
        expected = {
            1: [0, 1],
            2: [0, 2],
            3: [0, 2, 3],
            4: [0, 2, 3, 4],
            5: [0, 2, 5],
        }
        self.assertEqual(expected, paths)

    def test_bellman_ford_with_no_goal_set(self):
        path = retworkx.digraph_bellman_ford_shortest_path_lengths(self.graph, self.a, lambda x: 1)
        expected = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0, 5: 2.0}
        self.assertEqual(expected, path)

    def test_bellman_ford_with_no_path(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        g.add_node("B")
        path = retworkx.digraph_bellman_ford_shortest_path_lengths(g, a, lambda x: float(x))
        expected = {}
        self.assertEqual(expected, path)

    def test_bellman_ford_path_with_no_path(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        g.add_node("B")
        path = retworkx.digraph_bellman_ford_shortest_paths(g, a)
        expected = {}
        self.assertEqual(expected, path)

    def test_bellman_ford_with_disconnected_nodes(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_child(a, "B", 1.2)
        g.add_node("C")
        g.add_parent(b, "D", 2.4)
        path = retworkx.digraph_bellman_ford_shortest_path_lengths(g, a, lambda x: x)
        expected = {1: 1.2}
        self.assertEqual(expected, path)

    def test_bellman_ford_with_graph_input(self):
        g = retworkx.PyGraph()
        g.add_node(0)
        with self.assertRaises(TypeError):
            retworkx.digraph_bellman_ford_shortest_path_lengths(g, 0, lambda x: x)

    def bellman_ford_with_invalid_weights(self):
        graph = retworkx.generators.directed_path_graph(2)

        for as_undirected in [False, True]:
            with self.subTest(as_undirected=as_undirected):
                with self.assertRaises(ValueError):
                    retworkx.digraph_bellman_ford_shortest_paths(
                        graph,
                        source=0,
                        weight_fn=lambda _: float("nan"),
                        as_undirected=as_undirected,
                    )

    def bellman_ford_lengths_with_invalid_weights(self):
        graph = retworkx.generators.directed_path_graph(2)

        with self.assertRaises(ValueError):
            retworkx.digraph_bellman_ford_shortest_path_lengths(
                graph, node=0, edge_cost_fn=lambda _: float("nan")
            )

    def test_raises_negative_cycle_bellman_ford_paths(self):
        graph = retworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.add_edges_from(
            [
                (0, 1, 1),
                (1, 2, -1),
                (2, 3, -1),
                (3, 0, -1),
            ]
        )

        with self.assertRaises(retworkx.NegativeCycle):
            retworkx.bellman_ford_shortest_paths(graph, 0, weight_fn=float)

    def test_raises_negative_cycle_bellman_ford_path_lenghts(self):
        graph = retworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.add_edges_from(
            [
                (0, 1, 1),
                (1, 2, -1),
                (2, 3, -1),
                (3, 0, -1),
            ]
        )

        with self.assertRaises(retworkx.NegativeCycle):
            retworkx.bellman_ford_shortest_path_lengths(graph, 0, edge_cost_fn=float)

    def test_negative_edges_but_no_cycle(self):
        graph = retworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.add_edges_from(
            [
                (0, 1, 1),
                (1, 2, -1),
                (2, 3, -1),
                (3, 0, 1),
            ]
        )

        result = retworkx.bellman_ford_shortest_path_lengths(graph, 0, edge_cost_fn=float)

        expected = {k: v for k, v in retworkx.floyd_warshall(graph, float)[0].items() if k != 0}

        self.assertEqual(result, expected)

    def test_negative_cycle_find_true_cycle(self):
        graph = retworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.add_edges_from(
            [
                (0, 1, 1),
                (1, 2, -1),
                (2, 3, -1),
                (3, 0, -1),
            ]
        )

        self.assertTrue(retworkx.negative_edge_cycle(graph, float))

    def test_negative_cycle_find_no_cycle(self):
        graph = retworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.add_edges_from(
            [
                (0, 1, 1),
                (1, 2, -1),
                (2, 3, -1),
                (3, 0, 1),
            ]
        )

        self.assertFalse(retworkx.negative_edge_cycle(graph, float))
