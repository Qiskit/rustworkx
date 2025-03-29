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


class TestDijkstraDiGraph(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyDiGraph()
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

    def test_dijkstra(self):
        path = rustworkx.digraph_dijkstra_shortest_path_lengths(self.graph, self.a, float, self.e)
        expected = {4: 23.0}
        self.assertEqual(expected, path)

    def test_dijkstra_length_with_no_path(self):
        g = rustworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        path_lengths = rustworkx.digraph_dijkstra_shortest_path_lengths(
            g, a, edge_cost_fn=float, goal=b
        )
        expected = {}
        self.assertEqual(expected, path_lengths)

    def test_dijkstra_path(self):
        paths = rustworkx.digraph_dijkstra_shortest_paths(self.graph, self.a)
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

    def test_dijkstra_has_path(self):
        g = rustworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")

        edge_list = [
            (a, b, 7),
            (c, b, 9),
            (c, b, 10),
        ]
        g.add_edges_from(edge_list)

        self.assertFalse(rustworkx.digraph_has_path(g, a, c))

    def test_dijkstra_path_with_weight_fn(self):
        paths = rustworkx.digraph_dijkstra_shortest_paths(self.graph, self.a, weight_fn=lambda x: x)
        expected = {
            1: [0, 1],
            2: [0, 3, 2],
            3: [0, 3],
            4: [0, 3, 4],
            5: [0, 1, 5],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_with_target(self):
        paths = rustworkx.digraph_dijkstra_shortest_paths(self.graph, self.a, target=self.e)
        expected = {
            4: [0, 3, 4],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_with_weight_fn_and_target(self):
        paths = rustworkx.digraph_dijkstra_shortest_paths(
            self.graph, self.a, target=self.e, weight_fn=lambda x: x
        )
        expected = {
            4: [0, 3, 4],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_undirected(self):
        paths = rustworkx.digraph_dijkstra_shortest_paths(self.graph, self.a, as_undirected=True)
        expected = {
            1: [0, 1],
            2: [0, 2],
            3: [0, 3],
            4: [0, 3, 4],
            5: [0, 1, 5],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_undirected_with_weight_fn(self):
        paths = rustworkx.digraph_dijkstra_shortest_paths(
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

    def test_dijkstra_path_undirected_with_target(self):
        paths = rustworkx.digraph_dijkstra_shortest_paths(
            self.graph, self.a, target=self.e, as_undirected=True
        )
        expected = {
            4: [0, 3, 4],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_path_undirected_with_weight_fn_and_target(self):
        paths = rustworkx.digraph_dijkstra_shortest_paths(
            self.graph,
            self.a,
            target=self.e,
            weight_fn=lambda x: x,
            as_undirected=True,
        )
        expected = {
            4: [0, 2, 3, 4],
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_with_no_goal_set(self):
        path = rustworkx.digraph_dijkstra_shortest_path_lengths(self.graph, self.a, lambda x: 1)
        expected = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0, 5: 2.0}
        self.assertEqual(expected, path)

    def test_dijkstra_with_no_path(self):
        g = rustworkx.PyDiGraph()
        a = g.add_node("A")
        g.add_node("B")
        path = rustworkx.digraph_dijkstra_shortest_path_lengths(g, a, float)
        expected = {}
        self.assertEqual(expected, path)

    def test_dijkstra_path_with_no_path(self):
        g = rustworkx.PyDiGraph()
        a = g.add_node("A")
        g.add_node("B")
        path = rustworkx.digraph_dijkstra_shortest_paths(g, a)
        expected = {}
        self.assertEqual(expected, path)

    def test_dijkstra_with_disconnected_nodes(self):
        g = rustworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_child(a, "B", 1.2)
        g.add_node("C")
        g.add_parent(b, "D", 2.4)
        path = rustworkx.digraph_dijkstra_shortest_path_lengths(g, a, lambda x: x)
        expected = {1: 1.2}
        self.assertEqual(expected, path)

    def test_dijkstra_with_graph_input(self):
        g = rustworkx.PyGraph()
        g.add_node(0)
        with self.assertRaises(TypeError):
            rustworkx.digraph_dijkstra_shortest_path_lengths(g, 0, lambda x: x)

    def test_dijkstra_all_pair_path_lengths(self):
        lengths = rustworkx.digraph_all_pairs_dijkstra_path_lengths(self.graph, float)
        expected = {
            0: {1: 7.0, 2: 16.0, 3: 14.0, 4: 23.0, 5: 22.0},
            1: {0: 19.0, 2: 10.0, 3: 33.0, 4: 42.0, 5: 15.0},
            2: {0: 9.0, 1: 16.0, 3: 23.0, 4: 32.0, 5: 11.0},
            3: {0: 11.0, 1: 18.0, 2: 2.0, 4: 9.0, 5: 13.0},
            4: {5: 6.0},
            5: {},
        }
        self.assertEqual(expected, lengths)

    def test_dijkstra_all_pair_paths(self):
        paths = rustworkx.digraph_all_pairs_dijkstra_shortest_paths(self.graph, float)
        expected = {
            0: {1: [0, 1], 2: [0, 3, 2], 3: [0, 3], 4: [0, 3, 4], 5: [0, 1, 5]},
            1: {
                0: [1, 2, 0],
                2: [1, 2],
                3: [1, 2, 0, 3],
                4: [1, 2, 0, 3, 4],
                5: [1, 5],
            },
            2: {
                0: [2, 0],
                1: [2, 0, 1],
                3: [2, 0, 3],
                4: [2, 0, 3, 4],
                5: [2, 5],
            },
            3: {
                0: [3, 2, 0],
                1: [3, 2, 0, 1],
                2: [3, 2],
                4: [3, 4],
                5: [3, 2, 5],
            },
            4: {5: [4, 5]},
            5: {},
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_all_pair_path_lengths_with_node_removal(self):
        self.graph.remove_node(3)
        lengths = rustworkx.digraph_all_pairs_dijkstra_path_lengths(self.graph, float)
        expected = {
            0: {1: 7.0, 2: 17.0, 5: 22.0},
            1: {0: 19.0, 2: 10.0, 5: 15.0},
            2: {0: 9.0, 1: 16.0, 5: 11.0},
            4: {5: 6.0},
            5: {},
        }
        self.assertEqual(expected, lengths)

    def test_dijkstra_all_pair_paths_with_node_removal(self):
        self.graph.remove_node(3)
        lengths = rustworkx.digraph_all_pairs_dijkstra_shortest_paths(self.graph, float)
        expected = {
            0: {1: [0, 1], 2: [0, 1, 2], 5: [0, 1, 5]},
            1: {0: [1, 2, 0], 2: [1, 2], 5: [1, 5]},
            2: {0: [2, 0], 1: [2, 0, 1], 5: [2, 5]},
            4: {5: [4, 5]},
            5: {},
        }
        self.assertEqual(expected, lengths)

    def test_dijkstra_all_pair_path_lengths_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        self.assertEqual({}, rustworkx.digraph_all_pairs_dijkstra_path_lengths(graph, float))

    def test_dijkstra_all_pair_shortest_paths_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        self.assertEqual({}, rustworkx.digraph_all_pairs_dijkstra_shortest_paths(graph, float))

    def test_dijkstra_all_pair_path_lengths_graph_no_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(1000)))
        expected = {x: {} for x in range(1000)}
        self.assertEqual(
            expected,
            rustworkx.digraph_all_pairs_dijkstra_path_lengths(graph, float),
        )

    def test_dijkstra_all_pair_shortest_paths_no_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(1000)))
        expected = {x: {} for x in range(1000)}
        self.assertEqual(
            expected,
            rustworkx.digraph_all_pairs_dijkstra_shortest_paths(graph, float),
        )

    def dijkstra_with_invalid_weights(self):
        graph = rustworkx.generators.directed_path_graph(2)
        for invalid_weight in [float("nan"), -1]:
            for as_undirected in [False, True]:
                with self.subTest(invalid_weight=invalid_weight, as_undirected=as_undirected):
                    with self.assertRaises(ValueError):
                        rustworkx.digraph_dijkstra_shortest_paths(
                            graph,
                            source=0,
                            weight_fn=lambda _: invalid_weight,
                            as_undirected=as_undirected,
                        )

    def dijkstra_lengths_with_invalid_weights(self):
        graph = rustworkx.generators.directed_path_graph(2)
        for invalid_weight in [float("nan"), -1]:
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises(ValueError):
                    rustworkx.digraph_dijkstra_shortest_path_lengths(
                        graph, node=0, edge_cost_fn=lambda _: invalid_weight
                    )

    def all_pairs_dijkstra_with_invalid_weights(self):
        graph = rustworkx.generators.directed_path_graph(2)
        for invalid_weight in [float("nan"), -1]:
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises(ValueError):
                    rustworkx.digraph_all_pairs_dijkstra_shortest_paths(
                        graph, edge_cost_fn=lambda _: invalid_weight
                    )

    def all_pairs_dijkstra_lengths_with_invalid_weights(self):
        graph = rustworkx.generators.directed_path_graph(2)
        for invalid_weight in [float("nan"), -1]:
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises(ValueError):
                    rustworkx.digraph_all_pairs_dijkstra_path_lengths(
                        graph, edge_cost_fn=lambda _: invalid_weight
                    )

    def test_dijkstra_path_digraph_with_invalid_source(self):
        with self.assertRaises(IndexError):
            rustworkx.dijkstra_shortest_paths(self.graph, len(self.graph.node_indices()) + 1)

    def test_dijkstra_path_digraph_lengths_with_invalid_source(self):
        with self.assertRaises(IndexError):
            rustworkx.dijkstra_shortest_path_lengths(
                self.graph, len(self.graph.node_indices()) + 1, edge_cost_fn=lambda x: x
            )
