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


class TestDijkstraGraph(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyGraph()
        self.a = self.graph.add_node("A")
        self.b = self.graph.add_node("B")
        self.c = self.graph.add_node("C")
        self.d = self.graph.add_node("D")
        self.e = self.graph.add_node("E")
        self.f = self.graph.add_node("F")
        self.graph.add_edge(self.a, self.b, 7)
        self.graph.add_edge(self.c, self.a, 9)
        self.graph.add_edge(self.a, self.d, 14)
        self.graph.add_edge(self.b, self.c, 10)
        self.graph.add_edge(self.d, self.c, 2)
        self.graph.add_edge(self.d, self.e, 9)
        self.graph.add_edge(self.b, self.f, 15)
        self.graph.add_edge(self.c, self.f, 11)
        self.graph.add_edge(self.e, self.f, 6)

    def test_dijkstra(self):
        path = retworkx.graph_dijkstra_shortest_path_lengths(
            self.graph, self.a, lambda x: float(x), self.e
        )
        expected = {4: 20.0}
        self.assertEqual(expected, path)

    def test_dijkstra_path(self):
        path = retworkx.graph_dijkstra_shortest_paths(
            self.graph, self.a, weight_fn=lambda x: float(x), target=self.e
        )
        # a -> d -> e = 23
        # a -> c -> d -> e = 20
        expected = {4: [self.a, self.c, self.d, self.e]}
        self.assertEqual(expected, path)

    def test_dijkstra_with_no_goal_set(self):
        path = retworkx.graph_dijkstra_shortest_path_lengths(
            self.graph, self.a, lambda x: 1
        )
        expected = {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 2.0}
        self.assertEqual(expected, path)

    def test_dijkstra_length_with_no_path(self):
        g = retworkx.PyGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        path_lenghts = retworkx.graph_dijkstra_shortest_path_lengths(
            g, a, edge_cost_fn=float, goal=b
        )
        expected = {}
        self.assertEqual(expected, path_lenghts)

    def test_dijkstra_path_with_no_goal_set(self):
        path = retworkx.graph_dijkstra_shortest_paths(self.graph, self.a)
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
        a = g.add_node("A")
        g.add_node("B")
        path = retworkx.graph_dijkstra_shortest_path_lengths(
            g, a, lambda x: float(x)
        )
        expected = {}
        self.assertEqual(expected, path)

    def test_dijkstra_path_with_no_path(self):
        g = retworkx.PyGraph()
        a = g.add_node("A")
        g.add_node("B")
        path = retworkx.graph_dijkstra_shortest_paths(
            g, a, weight_fn=lambda x: float(x)
        )
        expected = {}
        self.assertEqual(expected, path)

    def test_dijkstra_with_disconnected_nodes(self):
        g = retworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        g.add_edge(a, b, 1.2)
        g.add_node("C")
        d = g.add_node("D")
        g.add_edge(b, d, 2.4)
        path = retworkx.digraph_dijkstra_shortest_path_lengths(
            g, a, lambda x: round(x, 1)
        )
        # Computers never work:
        expected = {1: 1.2, 3: 3.5999999999999996}
        self.assertEqual(expected, path)

    def test_dijkstra_graph_with_digraph_input(self):
        g = retworkx.PyDAG()
        g.add_node(0)
        with self.assertRaises(TypeError):
            retworkx.graph_dijkstra_shortest_path_lengths(g, 0, lambda x: x)

    def test_dijkstra_all_pair_path_lengths(self):
        lengths = retworkx.graph_all_pairs_dijkstra_path_lengths(
            self.graph, float
        )
        expected = {
            0: {1: 7.0, 2: 9.0, 3: 11.0, 4: 20.0, 5: 20.0},
            1: {0: 7.0, 2: 10.0, 3: 12.0, 4: 21.0, 5: 15.0},
            2: {0: 9.0, 1: 10.0, 3: 2.0, 4: 11.0, 5: 11.0},
            3: {0: 11.0, 1: 12.0, 2: 2.0, 4: 9.0, 5: 13.0},
            4: {0: 20.0, 1: 21.0, 2: 11.0, 3: 9.0, 5: 6.0},
            5: {0: 20.0, 1: 15.0, 2: 11.0, 3: 13.0, 4: 6.0},
        }
        self.assertEqual(expected, lengths)

    def test_dijkstra_all_pair_paths(self):
        paths = retworkx.graph_all_pairs_dijkstra_shortest_paths(
            self.graph, float
        )
        expected = {
            0: {
                1: [0, 1],
                2: [0, 2],
                3: [0, 2, 3],
                4: [0, 2, 3, 4],
                5: [0, 2, 5],
            },
            1: {0: [1, 0], 2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 3, 4], 5: [1, 5]},
            2: {0: [2, 0], 1: [2, 1], 3: [2, 3], 4: [2, 3, 4], 5: [2, 5]},
            3: {0: [3, 2, 0], 1: [3, 2, 1], 2: [3, 2], 4: [3, 4], 5: [3, 2, 5]},
            4: {
                0: [4, 3, 2, 0],
                1: [4, 5, 1],
                2: [4, 3, 2],
                3: [4, 3],
                5: [4, 5],
            },
            5: {0: [5, 2, 0], 1: [5, 1], 2: [5, 2], 3: [5, 2, 3], 4: [5, 4]},
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_all_pair_path_lengths_with_node_removal(self):
        self.graph.remove_node(3)
        lengths = retworkx.graph_all_pairs_dijkstra_path_lengths(
            self.graph, float
        )
        expected = {
            0: {1: 7.0, 2: 9.0, 4: 26.0, 5: 20.0},
            1: {0: 7.0, 2: 10.0, 4: 21.0, 5: 15.0},
            2: {0: 9.0, 1: 10.0, 4: 17.0, 5: 11.0},
            4: {0: 26.0, 1: 21.0, 2: 17.0, 5: 6.0},
            5: {0: 20.0, 1: 15.0, 2: 11.0, 4: 6.0},
        }
        self.assertEqual(expected, lengths)

    def test_dijkstra_all_pair_paths_with_node_removal(self):
        self.graph.remove_node(3)
        paths = retworkx.graph_all_pairs_dijkstra_shortest_paths(
            self.graph, float
        )
        expected = {
            0: {1: [0, 1], 2: [0, 2], 4: [0, 2, 5, 4], 5: [0, 2, 5]},
            1: {0: [1, 0], 2: [1, 2], 4: [1, 5, 4], 5: [1, 5]},
            2: {0: [2, 0], 1: [2, 1], 4: [2, 5, 4], 5: [2, 5]},
            4: {0: [4, 5, 2, 0], 1: [4, 5, 1], 2: [4, 5, 2], 5: [4, 5]},
            5: {0: [5, 2, 0], 1: [5, 1], 2: [5, 2], 4: [5, 4]},
        }
        self.assertEqual(expected, paths)

    def test_dijkstra_all_pair_path_lengths_empty_graph(self):
        graph = retworkx.PyGraph()
        self.assertEqual(
            {}, retworkx.graph_all_pairs_dijkstra_path_lengths(graph, float)
        )

    def test_dijkstra_all_pair_shortest_paths_empty_graph(self):
        graph = retworkx.PyGraph()
        self.assertEqual(
            {}, retworkx.graph_all_pairs_dijkstra_shortest_paths(graph, float)
        )

    def test_dijkstra_all_pair_path_lengths_graph_no_edges(self):
        graph = retworkx.PyGraph()
        graph.add_nodes_from(list(range(1000)))
        expected = {x: {} for x in range(1000)}
        self.assertEqual(
            expected,
            retworkx.graph_all_pairs_dijkstra_path_lengths(graph, float),
        )

    def test_dijkstra_all_pair_shortest_paths_no_edges(self):
        graph = retworkx.PyGraph()
        graph.add_nodes_from(list(range(1000)))
        expected = {x: {} for x in range(1000)}
        self.assertEqual(
            expected,
            retworkx.graph_all_pairs_dijkstra_shortest_paths(graph, float),
        )
