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

import numpy

import rustworkx


class TestFloydWarshall(unittest.TestCase):
    parallel_threshold = 300

    def test_vs_dijkstra_all_pairs(self):
        graph = rustworkx.PyGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")
        d = graph.add_node("D")
        e = graph.add_node("E")
        f = graph.add_node("F")
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
        graph.add_edges_from(edge_list)

        dijkstra_lengths = rustworkx.graph_all_pairs_dijkstra_path_lengths(graph, float)

        expected = {k: {**v, k: 0.0} for k, v in dijkstra_lengths.items()}

        result = rustworkx.graph_floyd_warshall(
            graph, float, parallel_threshold=self.parallel_threshold
        )

        self.assertEqual(result, expected)

    def test_vs_dijkstra_all_pairs_with_node_removal(self):
        graph = rustworkx.PyGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")
        d = graph.add_node("D")
        e = graph.add_node("E")
        f = graph.add_node("F")
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
        graph.add_edges_from(edge_list)
        graph.remove_node(d)

        dijkstra_lengths = rustworkx.graph_all_pairs_dijkstra_path_lengths(graph, float)

        expected = {k: {**v, k: 0.0} for k, v in dijkstra_lengths.items()}

        result = rustworkx.graph_floyd_warshall(
            graph, float, parallel_threshold=self.parallel_threshold
        )

        self.assertEqual(result, expected)

    def test_floyd_warshall_empty_graph(self):
        graph = rustworkx.PyGraph()
        self.assertEqual({}, rustworkx.graph_floyd_warshall(graph, float))

    def test_floyd_warshall_graph_no_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(1000)))
        expected = {x: {} for x in range(1000)}
        self.assertEqual(
            expected,
            rustworkx.graph_floyd_warshall(graph, float),
        )

    def test_floyd_warshall_numpy_three_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(6)))
        weights = [2, 12, 1, 5, 1]
        graph.add_edges_from([(i, i + 1, weights[i]) for i in range(5)])
        graph.add_edge(5, 0, 10)
        dist = rustworkx.graph_floyd_warshall_numpy(
            graph, lambda x: x, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 3], 15)
        self.assertEqual(dist[3, 0], 15)

    def test_weighted_numpy_two_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(8)))
        graph.add_edges_from(
            [
                (0, 1, 2),
                (1, 2, 2),
                (2, 3, 1),
                (3, 4, 1),
                (4, 5, 1),
                (5, 6, 1),
                (6, 7, 1),
                (7, 0, 1),
            ]
        )
        dist = rustworkx.graph_floyd_warshall_numpy(
            graph, lambda x: x, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 2], 4)
        self.assertEqual(dist[2, 0], 4)

    def test_weighted_numpy_negative_cycle(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(4)))
        graph.add_edges_from(
            [
                (0, 1, 1),
                (1, 2, -1),
                (2, 3, -1),
                (3, 0, -1),
            ]
        )
        dist = rustworkx.graph_floyd_warshall_numpy(
            graph, lambda x: x, parallel_threshold=self.parallel_threshold
        )
        self.assertTrue(numpy.all(numpy.diag(dist) < 0))

    def test_floyd_warshall_numpy_cycle(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = rustworkx.graph_floyd_warshall_numpy(
            graph, lambda x: 1, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 3], 3)
        self.assertEqual(dist[0, 4], 3)

    def test_numpy_no_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(4)))
        dist = rustworkx.graph_floyd_warshall_numpy(
            graph, lambda x: x, parallel_threshold=self.parallel_threshold
        )
        expected = numpy.full((4, 4), numpy.inf)
        numpy.fill_diagonal(expected, 0)
        self.assertTrue(numpy.array_equal(dist, expected))

    def test_floyd_warshall_numpy_graph_cycle_with_removals(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(8)))
        graph.remove_node(0)
        graph.add_edges_from_no_data([(1, 2), (1, 7), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
        dist = rustworkx.graph_floyd_warshall_numpy(
            graph, lambda x: 1, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 3], 3)
        self.assertEqual(dist[0, 4], 3)

    def test_floyd_warshall_numpy_graph_cycle_no_weight_fn(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(8)))
        graph.remove_node(0)
        graph.add_edges_from_no_data([(1, 2), (1, 7), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
        dist = rustworkx.graph_floyd_warshall_numpy(graph)
        self.assertEqual(dist[0, 3], 3)
        self.assertEqual(dist[0, 4], 3)

    def test_floyd_warshall_numpy_graph_cycle_default_weight(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(8)))
        graph.remove_node(0)
        graph.add_edges_from_no_data([(1, 2), (1, 7), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
        dist = rustworkx.graph_floyd_warshall_numpy(
            graph, default_weight=2, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 3], 6)
        self.assertEqual(dist[0, 4], 6)

    def test_floyd_warshall_successors_numpy(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(9)))
        graph.add_edges_from_no_data(
            [(1, 2), (1, 7), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 8)]
        )
        dist, succ = rustworkx.graph_floyd_warshall_successor_and_distance_numpy(
            graph, default_weight=2, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(succ[1, 1], 1)
        self.assertEqual(succ[1, 4], 2)
        self.assertEqual(succ[1, 6], 7)
        self.assertEqual(succ[1, 7], 7)
        self.assertEqual(succ[1, 8], 8)
        self.assertEqual(succ[0, 8], 8)


class TestParallelFloydWarshall(TestFloydWarshall):
    parallel_threshold = 0
