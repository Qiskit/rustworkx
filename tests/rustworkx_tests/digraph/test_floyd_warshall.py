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

    def test_floyd_warshall(self):
        """Test the algorithm on a 5q x 4 depth circuit."""
        dag = rustworkx.PyDAG()
        # inputs
        qr_0 = dag.add_node("qr[0]")
        qr_1 = dag.add_node("qr[1]")
        qr_2 = dag.add_node("qr[2]")
        cr_0 = dag.add_node("cr[0]")
        cr_1 = dag.add_node("cr[1]")
        # wires
        cx_1 = dag.add_node("cx_1")
        dag.add_edge(qr_0, cx_1, "qr[0]")
        dag.add_edge(qr_1, cx_1, "qr[1]")
        h_1 = dag.add_node("h_1")
        dag.add_edge(cx_1, h_1, "qr[0]")
        cx_2 = dag.add_node("cx_2")
        dag.add_edge(cx_1, cx_2, "qr[1]")
        dag.add_edge(qr_2, cx_2, "qr[2]")
        cx_3 = dag.add_node("cx_3")
        dag.add_edge(h_1, cx_3, "qr[0]")
        dag.add_edge(cx_2, cx_3, "qr[2]")
        h_2 = dag.add_node("h_2")
        dag.add_edge(cx_3, h_2, "qr[2]")
        # # outputs
        qr_0_out = dag.add_node("qr[0]_out")
        dag.add_edge(cx_3, qr_0_out, "qr[0]")
        qr_1_out = dag.add_node("qr[1]_out")
        dag.add_edge(cx_2, qr_1_out, "qr[1]")
        qr_2_out = dag.add_node("qr[2]_out")
        dag.add_edge(h_2, qr_2_out, "qr[2]")
        cr_0_out = dag.add_node("cr[0]_out")
        dag.add_edge(cr_0, cr_0_out, "qr[2]")
        cr_1_out = dag.add_node("cr[1]_out")
        dag.add_edge(cr_1, cr_1_out, "cr[1]")

        result = rustworkx.floyd_warshall(dag)
        expected = {
            0: {0: 0, 5: 1, 6: 2, 7: 2, 8: 3, 9: 4, 10: 4, 11: 3, 12: 5},
            1: {1: 0, 5: 1, 6: 2, 7: 2, 8: 3, 9: 4, 10: 4, 11: 3, 12: 5},
            2: {2: 0, 7: 1, 8: 2, 9: 3, 10: 3, 11: 2, 12: 4},
            3: {3: 0, 13: 1},
            4: {4: 0, 14: 1},
            5: {5: 0, 6: 1, 7: 1, 8: 2, 9: 3, 10: 3, 11: 2, 12: 4},
            6: {6: 0, 8: 1, 9: 2, 10: 2, 12: 3},
            7: {7: 0, 8: 1, 9: 2, 10: 2, 11: 1, 12: 3},
            8: {8: 0, 9: 1, 10: 1, 12: 2},
            9: {9: 0, 12: 1},
            10: {10: 0},
            11: {11: 0},
            12: {12: 0},
            13: {13: 0},
            14: {14: 0},
        }

        self.assertEqual(result, expected)

    def test_vs_dijkstra_all_pairs(self):
        graph = rustworkx.PyDiGraph()
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

        dijkstra_lengths = rustworkx.digraph_all_pairs_dijkstra_path_lengths(graph, float)

        expected = {k: {**v, k: 0.0} for k, v in dijkstra_lengths.items()}

        result = rustworkx.digraph_floyd_warshall(
            graph, float, parallel_threshold=self.parallel_threshold
        )

        self.assertEqual(result, expected)

    def test_vs_dijkstra_all_pairs_with_node_removal(self):
        graph = rustworkx.PyDiGraph()
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

        dijkstra_lengths = rustworkx.digraph_all_pairs_dijkstra_path_lengths(graph, float)

        expected = {k: {**v, k: 0.0} for k, v in dijkstra_lengths.items()}

        result = rustworkx.digraph_floyd_warshall(
            graph, float, parallel_threshold=self.parallel_threshold
        )

        self.assertEqual(result, expected)

    def test_floyd_warshall_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        self.assertEqual({}, rustworkx.digraph_floyd_warshall(graph, float))

    def test_floyd_warshall_graph_no_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(1000)))
        expected = {x: {} for x in range(1000)}
        self.assertEqual(
            expected,
            rustworkx.digraph_floyd_warshall(graph, float),
        )

    def test_directed_floyd_warshall_cycle_as_undirected(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = rustworkx.digraph_floyd_warshall(
            graph,
            lambda _: 1,
            as_undirected=True,
            parallel_threshold=self.parallel_threshold,
        )
        expected = {
            0: {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 3.0, 5: 2.0, 6: 1.0},
            1: {0: 1.0, 1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 3.0, 6: 2.0},
            2: {0: 2.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 2.0, 5: 3.0, 6: 3.0},
            3: {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 2.0, 6: 3.0},
            4: {0: 3.0, 1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 2.0},
            5: {0: 2.0, 1: 3.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 0.0, 6: 1.0},
            6: {0: 1.0, 1: 2.0, 2: 3.0, 3: 3.0, 4: 2.0, 5: 1.0, 6: 0.0},
        }
        self.assertEqual(dist, expected)

    def test_directed_floyd_warshall_numpy_cycle_as_undirected(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = rustworkx.digraph_floyd_warshall_numpy(graph, lambda x: 1, as_undirected=True)
        expected = numpy.array(
            [
                [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
                [1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 2.0],
                [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0],
                [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0],
                [3.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0],
                [2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 1.0],
                [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0],
            ]
        )
        self.assertTrue(numpy.array_equal(dist, expected))

    def test_floyd_warshall_numpy_digraph_three_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(6)))
        weights = [2, 12, 1, 5, 1]
        graph.add_edges_from([(i, i + 1, weights[i]) for i in range(5)])
        graph.add_edge(5, 0, 10)
        dist = rustworkx.digraph_floyd_warshall_numpy(
            graph, lambda x: x, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 3], 15)
        self.assertEqual(dist[3, 0], 16)

    def test_weighted_numpy_digraph_two_edges(self):
        graph = rustworkx.PyDiGraph()
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
        dist = rustworkx.digraph_floyd_warshall_numpy(
            graph, lambda x: x, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 2], 4)
        self.assertEqual(dist[2, 0], 6)

    def test_floyd_warshall_numpy_digraph_cycle(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = rustworkx.digraph_floyd_warshall_numpy(
            graph, lambda x: 1, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 3], 3)
        self.assertEqual(dist[0, 4], 4)

    def test_weighted_numpy_directed_negative_cycle(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.add_edges_from(
            [
                (0, 1, 1),
                (1, 2, -1),
                (2, 3, -1),
                (3, 0, -1),
            ]
        )
        dist = rustworkx.digraph_floyd_warshall_numpy(graph, lambda x: x)
        self.assertTrue(numpy.all(numpy.diag(dist) < 0))

    def test_numpy_directed_no_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        dist = rustworkx.digraph_floyd_warshall_numpy(
            graph, lambda x: x, parallel_threshold=self.parallel_threshold
        )
        expected = numpy.full((4, 4), numpy.inf)
        numpy.fill_diagonal(expected, 0)
        self.assertTrue(numpy.array_equal(dist, expected))

    def test_floyd_warshall_numpy_digraph_cycle_with_removals(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(8)))
        graph.remove_node(0)
        graph.add_edges_from_no_data([(1, 2), (1, 7), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
        dist = rustworkx.digraph_floyd_warshall_numpy(
            graph, lambda x: 1, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 3], 3)
        self.assertEqual(dist[0, 4], 4)

    def test_floyd_warshall_numpy_digraph_cycle_no_weight_fn(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(8)))
        graph.remove_node(0)
        graph.add_edges_from_no_data([(1, 2), (1, 7), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
        dist = rustworkx.digraph_floyd_warshall_numpy(graph)
        self.assertEqual(dist[0, 3], 3)
        self.assertEqual(dist[0, 4], 4)

    def test_floyd_warshall_numpy_digraph_cycle_default_weight(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(8)))
        graph.remove_node(0)
        graph.add_edges_from_no_data([(1, 2), (1, 7), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
        dist = rustworkx.digraph_floyd_warshall_numpy(
            graph, default_weight=2, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(dist[0, 3], 6)
        self.assertEqual(dist[0, 4], 8)

    def test_floyd_warshall_successors_numpy(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(9)))
        graph.add_edges_from_no_data(
            [(1, 2), (1, 7), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 8)]
        )
        dist, succ = rustworkx.digraph_floyd_warshall_successor_and_distance_numpy(
            graph, default_weight=2, parallel_threshold=self.parallel_threshold
        )
        self.assertEqual(succ[1, 1], 1)
        self.assertEqual(succ[1, 4], 2)
        self.assertEqual(succ[1, 6], 2)
        self.assertEqual(succ[1, 7], 7)
        self.assertEqual(succ[1, 8], 8)
        self.assertEqual(succ[0, 8], 8)


class TestParallelFloydWarshall(TestFloydWarshall):
    parallel_threshold = 0
