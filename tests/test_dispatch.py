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

import numpy


class TestDispatchPyGraph(unittest.TestCase):

    class_type = "PyGraph"

    def setUp(self):
        super().setUp()
        if self.class_type == "PyGraph":
            self.graph = rustworkx.undirected_gnp_random_graph(10, 0.5, seed=42)
        else:
            self.graph = rustworkx.directed_gnp_random_graph(10, 0.5, seed=42)

    def test_distance_matrix(self):
        res = rustworkx.distance_matrix(self.graph)
        self.assertIsInstance(res, numpy.ndarray)

    def test_distance_matrix_as_undirected(self):
        if self.class_type == "PyGraph":
            with self.assertRaises(TypeError):
                rustworkx.distance_matrix(self.graph, as_undirected=True)
        else:
            res = rustworkx.distance_matrix(self.graph, as_undirected=True)
            self.assertIsInstance(res, numpy.ndarray)

    def test_adjacency_matrix(self):
        res = rustworkx.adjacency_matrix(self.graph)
        self.assertIsInstance(res, numpy.ndarray)

    def test_all_simple_paths(self):
        res = rustworkx.all_simple_paths(self.graph, 0, 1)
        self.assertIsInstance(res, list)

    def test_floyd_warshall(self):
        res = rustworkx.floyd_warshall(self.graph)
        self.assertIsInstance(res, rustworkx.AllPairsPathLengthMapping)

    def test_floyd_warshall_numpy(self):
        res = rustworkx.floyd_warshall_numpy(self.graph)
        self.assertIsInstance(res, numpy.ndarray)

        if self.class_type == "PyGraph":
            expected_res = rustworkx.graph_floyd_warshall_numpy(self.graph)
        else:
            expected_res = rustworkx.digraph_floyd_warshall_numpy(self.graph)

        self.assertTrue(numpy.array_equal(expected_res, res))

    def test_astar_shortest_path(self):
        res = rustworkx.astar_shortest_path(self.graph, 0, lambda _: True, lambda _: 1, lambda _: 1)
        self.assertIsInstance(list(res), list)

    def test_dijkstra_shortest_paths(self):
        res = rustworkx.dijkstra_shortest_paths(self.graph, 0)
        self.assertIsInstance(res, rustworkx.PathMapping)

    def test_dijkstra_shortest_path_lengths(self):
        res = rustworkx.dijkstra_shortest_path_lengths(self.graph, 0, lambda _: 1)
        self.assertIsInstance(res, rustworkx.PathLengthMapping)

    def test_k_shortest_path_lengths(self):
        res = rustworkx.k_shortest_path_lengths(self.graph, 0, 2, lambda _: 1)
        self.assertIsInstance(res, rustworkx.PathLengthMapping)

    def test_dfs_edges(self):
        res = rustworkx.dfs_edges(self.graph, 0)
        self.assertIsInstance(list(res), list)

    def test_all_pairs_dijkstra_shortest_paths(self):
        res = rustworkx.all_pairs_dijkstra_shortest_paths(self.graph, lambda _: 1)
        self.assertIsInstance(res, rustworkx.AllPairsPathMapping)

    def test_all_pairs_dijkstra_path_lengths(self):
        res = rustworkx.all_pairs_dijkstra_path_lengths(self.graph, lambda _: 1)
        self.assertIsInstance(res, rustworkx.AllPairsPathLengthMapping)

    def test_is_isomorphic_nodes_incompatible_raises(self):
        with self.assertRaises(TypeError):
            if self.class_type == "PyGraph":
                rustworkx.is_isomorphic(self.graph, rustworkx.PyDiGraph())
            else:
                rustworkx.is_isomorphic(self.graph, rustworkx.PyGraph())

    def test_betweenness_centrality(self):
        res = rustworkx.betweenness_centrality(self.graph)
        self.assertIsInstance(res, rustworkx.CentralityMapping)


class TestDispatchPyDiGraph(TestDispatchPyGraph):

    class_type = "PyDiGraph"
