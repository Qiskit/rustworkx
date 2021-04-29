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

import numpy


class TestDispatchPyGraph(unittest.TestCase):

    class_type = "PyGraph"

    def setUp(self):
        super().setUp()
        if self.class_type == "PyGraph":
            self.graph = retworkx.undirected_gnp_random_graph(10, .5, seed=42)
        else:
            self.graph = retworkx.directed_gnp_random_graph(10, .5, seed=42)

    def test_distance_matrix(self):
        res = retworkx.distance_matrix(self.graph)
        self.assertIsInstance(res, numpy.ndarray)

    def test_distance_matrix_as_undirected(self):
        if self.class_type == "PyGraph":
            with self.assertRaises(TypeError):
                retworkx.distance_matrix(self.graph, as_undirected=True)
        else:
            res = retworkx.distance_matrix(self.graph, as_undirected=True)
            self.assertIsInstance(res, numpy.ndarray)

    def test_adjacency_matrix(self):
        res = retworkx.adjacency_matrix(self.graph)
        self.assertIsInstance(res, numpy.ndarray)

    def test_all_simple_paths(self):
        res = retworkx.all_simple_paths(self.graph, 0, 1)
        self.assertIsInstance(res, list)

    def test_floyd_warshall_numpy(self):
        res = retworkx.floyd_warshall_numpy(self.graph)
        self.assertIsInstance(res, numpy.ndarray)

        if self.class_type == "PyGraph":
            expected_res = retworkx.graph_floyd_warshall_numpy(self.graph)
        else:
            expected_res = retworkx.digraph_floyd_warshall_numpy(self.graph)

        self.assertTrue(numpy.array_equal(expected_res, res))

    def test_astar_shortest_path(self):
        res = retworkx.astar_shortest_path(self.graph, 0, lambda _: True,
                                           lambda _: 1, lambda _: 1)
        self.assertIsInstance(list(res), list)

    def test_dijkstra_shortest_paths(self):
        res = retworkx.dijkstra_shortest_paths(self.graph, 0)
        self.assertIsInstance(res, dict)

    def test_dijkstra_shortest_path_lengths(self):
        res = retworkx.dijkstra_shortest_path_lengths(self.graph, 0,
                                                      lambda _: 1)
        self.assertIsInstance(res, dict)

    def test_k_shortest_path_lengths(self):
        res = retworkx.k_shortest_path_lengths(self.graph, 0, 2, lambda _: 1)
        self.assertIsInstance(res, dict)

    def test_dfs_edges(self):
        res = retworkx.dfs_edges(self.graph, 0)
        self.assertIsInstance(list(res), list)

    def test_is_isomorphic_nodes_incompatible_raises(self):
        with self.assertRaises(TypeError):
            if self.class_type == "PyGraph":
                retworkx.is_isomorphic(self.graph, retworkx.PyDiGraph())
            else:
                retworkx.is_isomorphic(self.graph, retworkx.PyGraph())


class TestDispatchPyDiGraph(TestDispatchPyGraph):

    class_type = "PyDiGraph"
