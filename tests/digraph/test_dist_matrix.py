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

import numpy as np

import rustworkx


class TestDistanceMatrix(unittest.TestCase):
    def test_digraph_distance_matrix(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = rustworkx.digraph_distance_matrix(graph)
        expected = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
                [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(np.array_equal(dist, expected))

    def test_digraph_distance_matrix_parallel(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = rustworkx.digraph_distance_matrix(graph, parallel_threshold=5)
        expected = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
                [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(np.array_equal(dist, expected))

    def test_digraph_distance_matrix_as_undirected(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = rustworkx.digraph_distance_matrix(graph, as_undirected=True)
        expected = np.array(
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
        self.assertTrue(np.array_equal(dist, expected))

    def test_digraph_distance_matrix_parallel_as_undirected(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = rustworkx.digraph_distance_matrix(graph, parallel_threshold=5, as_undirected=True)
        expected = np.array(
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
        self.assertTrue(np.array_equal(dist, expected))

    def test_digraph_distance_matrix_non_zero_null(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        graph.add_node(7)
        dist = rustworkx.distance_matrix(graph, as_undirected=True, null_value=np.nan)
        expected = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, np.nan],
                [1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 2.0, np.nan],
                [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, np.nan],
                [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, np.nan],
                [3.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, np.nan],
                [2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 1.0, np.nan],
                [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0],
            ]
        )
        self.assertTrue(np.array_equal(dist, expected, equal_nan=True))

    def test_digraph_distance_matrix_parallel_non_zero_null(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        graph.add_node(7)
        dist = rustworkx.distance_matrix(
            graph, as_undirected=True, parallel_threshold=5, null_value=np.nan
        )
        expected = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, np.nan],
                [1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 2.0, np.nan],
                [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, np.nan],
                [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, np.nan],
                [3.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, np.nan],
                [2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 1.0, np.nan],
                [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0],
            ]
        )
        self.assertTrue(np.array_equal(dist, expected, equal_nan=True))

    def test_digraph_distance_matrix_node_hole(self):
        graph = rustworkx.generators.directed_path_graph(4)
        graph.remove_node(0)
        dist = rustworkx.digraph_distance_matrix(graph)
        expected = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        self.assertTrue(np.array_equal(dist, expected))
