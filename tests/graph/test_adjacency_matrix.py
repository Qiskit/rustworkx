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
import numpy as np


class TestGraphAdjacencyMatrix(unittest.TestCase):
    def test_single_neighbor(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edge_a")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "edge_b")
        res = rustworkx.graph_adjacency_matrix(graph, lambda x: 1)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(
            np.array_equal(
                np.array(
                    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
                res,
            )
        )

    def test_no_weight_fn(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edge_a")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "edge_b")
        res = rustworkx.graph_adjacency_matrix(graph)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(
            np.array_equal(
                np.array(
                    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
                res,
            )
        )

    def test_default_weight(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edge_a")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "edge_b")
        res = rustworkx.graph_adjacency_matrix(graph, default_weight=4)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(
            np.array_equal(
                np.array(
                    [[0.0, 4.0, 0.0], [4.0, 0.0, 4.0], [0.0, 4.0, 0.0]],
                    dtype=np.float64,
                ),
                res,
            )
        )

    def test_float_cast_weight_func(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 7.0)
        res = rustworkx.graph_adjacency_matrix(graph, float)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(np.array([[0.0, 7.0], [7.0, 0.0]]), res))

    def test_multigraph_sum_cast_weight_func(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 7.0)
        graph.add_edge(node_a, node_b, 0.5)
        res = rustworkx.graph_adjacency_matrix(graph, float)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(np.array([[0.0, 7.5], [7.5, 0.0]]), res))

    def test_multigraph_sum_cast_weight_func_non_zero_null(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 7.0)
        graph.add_edge(node_a, node_b, 0.5)
        res = rustworkx.graph_adjacency_matrix(graph, float, null_value=np.inf)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(np.array([[np.inf, 7.5], [7.5, np.inf]]), res))

    def test_dag_to_graph_adjacency_matrix(self):
        dag = rustworkx.PyDAG()
        self.assertRaises(TypeError, rustworkx.graph_adjacency_matrix, dag)

    def test_no_edge_graph_adjacency_matrix(self):
        graph = rustworkx.PyGraph()
        for i in range(50):
            graph.add_node(i)
        res = rustworkx.graph_adjacency_matrix(graph, lambda x: 1)
        self.assertTrue(np.array_equal(np.zeros([50, 50]), res))

    def test_graph_with_index_holes(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 1)
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, 1)
        graph.remove_node(node_b)
        res = rustworkx.graph_adjacency_matrix(graph, lambda x: 1)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(np.array([[0, 1], [1, 0]]), res))

    def test_from_adjacency_matrix(self):
        input_array = np.array(
            [[0.0, 4.0, 0.0], [4.0, 0.0, 4.0], [0.0, 4.0, 0.0]],
            dtype=np.float64,
        )
        graph = rustworkx.PyGraph.from_adjacency_matrix(input_array)
        out_array = rustworkx.graph_adjacency_matrix(graph, lambda x: x)
        self.assertTrue(np.array_equal(input_array, out_array))

    def test_random_graph_full_path(self):
        graph = rustworkx.undirected_gnp_random_graph(100, 0.95, seed=42)
        adjacency_matrix = rustworkx.graph_adjacency_matrix(graph)
        new_graph = rustworkx.PyGraph.from_adjacency_matrix(adjacency_matrix)
        new_adjacency_matrix = rustworkx.graph_adjacency_matrix(new_graph)
        self.assertTrue(np.array_equal(adjacency_matrix, new_adjacency_matrix))

    def test_random_graph_different_dtype(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int64)
        with self.assertRaises(TypeError):
            rustworkx.PyGraph.from_adjacency_matrix(input_matrix)

    def test_random_graph_different_dtype_astype_no_copy(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int64)
        graph = rustworkx.PyGraph.from_adjacency_matrix(input_matrix.astype(np.float64, copy=False))
        adj_matrix = rustworkx.graph_adjacency_matrix(graph, lambda x: x)
        self.assertTrue(np.array_equal(adj_matrix, input_matrix))

    def test_random_graph_float_dtype(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        graph = rustworkx.PyGraph.from_adjacency_matrix(input_matrix)
        adj_matrix = rustworkx.graph_adjacency_matrix(graph, lambda x: x)
        self.assertTrue(np.array_equal(adj_matrix, input_matrix))

    def test_graph_to_digraph_adjacency_matrix(self):
        graph = rustworkx.PyGraph()
        self.assertRaises(TypeError, rustworkx.digraph_adjacency_matrix, graph)

    def test_non_zero_null(self):
        input_matrix = np.array(
            [[np.inf, 1, np.inf], [1, np.inf, 1], [np.inf, 1, np.inf]],
            dtype=np.float64,
        )
        graph = rustworkx.PyGraph.from_adjacency_matrix(input_matrix, null_value=np.inf)
        adj_matrix = rustworkx.adjacency_matrix(graph, float)
        expected_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
        self.assertTrue(np.array_equal(adj_matrix, expected_matrix))

    def test_negative_weight(self):
        input_matrix = np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]], dtype=float)
        graph = rustworkx.PyGraph.from_adjacency_matrix(input_matrix)
        adj_matrix = rustworkx.graph_adjacency_matrix(graph, lambda x: x)
        self.assertTrue(np.array_equal(adj_matrix, input_matrix))
        self.assertEqual([(0, 1, -1), (1, 2, -1)], graph.weighted_edge_list())

    def test_nan_null(self):
        input_matrix = np.array(
            [[np.nan, 1, np.nan], [1, np.nan, 1], [np.nan, 1, np.nan]],
            dtype=np.float64,
        )
        graph = rustworkx.PyGraph.from_adjacency_matrix(input_matrix, null_value=np.nan)
        adj_matrix = rustworkx.adjacency_matrix(graph, float)
        expected_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
        self.assertTrue(np.array_equal(adj_matrix, expected_matrix))


class TestFromComplexAdjacencyMatrix(unittest.TestCase):
    def test_from_adjacency_matrix(self):
        input_array = np.array(
            [[0.0, 4.0, 0.0], [4.0, 0.0, 4.0], [0.0, 4.0, 0.0]],
            dtype=np.complex128,
        )
        graph = rustworkx.PyGraph.from_complex_adjacency_matrix(input_array)
        expected = [
            (0, 1, 4 + 0j),
            (1, 2, 4 + 0j),
        ]
        self.assertEqual(graph.weighted_edge_list(), expected)

    def test_random_graph_different_dtype(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int64)
        with self.assertRaises(TypeError):
            rustworkx.PyGraph.from_complex_adjacency_matrix(input_matrix)

    def test_random_graph_different_dtype_astype_no_copy(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int64)
        graph = rustworkx.PyGraph.from_complex_adjacency_matrix(
            input_matrix.astype(np.complex128, copy=False)
        )
        expected = [
            (0, 1, 1 + 0j),
            (1, 2, 1 + 0j),
        ]
        self.assertEqual(graph.weighted_edge_list(), expected)

    def test_random_graph_complex_dtype(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        graph = rustworkx.PyGraph.from_complex_adjacency_matrix(input_matrix)
        expected = [
            (0, 1, 1 + 0j),
            (1, 2, 1 + 0j),
        ]
        self.assertEqual(graph.weighted_edge_list(), expected)

    def test_non_zero_null(self):
        input_matrix = np.array(
            [[np.inf, 1, np.inf], [1, np.inf, 1], [np.inf, 1, np.inf]],
            dtype=np.complex128,
        )
        graph = rustworkx.PyGraph.from_complex_adjacency_matrix(input_matrix, null_value=np.inf)
        expected = [
            (0, 1, 1 + 0j),
            (1, 2, 1 + 0j),
        ]
        self.assertEqual(graph.weighted_edge_list(), expected)

    def test_negative_weight(self):
        input_matrix = np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]], dtype=complex)
        graph = rustworkx.PyGraph.from_complex_adjacency_matrix(input_matrix)
        self.assertEqual(
            [(0, 1, 1), (1, 2, -1)],
            graph.weighted_edge_list(),
        )

    def test_nan_null(self):
        input_matrix = np.array(
            [[np.nan, 1, np.nan], [1, np.nan, 1], [np.nan, 1, np.nan]],
            dtype=np.complex128,
        )
        graph = rustworkx.PyGraph.from_complex_adjacency_matrix(input_matrix, null_value=np.nan)
        edge_list = graph.weighted_edge_list()
        self.assertEqual(
            edge_list,
            [(0, 1, 1 + 0j), (1, 2, 1 + 0j)],
        )

    def test_parallel_edge(self):
        graph = rustworkx.PyGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")

        graph.add_edges_from(
            [
                (a, b, 3.0),
                (a, b, 1.0),
                (a, c, 2.0),
                (b, c, 7.0),
                (c, a, 1.0),
                (b, c, 2.0),
                (a, b, 4.0),
            ]
        )

        min_matrix = rustworkx.graph_adjacency_matrix(graph, weight_fn=float, parallel_edge="min")
        np.testing.assert_array_equal(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0]], min_matrix
        )

        max_matrix = rustworkx.graph_adjacency_matrix(graph, weight_fn=float, parallel_edge="max")
        np.testing.assert_array_equal(
            [[0.0, 4.0, 2.0], [4.0, 0.0, 7.0], [2.0, 7.0, 0.0]], max_matrix
        )

        avg_matrix = rustworkx.graph_adjacency_matrix(graph, weight_fn=float, parallel_edge="avg")
        np.testing.assert_array_equal(
            [[0.0, 8 / 3.0, 1.5], [8 / 3.0, 0.0, 4.5], [1.5, 4.5, 0.0]], avg_matrix
        )

        sum_matrix = rustworkx.graph_adjacency_matrix(graph, weight_fn=float, parallel_edge="sum")
        np.testing.assert_array_equal(
            [[0.0, 8.0, 3.0], [8.0, 0.0, 9.0], [3.0, 9.0, 0.0]], sum_matrix
        )

        with self.assertRaises(ValueError):
            rustworkx.graph_adjacency_matrix(graph, weight_fn=float, parallel_edge="error")
