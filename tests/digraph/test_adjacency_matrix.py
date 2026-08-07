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

try:
    import scipy.sparse as sp
except ModuleNotFoundError:
    sp = None


class TestDAGAdjacencyMatrix(unittest.TestCase):
    def test_single_neighbor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"a": 1})
        dag.add_child(node_a, "c", {"a": 2})
        res = rustworkx.digraph_adjacency_matrix(dag, lambda x: 1)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(
            np.array_equal(
                np.array(
                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                res,
            )
        )

    def test_no_weight_fn(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"a": 1})
        dag.add_child(node_a, "c", {"a": 2})
        res = rustworkx.digraph_adjacency_matrix(dag)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(
            np.array_equal(
                np.array(
                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                res,
            )
        )

    def test_default_weight(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"a": 1})
        dag.add_child(node_a, "c", {"a": 2})
        res = rustworkx.digraph_adjacency_matrix(dag, default_weight=4)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(
            np.array_equal(
                np.array(
                    [[0.0, 4.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                res,
            )
        )

    def test_float_cast_weight_func(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", 7.0)
        res = rustworkx.digraph_adjacency_matrix(dag, lambda x: float(x))
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(np.array([[0.0, 7.0], [0.0, 0.0]]), res))

    def test_multigraph_sum_cast_weight_func(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", 7.0)
        dag.add_edge(node_a, node_b, 0.5)
        res = rustworkx.digraph_adjacency_matrix(dag, lambda x: float(x))
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(np.array([[0.0, 7.5], [0.0, 0.0]]), res))

    def test_multigraph_sum_cast_weight_func_non_zero_null(self):
        graph = rustworkx.PyDiGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 7.0)
        graph.add_edge(node_a, node_b, 0.5)
        res = rustworkx.adjacency_matrix(graph, lambda x: float(x), null_value=np.inf)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(np.array([[np.inf, 7.5], [np.inf, np.inf]]), res))

    def test_graph_to_digraph_adjacency_matrix(self):
        graph = rustworkx.PyGraph()
        self.assertRaises(TypeError, rustworkx.digraph_adjacency_matrix, graph)

    def test_no_edge_digraph_adjacency_matrix(self):
        dag = rustworkx.PyDAG()
        for i in range(50):
            dag.add_node(i)
        res = rustworkx.digraph_adjacency_matrix(dag, lambda x: 1)
        self.assertTrue(np.array_equal(np.zeros([50, 50]), res))

    def test_digraph_with_index_holes(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", 1)
        dag.add_child(node_a, "c", 1)
        dag.remove_node(node_b)
        res = rustworkx.digraph_adjacency_matrix(dag, lambda x: 1)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(np.array([[0, 1], [0, 0]]), res))

    def test_node_list_with_index_holes(self):
        graph = rustworkx.PyDiGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        graph.add_edge(node_a, node_b, 1.0)
        graph.add_edge(node_b, node_c, 2.0)
        graph.add_edge(node_c, node_d, 3.0)
        graph.add_edge(node_a, node_d, 4.0)
        graph.remove_node(node_b)

        res = rustworkx.digraph_adjacency_matrix(graph, lambda x: float(x))

        self.assertTrue(
            np.array_equal(
                np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]),
                res,
            )
        )

        res = rustworkx.digraph_adjacency_matrix(
            graph, lambda x: float(x), node_list=[node_d, node_a, node_c]
        )

        self.assertTrue(
            np.array_equal(
                np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
                res,
            )
        )

    def test_node_list_order_and_subset(self):
        graph = rustworkx.PyDiGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_b, 1.0)
        graph.add_edge(node_b, node_c, 2.0)
        graph.add_edge(node_a, node_c, 3.0)

        res = rustworkx.digraph_adjacency_matrix(graph, lambda x: float(x), node_list=[2, 0, 1])

        self.assertTrue(
            np.array_equal(
                np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0], [2.0, 0.0, 0.0]]),
                res,
            )
        )

        res = rustworkx.adjacency_matrix(graph, lambda x: float(x), 1.0, 0.0, node_list=[2, 0])

        self.assertTrue(np.array_equal(np.array([[0.0, 0.0], [3.0, 0.0]]), res))

    def test_node_list_errors(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node("a")

        with self.assertRaises(rustworkx.InvalidNode):
            rustworkx.digraph_adjacency_matrix(graph, node_list=[0, 1])

        with self.assertRaises(ValueError):
            rustworkx.digraph_adjacency_matrix(graph, node_list=[0, 0])

    def test_from_adjacency_matrix(self):
        input_array = np.array(
            [[0.0, 4.0, 0.0], [4.0, 0.0, 4.0], [0.0, 4.0, 0.0]],
            dtype=np.float64,
        )
        graph = rustworkx.PyDiGraph.from_adjacency_matrix(input_array)
        out_array = rustworkx.digraph_adjacency_matrix(graph, lambda x: x)
        self.assertTrue(np.array_equal(input_array, out_array))

    def test_random_graph_full_path(self):
        graph = rustworkx.directed_gnp_random_graph(100, 0.95, seed=42)
        adjacency_matrix = rustworkx.digraph_adjacency_matrix(graph)
        new_graph = rustworkx.PyDiGraph.from_adjacency_matrix(adjacency_matrix)
        new_adjacency_matrix = rustworkx.digraph_adjacency_matrix(new_graph)
        self.assertTrue(np.array_equal(adjacency_matrix, new_adjacency_matrix))

    def test_random_graph_different_dtype(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int64)
        with self.assertRaises(TypeError):
            rustworkx.PyDiGraph.from_adjacency_matrix(input_matrix)

    def test_random_graph_different_dtype_astype_no_copy(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int64)
        graph = rustworkx.PyDiGraph.from_adjacency_matrix(
            input_matrix.astype(np.float64, copy=False)
        )
        adj_matrix = rustworkx.digraph_adjacency_matrix(graph, lambda x: x)
        self.assertTrue(np.array_equal(adj_matrix, input_matrix))

    def test_random_graph_float_dtype(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        graph = rustworkx.PyDiGraph.from_adjacency_matrix(input_matrix)
        adj_matrix = rustworkx.digraph_adjacency_matrix(graph, lambda x: x)
        self.assertTrue(np.array_equal(adj_matrix, input_matrix))

    def test_non_zero_null(self):
        input_matrix = np.array(
            [[np.inf, 1, np.inf], [1, np.inf, 1], [np.inf, 1, np.inf]],
            dtype=np.float64,
        )
        graph = rustworkx.PyDiGraph.from_adjacency_matrix(input_matrix, null_value=np.inf)
        adj_matrix = rustworkx.adjacency_matrix(graph, float)
        expected_matrix = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        self.assertTrue(np.array_equal(adj_matrix, expected_matrix))

    def test_negative_weight(self):
        input_matrix = np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]], dtype=float)
        graph = rustworkx.PyDiGraph.from_adjacency_matrix(input_matrix)
        adj_matrix = rustworkx.digraph_adjacency_matrix(graph, lambda x: x)
        self.assertTrue(np.array_equal(adj_matrix, input_matrix))
        self.assertEqual(
            [(0, 1, 1), (1, 0, -1), (1, 2, -1), (2, 1, 1)],
            graph.weighted_edge_list(),
        )

    def test_nan_null(self):
        input_matrix = np.array(
            [[np.nan, 1, np.nan], [1, np.nan, 1], [np.nan, 1, np.nan]],
            dtype=np.float64,
        )
        graph = rustworkx.PyDiGraph.from_adjacency_matrix(input_matrix, null_value=np.nan)
        adj_matrix = rustworkx.adjacency_matrix(graph, float)
        expected_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
        self.assertTrue(np.array_equal(adj_matrix, expected_matrix))


class TestFromComplexAdjacencyMatrix(unittest.TestCase):
    def test_from_adjacency_matrix(self):
        input_array = np.array(
            [[0.0, 4.0, 0.0], [4.0, 0.0, 4.0], [0.0, 4.0, 0.0]],
            dtype=np.complex128,
        )
        graph = rustworkx.PyDiGraph.from_complex_adjacency_matrix(input_array)
        expected = [
            (0, 1, 4 + 0j),
            (1, 0, 4 + 0j),
            (1, 2, 4 + 0j),
            (2, 1, 4 + 0j),
        ]
        self.assertEqual(graph.weighted_edge_list(), expected)

    def test_random_graph_different_dtype(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int64)
        with self.assertRaises(TypeError):
            rustworkx.PyDiGraph.from_complex_adjacency_matrix(input_matrix)

    def test_random_graph_different_dtype_astype_no_copy(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int64)
        graph = rustworkx.PyDiGraph.from_complex_adjacency_matrix(
            input_matrix.astype(np.complex128, copy=False)
        )
        expected = [
            (0, 1, 1 + 0j),
            (1, 0, 1 + 0j),
            (1, 2, 1 + 0j),
            (2, 1, 1 + 0j),
        ]
        self.assertEqual(graph.weighted_edge_list(), expected)

    def test_random_graph_complex_dtype(self):
        input_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        graph = rustworkx.PyDiGraph.from_complex_adjacency_matrix(input_matrix)
        expected = [
            (0, 1, 1 + 0j),
            (1, 0, 1 + 0j),
            (1, 2, 1 + 0j),
            (2, 1, 1 + 0j),
        ]
        self.assertEqual(graph.weighted_edge_list(), expected)

    def test_non_zero_null(self):
        input_matrix = np.array(
            [[np.inf, 1, np.inf], [1, np.inf, 1], [np.inf, 1, np.inf]],
            dtype=np.complex128,
        )
        graph = rustworkx.PyDiGraph.from_complex_adjacency_matrix(input_matrix, null_value=np.inf)
        expected = [
            (0, 1, 1 + 0j),
            (1, 0, 1 + 0j),
            (1, 2, 1 + 0j),
            (2, 1, 1 + 0j),
        ]
        self.assertEqual(graph.weighted_edge_list(), expected)

    def test_negative_weight(self):
        input_matrix = np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]], dtype=complex)
        graph = rustworkx.PyDiGraph.from_complex_adjacency_matrix(input_matrix)
        self.assertEqual(
            [(0, 1, 1), (1, 0, -1), (1, 2, -1), (2, 1, 1)],
            graph.weighted_edge_list(),
        )

    def test_nan_null(self):
        input_matrix = np.array(
            [[np.nan, 1, np.nan], [1, np.nan, 1], [np.nan, 1, np.nan]],
            dtype=np.complex128,
        )
        graph = rustworkx.PyDiGraph.from_complex_adjacency_matrix(input_matrix, null_value=np.nan)
        edge_list = graph.weighted_edge_list()
        self.assertEqual(
            edge_list,
            [(0, 1, 1 + 0j), (1, 0, 1 + 0j), (1, 2, 1 + 0j), (2, 1, 1 + 0j)],
        )

    def test_parallel_edge(self):
        graph = rustworkx.PyDiGraph()
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

        min_matrix = rustworkx.digraph_adjacency_matrix(
            graph, weight_fn=lambda x: float(x), parallel_edge="min"
        )
        np.testing.assert_array_equal(
            [[0.0, 1.0, 2.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0]], min_matrix
        )

        max_matrix = rustworkx.digraph_adjacency_matrix(
            graph, weight_fn=lambda x: float(x), parallel_edge="max"
        )
        np.testing.assert_array_equal(
            [[0.0, 4.0, 2.0], [0.0, 0.0, 7.0], [1.0, 0.0, 0.0]], max_matrix
        )

        avg_matrix = rustworkx.digraph_adjacency_matrix(
            graph, weight_fn=lambda x: float(x), parallel_edge="avg"
        )
        np.testing.assert_array_equal(
            [[0.0, 8 / 3.0, 2.0], [0.0, 0.0, 4.5], [1.0, 0.0, 0.0]], avg_matrix
        )

        sum_matrix = rustworkx.digraph_adjacency_matrix(
            graph, weight_fn=lambda x: float(x), parallel_edge="sum"
        )
        np.testing.assert_array_equal(
            [[0.0, 8.0, 2.0], [0.0, 0.0, 9.0], [1.0, 0.0, 0.0]], sum_matrix
        )

        with self.assertRaises(ValueError):
            rustworkx.digraph_adjacency_matrix(
                graph, weight_fn=lambda x: float(x), parallel_edge="error"
            )


@unittest.skipIf(sp is None, "SciPy is not installed, skipping biadjacency matrix tests")
class TestDiGraphBiadjacencyMatrix(unittest.TestCase):
    def test_from_biadjacency_matrix(self):
        matrix = sp.csr_array(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
            dtype=np.float64,
        )
        graph = rustworkx.PyDiGraph.from_biadjacency_matrix(matrix)
        self.assertEqual(5, graph.num_nodes())
        self.assertEqual(
            [(0, 2, 1.0), (0, 4, 2.0), (1, 3, 3.0)],
            graph.weighted_edge_list(),
        )

    def test_from_biadjacency_matrix_integer_dtype(self):
        matrix = sp.csr_array([[1, 0], [0, 2]], dtype=np.int64)
        graph = rustworkx.PyDiGraph.from_biadjacency_matrix(matrix)
        self.assertEqual(
            [(0, 2, 1.0), (1, 3, 2.0)],
            graph.weighted_edge_list(),
        )

    def test_from_biadjacency_matrix_stored_zero(self):
        matrix = sp.coo_array(([0.0], ([0], [1])), shape=(1, 2))
        graph = rustworkx.PyDiGraph.from_biadjacency_matrix(matrix)
        self.assertEqual(
            [(0, 2, 0.0)],
            graph.weighted_edge_list(),
        )

    def test_from_biadjacency_matrix_rejects_dense_input(self):
        input_array = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        with self.assertRaises(TypeError):
            rustworkx.PyDiGraph.from_biadjacency_matrix(input_array)

    def test_from_biadjacency_matrix_empty_dimension(self):
        graph = rustworkx.PyDiGraph.from_biadjacency_matrix(sp.csr_array((3, 0), dtype=np.float64))
        self.assertEqual(3, graph.num_nodes())
        self.assertEqual([], graph.weighted_edge_list())

    def test_biadjacency_matrix(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(5))
        graph.add_edges_from([(0, 2, 1.0), (0, 4, 2.0), (1, 3, 3.0)])

        matrix = rustworkx.digraph_biadjacency_matrix(
            graph, [0, 1], [2, 3, 4], weight_fn=lambda x: x
        )
        expected = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        self.assertTrue(sp.issparse(matrix))
        self.assertEqual("csr", matrix.format)
        np.testing.assert_array_equal(expected, matrix.toarray())

    def test_universal_biadjacency_matrix(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(2))
        graph.add_edge(0, 1, 5.0)

        matrix = rustworkx.biadjacency_matrix(graph, [0], [1], weight_fn=float)
        np.testing.assert_array_equal([[5.0]], matrix.toarray())

    def test_biadjacency_matrix_ignores_incoming_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(3))
        graph.add_edge(2, 0, 7.0)

        matrix = rustworkx.digraph_biadjacency_matrix(graph, [0], [2], weight_fn=float)
        np.testing.assert_array_equal([[0.0]], matrix.toarray())

    def test_biadjacency_matrix_parallel_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(2))
        graph.add_edges_from([(0, 1, 1.0), (0, 1, 3.0)])

        max_matrix = rustworkx.digraph_biadjacency_matrix(
            graph, [0], [1], weight_fn=float, parallel_edge="max"
        )
        np.testing.assert_array_equal([[3.0]], max_matrix.toarray())

        min_matrix = rustworkx.digraph_biadjacency_matrix(
            graph, [0], [1], weight_fn=float, parallel_edge="min"
        )
        np.testing.assert_array_equal([[1.0]], min_matrix.toarray())

    def test_biadjacency_matrix_default_weight(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(2))
        graph.add_edge(0, 1, "edge")

        matrix = rustworkx.digraph_biadjacency_matrix(graph, [0], [1], default_weight=5.0)
        np.testing.assert_array_equal([[5.0]], matrix.toarray())

    def test_biadjacency_matrix_sparse_format(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(3))
        graph.add_edge(0, 1, 2.0)

        matrix = rustworkx.digraph_biadjacency_matrix(
            graph, [0], [1, 2], weight_fn=float, format="coo"
        )
        self.assertEqual("coo", matrix.format)
        np.testing.assert_array_equal([[2.0, 0.0]], matrix.toarray())

    def test_biadjacency_matrix_invalid_parallel_edge(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(2))
        graph.add_edge(0, 1, 1.0)

        with self.assertRaises(ValueError):
            rustworkx.digraph_biadjacency_matrix(
                graph, [0], [1], weight_fn=float, parallel_edge="error"
            )

    def test_biadjacency_matrix_missing_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        with self.assertRaises(ValueError):
            rustworkx.digraph_biadjacency_matrix(graph, [0], [1])

    def test_biadjacency_matrix_duplicate_node_order(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(3))

        with self.assertRaisesRegex(ValueError, "row_order contains duplicate node index 0"):
            rustworkx.digraph_biadjacency_matrix(graph, [0, 0], [1])
        with self.assertRaisesRegex(ValueError, "column_order contains duplicate node index 1"):
            rustworkx.digraph_biadjacency_matrix(graph, [0], [1, 1])

    def test_biadjacency_matrix_overlapping_node_order(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(2))

        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            rustworkx.digraph_biadjacency_matrix(graph, [0], [0, 1])

    def test_biadjacency_matrix_non_contiguous_node_indices(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(4))
        graph.add_edge(0, 3, 7.0)
        graph.remove_node(1)

        matrix = rustworkx.digraph_biadjacency_matrix(graph, [0], [3], weight_fn=float)
        np.testing.assert_array_equal([[7.0]], matrix.toarray())

        with self.assertRaises(ValueError):
            rustworkx.digraph_biadjacency_matrix(graph, [0], [1])

    def test_biadjacency_matrix_empty_order(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(2))

        matrix = rustworkx.digraph_biadjacency_matrix(graph, [0], [])
        self.assertTrue(sp.issparse(matrix))
        self.assertEqual((1, 0), matrix.shape)
