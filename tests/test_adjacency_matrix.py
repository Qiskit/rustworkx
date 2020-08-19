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
import numpy as np


class TestDAGAdjacencyMatrix(unittest.TestCase):
    def test_single_neighbor(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.add_child(node_a, 'b', {'a': 1})
        dag.add_child(node_a, 'c', {'a': 2})
        res = retworkx.digraph_adjacency_matrix(dag, lambda x: 1)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(
            np.array(
                [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float64),
            res))

    def test_float_cast_weight_func(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.add_child(node_a, 'b', 7.0)
        res = retworkx.digraph_adjacency_matrix(dag, lambda x: float(x))
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(
            np.array([[0.0, 7.0], [0.0, 0.0]]), res))

    def test_multigraph_sum_cast_weight_func(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', 7.0)
        dag.add_edge(node_a, node_b, 0.5)
        res = retworkx.digraph_adjacency_matrix(dag, lambda x: float(x))
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(
            np.array([[0.0, 7.5], [0.0, 0.0]]), res))

    def test_graph_to_digraph_adjacency_matrix(self):
        graph = retworkx.PyGraph()
        self.assertRaises(TypeError, retworkx.digraph_adjacency_matrix, graph)

    def test_no_edge_digraph_adjacency_matrix(self):
        dag = retworkx.PyDAG()
        for i in range(50):
            dag.add_node(i)
        res = retworkx.digraph_adjacency_matrix(dag, lambda x: 1)
        self.assertTrue(np.array_equal(np.zeros([50, 50]), res))

    def test_digraph_with_index_holes(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', 1)
        dag.add_child(node_a, 'c', 1)
        dag.remove_node(node_b)
        res = retworkx.digraph_adjacency_matrix(dag, lambda x: 1)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(
            np.array([[0, 1], [0, 0]]), res))


class TestGraphAdjacencyMatrix(unittest.TestCase):
    def test_single_neighbor(self):
        graph = retworkx.PyGraph()
        node_a = graph.add_node('a')
        node_b = graph.add_node('b')
        graph.add_edge(node_a, node_b, 'edge_a')
        node_c = graph.add_node('c')
        graph.add_edge(node_b, node_c, 'edge_b')
        res = retworkx.graph_adjacency_matrix(graph, lambda x: 1)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(
            np.array(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=np.float64),
            res))

    def test_float_cast_weight_func(self):
        graph = retworkx.PyGraph()
        node_a = graph.add_node('a')
        node_b = graph.add_node('b')
        graph.add_edge(node_a, node_b, 7.0)
        res = retworkx.graph_adjacency_matrix(graph, lambda x: float(x))
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(
            np.array([[0.0, 7.0], [7.0, 0.0]]), res))

    def test_multigraph_sum_cast_weight_func(self):
        graph = retworkx.PyGraph()
        node_a = graph.add_node('a')
        node_b = graph.add_node('b')
        graph.add_edge(node_a, node_b, 7.0)
        graph.add_edge(node_a, node_b, 0.5)
        res = retworkx.graph_adjacency_matrix(graph, lambda x: float(x))
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(
            np.array([[0.0, 7.5], [7.5, 0.0]]), res))

    def test_dag_to_graph_adjacency_matrix(self):
        dag = retworkx.PyDAG()
        self.assertRaises(TypeError, retworkx.graph_adjacency_matrix, dag)

    def test_no_edge_graph_adjacency_matrix(self):
        graph = retworkx.PyGraph()
        for i in range(50):
            graph.add_node(i)
        res = retworkx.graph_adjacency_matrix(graph, lambda x: 1)
        self.assertTrue(np.array_equal(np.zeros([50, 50]), res))

    def test_graph_with_index_holes(self):
        graph = retworkx.PyGraph()
        node_a = graph.add_node('a')
        node_b = graph.add_node('b')
        graph.add_edge(node_a, node_b, 1)
        node_c = graph.add_node('c')
        graph.add_edge(node_a, node_c, 1)
        graph.remove_node(node_b)
        res = retworkx.graph_adjacency_matrix(graph, lambda x: 1)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.array_equal(
            np.array([[0, 1], [1, 0]]), res))
