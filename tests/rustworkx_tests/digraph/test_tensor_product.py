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


class TestTensorProduct(unittest.TestCase):
    def test_directed_null_tensor_null(self):
        graph_1 = rustworkx.PyDiGraph()
        graph_2 = rustworkx.PyDiGraph()

        graph_product, _ = rustworkx.digraph_tensor_product(graph_1, graph_2)
        self.assertEqual(graph_product.num_nodes(), 0)
        self.assertEqual(graph_product.num_edges(), 0)

    def test_directed_path_2_tensor_path_2(self):
        graph_1 = rustworkx.generators.directed_path_graph(2)
        graph_2 = rustworkx.generators.directed_path_graph(2)

        graph_product, node_map = rustworkx.digraph_tensor_product(graph_1, graph_2)
        expected_node_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        self.assertEqual(node_map, expected_node_map)

        expected_edges = [(0, 3)]
        self.assertEqual(graph_product.num_nodes(), 4)
        self.assertEqual(graph_product.num_edges(), 1)
        self.assertEqual(graph_product.edge_list(), expected_edges)

    def test_directed_path_2_tensor_path_3(self):
        graph_1 = rustworkx.generators.directed_path_graph(2)
        graph_2 = rustworkx.generators.directed_path_graph(3)

        graph_product, node_map = rustworkx.digraph_tensor_product(graph_1, graph_2)
        expected_node_map = {(0, 1): 1, (1, 0): 3, (0, 0): 0, (1, 2): 5, (0, 2): 2, (1, 1): 4}
        self.assertEqual(dict(node_map), expected_node_map)

        expected_edges = [(0, 4), (1, 5)]
        self.assertEqual(graph_product.num_nodes(), 6)
        self.assertEqual(graph_product.num_edges(), 2)
        self.assertEqual(graph_product.edge_list(), expected_edges)

    def test_directed_node_weights_tensor(self):
        graph_1 = rustworkx.PyDiGraph()
        graph_1.add_node("a_1")
        graph_2 = rustworkx.PyDiGraph()
        graph_2.add_node(0)

        graph_product, _ = rustworkx.digraph_tensor_product(graph_1, graph_2)
        self.assertEqual([("a_1", 0)], graph_product.nodes())

    def test_directed_edge_weights_tensor(self):
        graph_1 = rustworkx.PyDiGraph()
        graph_1.add_nodes_from([0, 1])
        graph_1.add_edge(0, 1, "w_1")
        graph_2 = rustworkx.PyDiGraph()
        graph_2.add_nodes_from([0, 1])
        graph_2.add_edge(0, 1, "w_2")

        graph_product, _ = rustworkx.digraph_tensor_product(graph_1, graph_2)
        self.assertEqual([("w_1", "w_2")], graph_product.edges())

    def test_multi_graph_1(self):
        graph_1 = rustworkx.generators.directed_path_graph(2)
        graph_1.add_edge(0, 1, None)
        graph_2 = rustworkx.generators.directed_path_graph(2)

        graph_product, _ = rustworkx.digraph_tensor_product(graph_1, graph_2)
        expected_edges = [(0, 3), (0, 3)]
        self.assertEqual(graph_product.num_edges(), 2)
        self.assertEqual(graph_product.edge_list(), expected_edges)

    def test_multi_graph_2(self):
        graph_1 = rustworkx.generators.directed_path_graph(2)
        graph_1.add_edge(0, 0, None)
        graph_2 = rustworkx.generators.directed_path_graph(2)

        graph_product, _ = rustworkx.digraph_tensor_product(graph_1, graph_2)
        expected_edges = [(0, 3), (0, 1)]
        self.assertEqual(graph_product.num_edges(), 2)
        self.assertEqual(graph_product.edge_list(), expected_edges)

    def test_multi_graph_3(self):
        graph_1 = rustworkx.generators.directed_path_graph(2)
        graph_2 = rustworkx.generators.directed_path_graph(2)
        graph_2.add_edge(0, 1, None)

        graph_product, _ = rustworkx.digraph_tensor_product(graph_1, graph_2)
        expected_edges = [(0, 3), (0, 3)]
        self.assertEqual(graph_product.num_edges(), 2)
        self.assertEqual(graph_product.edge_list(), expected_edges)

    def test_multi_graph_4(self):
        graph_1 = rustworkx.generators.directed_path_graph(2)
        graph_2 = rustworkx.generators.directed_path_graph(2)
        graph_2.add_edge(0, 0, None)

        graph_product, _ = rustworkx.digraph_tensor_product(graph_1, graph_2)
        expected_edges = [(0, 3), (0, 2)]
        self.assertEqual(graph_product.num_edges(), 2)
        self.assertEqual(graph_product.edge_list(), expected_edges)
