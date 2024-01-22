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


class TestCartesianProduct(unittest.TestCase):
    def test_null_cartesian_null(self):
        graph_1 = rustworkx.PyGraph()
        graph_2 = rustworkx.PyGraph()

        graph_product, _ = rustworkx.graph_cartesian_product(graph_1, graph_2)
        self.assertEqual(graph_product.num_nodes(), 0)
        self.assertEqual(graph_product.num_edges(), 0)

    def test_path_2_cartesian_path_2(self):
        graph_1 = rustworkx.generators.path_graph(2)
        graph_2 = rustworkx.generators.path_graph(2)

        graph_product, _ = rustworkx.graph_cartesian_product(graph_1, graph_2)
        self.assertEqual(graph_product.num_nodes(), 4)
        self.assertEqual(graph_product.num_edges(), 4)

    def test_path_2_cartesian_path_3(self):
        graph_1 = rustworkx.generators.path_graph(2)
        graph_2 = rustworkx.generators.path_graph(3)

        graph_product, _ = rustworkx.graph_cartesian_product(graph_1, graph_2)
        self.assertEqual(graph_product.num_nodes(), 6)
        self.assertEqual(graph_product.num_edges(), 7)

    def test_node_weights_cartesian(self):
        graph_1 = rustworkx.PyGraph()
        graph_1.add_node("a_1")
        graph_2 = rustworkx.PyGraph()
        graph_2.add_node(0)

        graph_product, _ = rustworkx.graph_cartesian_product(graph_1, graph_2)
        self.assertEqual([("a_1", 0)], graph_product.nodes())

    def test_edge_weights_cartesian(self):
        graph_1 = rustworkx.PyGraph()
        graph_1.add_nodes_from([0, 1])
        graph_1.add_edge(0, 1, "w_1")
        graph_2 = rustworkx.PyGraph()
        graph_2.add_nodes_from([0, 1])
        graph_1.add_edge(0, 1, "w_2")

        graph_product, _ = rustworkx.graph_cartesian_product(graph_1, graph_2)
        self.assertEqual(["w_1", "w_1", "w_2", "w_2"], graph_product.edges())
