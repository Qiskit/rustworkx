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


class TestLineGraph(unittest.TestCase):
    def test_graph(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        edge_ab = graph.add_edge(node_a, node_b, 1)
        edge_ac = graph.add_edge(node_a, node_c, 1)
        edge_bc = graph.add_edge(node_b, node_c, 1)
        edge_ad = graph.add_edge(node_a, node_d, 1)

        out_graph, out_edge_map = rustworkx.graph_line_graph(graph)
        expected_nodes = [0, 1, 2, 3]
        expected_edge_map = {edge_ab: 0, edge_ac: 1, edge_bc: 2, edge_ad: 3}
        expected_edges = [(3, 1), (3, 0), (1, 0), (2, 0), (2, 1)]
        self.assertEqual(out_graph.node_indices(), expected_nodes)
        self.assertEqual(out_graph.edge_list(), expected_edges)
        self.assertEqual(out_edge_map, expected_edge_map)

    def test_graph_with_holes(self):
        """Graph with missing node and edge indices."""
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        node_e = graph.add_node("e")
        edge_ab = graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_b, node_c, 1)
        graph.add_edge(node_c, node_d, 1)
        edge_de = graph.add_edge(node_d, node_e, 1)
        graph.remove_node(node_c)
        out_graph, out_edge_map = rustworkx.graph_line_graph(graph)

        expected_nodes = [0, 1]
        expected_edge_map = {edge_ab: 0, edge_de: 1}
        expected_edges = []
        self.assertEqual(out_graph.node_indices(), expected_nodes)
        self.assertEqual(out_graph.edge_list(), expected_edges)
        self.assertEqual(out_edge_map, expected_edge_map)
