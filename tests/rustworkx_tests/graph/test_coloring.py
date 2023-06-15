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


class TestGraphColoring(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.graph_greedy_color(graph)
        self.assertEqual({}, res)

    def test_simple_graph(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node(1)
        node_b = graph.add_node(2)
        graph.add_edge(node_a, node_b, 1)
        node_c = graph.add_node(3)
        graph.add_edge(node_a, node_c, 1)
        res = rustworkx.graph_greedy_color(graph)
        self.assertEqual({0: 0, 1: 1, 2: 1}, res)

    def test_simple_graph_large_degree(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node(1)
        node_b = graph.add_node(2)
        graph.add_edge(node_a, node_b, 1)
        node_c = graph.add_node(3)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        res = rustworkx.graph_greedy_color(graph)
        self.assertEqual({0: 0, 1: 1, 2: 1}, res)


class TestMisraGriesColoring(unittest.TestCase):

    def test_simple_graph(self):
        graph = rustworkx.PyGraph()
        node0 = graph.add_node(0)
        node1 = graph.add_node(1)
        node2 = graph.add_node(2)
        node3 = graph.add_node(3)

        graph.add_edge(node0, node1, 1)
        graph.add_edge(node0, node2, 1)
        graph.add_edge(node1, node2, 1)
        graph.add_edge(node2, node3, 1)

        print("======================")
        res = rustworkx.graph_misra_gries_edge_color(graph)
        print(f"{res = }")


    def test_grid(self):
        graph = rustworkx.generators.grid_graph(10, 10);

        print("======================")
        res = rustworkx.graph_misra_gries_edge_color(graph)
        print(f"{res = }")


    def test_heavy_hex(self):
        graph = rustworkx.generators.heavy_hex_graph(7);

        print("======================")
        res = rustworkx.graph_misra_gries_edge_color(graph)
        print(f"{res = }")

    def test_barbell(self):

        # graph = rustworkx.generators.barbell_graph(4, 3)
        graph = rustworkx.generators.complete_graph(10)

        print("======================")
        res = rustworkx.graph_misra_gries_edge_color(graph)
        print(f"{res = }")