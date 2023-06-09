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


class TestGraphEdgeColoring(unittest.TestCase):
    def test_simple_graph(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node(1)
        node_b = graph.add_node(2)
        node_c = graph.add_node(3)
        node_d = graph.add_node(4)
        node_e = graph.add_node(5)

        edge_ab = graph.add_edge(node_a, node_b, 1)
        edge_ac = graph.add_edge(node_a, node_c, 1)
        edge_ad = graph.add_edge(node_a, node_d, 1)
        edge_de = graph.add_edge(node_d, node_e, 1)

        print(edge_ab)
        print(edge_ac)
        print(edge_ad)
        print(edge_de)

        print("============")
        res = rustworkx.graph_greedy_edge_color(graph)
        print("=====RESULT=======")
        print(res)
        print(graph.edge_indices()[2])


if __name__ == "__main__":
    unittest.main()
