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


class TestPlanarLayout(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
        node_a = self.graph.add_node(1)
        node_b = self.graph.add_node(2)
        self.graph.add_edge(node_a, node_b, 1)
        node_c = self.graph.add_node(3)
        self.graph.add_edge(node_a, node_c, 2)

    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.planar_layout(graph)
        self.assertEqual({}, res)

    def test_simple_graph(self):
        res = rustworkx.planar_layout(self.graph)
        self.assertEqual(len(res), 3)
        self.assertEqual(len(res[0]), 2)
        self.assertIsInstance(res[0][0], float)

    def test_simple_graph_center(self):
        res = rustworkx.planar_layout(self.graph, center=[0.5, 0.5])
        self.assertEqual(len(res), 3)
        self.assertEqual(len(res[0]), 2)
        self.assertIsInstance(res[0][0], float)

    def test_graph_with_removed_nodes(self):
        graph = rustworkx.PyGraph()
        nodes = graph.add_nodes_from([0, 1, 2])
        graph.remove_node(nodes[1])
        res = rustworkx.planar_layout(graph)
        self.assertEqual(len(res), 2)
        self.assertTrue(nodes[0] in res)
        self.assertTrue(nodes[2] in res)
        self.assertFalse(nodes[1] in res)

    def test_graph_with_more_removed_nodes(self):
        graph = rustworkx.PyGraph()
        nodes = graph.add_nodes_from([0, 1, 2, 3, 4, 5])
        graph.remove_node(nodes[3])
        res = rustworkx.planar_layout(graph)
        self.assertEqual(len(res), 5)
        self.assertTrue(nodes[0] in res)
        self.assertTrue(nodes[4] in res)
        self.assertFalse(nodes[3] in res)
