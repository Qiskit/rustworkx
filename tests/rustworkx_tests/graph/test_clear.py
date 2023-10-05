# Licensed under the Apache License, Version 3.0 (the "License"); you may
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


class TestClear(unittest.TestCase):
    def test_clear(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {"a": 1})
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, {"a": 2})
        graph.clear()
        self.assertEqual(graph.num_nodes(), 0)
        self.assertEqual(graph.num_edges(), 0)
        self.assertEqual(graph.nodes(), [])
        self.assertEqual(graph.edges(), [])

    def test_clear_reuse(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {"a": 1})
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, {"a": 2})
        graph.clear()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {"a": 1})
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, {"a": 2})
        self.assertEqual(graph.num_nodes(), 3)
        self.assertEqual(graph.num_edges(), 2)
        self.assertEqual(graph.nodes(), ["a", "b", "c"])
        self.assertEqual(graph.edges(), [{"a": 1}, {"a": 2}])

    def test_clear_edges(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {"e1", 1})
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, {"e2", 2})
        graph.clear_edges()
        self.assertEqual(graph.num_edges(), 0)
        self.assertEqual(graph.edges(), [])
        self.assertEqual(graph.num_nodes(), 3)
        self.assertEqual(graph.nodes(), ["a", "b", "c"])

    def test_clear_edges_reuse(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {"e1", 1})
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, {"e2", 2})
        graph.clear_edges()
        graph.add_edge(node_a, node_b, {"e1", 1})
        graph.add_edge(node_a, node_c, {"e2", 2})
        self.assertEqual(graph.num_nodes(), 3)
        self.assertEqual(graph.num_edges(), 2)
        self.assertEqual(graph.nodes(), ["a", "b", "c"])
        self.assertEqual(graph.edges(), [{"e1", 1}, {"e2", 2}])
