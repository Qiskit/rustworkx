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
import copy

import rustworkx


class TestCopy(unittest.TestCase):
    def test_copy_returns_graph(self):
        graph_a = rustworkx.PyDiGraph()
        node_a = graph_a.add_node("a_1")
        node_b = graph_a.add_node("a_2")
        graph_a.add_edge(node_a, node_b, "edge_1")
        node_c = graph_a.add_node("a_3")
        graph_a.add_edge(node_b, node_c, "edge_2")
        graph_b = graph_a.copy()
        self.assertIsInstance(graph_b, rustworkx.PyDiGraph)

    def test_copy_with_holes_returns_graph(self):
        graph_a = rustworkx.PyDiGraph()
        node_a = graph_a.add_node("a_1")
        node_b = graph_a.add_node("a_2")
        graph_a.add_edge(node_a, node_b, "edge_1")
        node_c = graph_a.add_node("a_3")
        graph_a.add_edge(node_b, node_c, "edge_2")
        graph_a.remove_node(node_b)
        graph_b = graph_a.copy()
        self.assertIsInstance(graph_b, rustworkx.PyDiGraph)
        self.assertEqual([node_a, node_c], graph_b.node_indexes())

    def test_copy_empty(self):
        graph = rustworkx.PyDiGraph()
        empty_copy = graph.copy()
        self.assertEqual(len(empty_copy), 0)

    def test_copy_shared_ref(self):
        graph_a = rustworkx.PyDiGraph()
        node_a = graph_a.add_node({"a": 1})
        node_b = graph_a.add_node({"b": 2})
        graph_a.add_edge(node_a, node_b, {"edge": 1})
        graph_b = graph_a.copy()
        graph_a[0]["a"] = 42
        graph_b.get_edge_data(0, 1)["edge"] = 162
        self.assertEqual(graph_b[0]["a"], 42)
        self.assertEqual(graph_a.get_edge_data(0, 1), {"edge": 162})

    def test_python_copy_check_cycle(self):
        graph_a = rustworkx.PyDiGraph(check_cycle=True)
        graph_b = copy.copy(graph_a)
        graph_c = rustworkx.PyDiGraph(check_cycle=False)
        graph_d = copy.copy(graph_c)
        self.assertTrue(graph_b.check_cycle)
        self.assertFalse(graph_d.check_cycle)

    def test_python_copy_same_objects(self):
        graph_a = rustworkx.PyDiGraph(attrs=[1])
        node_a = graph_a.add_node([2])
        node_b = graph_a.add_child(node_a, [3], [4])
        graph_b = copy.copy(graph_a)
        self.assertEqual(graph_a.attrs, graph_b.attrs)
        self.assertIs(graph_a.attrs, graph_b.attrs)
        self.assertEqual(graph_a[node_a], graph_b[node_a])
        self.assertIs(graph_a[node_a], graph_b[node_a])
        self.assertEqual(
            graph_a.get_edge_data(node_a, node_b), graph_b.get_edge_data(node_a, node_b)
        )
        self.assertIs(graph_a.get_edge_data(node_a, node_b), graph_b.get_edge_data(node_a, node_b))
