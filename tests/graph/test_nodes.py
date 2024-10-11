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


class TestNodes(unittest.TestCase):
    def test_nodes(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        res = graph.nodes()
        self.assertEqual(["a", "b"], res)
        self.assertEqual([0, 1], graph.node_indexes())

    def test_node_indices(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        self.assertEqual([0, 1], graph.node_indices())

    def test_no_nodes(self):
        graph = rustworkx.PyGraph()
        self.assertEqual([], graph.nodes())
        self.assertEqual([], graph.node_indexes())
        self.assertEqual([], graph.node_indices())

    def test_remove_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_node("c")
        graph.remove_node(node_b)
        res = graph.nodes()
        self.assertEqual(["a", "c"], res)
        self.assertEqual([0, 2], graph.node_indexes())

    def test_remove_node_invalid_index(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.remove_node(76)
        res = graph.nodes()
        self.assertEqual(["a", "b", "c"], res)
        self.assertEqual([0, 1, 2], graph.node_indexes())

    def test_remove_nodes_from(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Edgy_mk2")
        graph.remove_nodes_from([node_b, node_c])
        res = graph.nodes()
        self.assertEqual(["a"], res)
        self.assertEqual([0], graph.node_indexes())

    def test_remove_nodes_from_gen(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Edgy_mk2")
        graph.remove_nodes_from(n for n in [node_b, node_c])
        res = graph.nodes()
        self.assertEqual(["a"], res)
        self.assertEqual([0], graph.node_indexes())

    def test_remove_nodes_from_with_invalid_index(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Edgy_mk2")
        graph.remove_nodes_from([node_b, node_c, 76])
        res = graph.nodes()
        self.assertEqual(["a"], res)
        self.assertEqual([0], graph.node_indexes())

    def test_get_node_data(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertEqual("b", graph.get_node_data(node_b))

    def test_get_node_data_bad_index(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        self.assertRaises(IndexError, graph.get_node_data, 42)

    def test_pygraph_length(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "An_edge")
        self.assertEqual(2, len(graph))

    def test_pygraph_num_nodes(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "An_edge")
        self.assertEqual(2, graph.num_nodes())

    def test_pygraph_length_empty(self):
        graph = rustworkx.PyGraph()
        self.assertEqual(0, len(graph))

    def test_pygraph_num_nodes_empty(self):
        graph = rustworkx.PyGraph()
        self.assertEqual(0, graph.num_nodes())

    def test_add_nodes_from(self):
        graph = rustworkx.PyGraph()
        nodes = list(range(100))
        res = graph.add_nodes_from(nodes)
        self.assertEqual(len(res), 100)
        self.assertEqual(res, nodes)

    def test_add_nodes_from_gen(self):
        graph = rustworkx.PyGraph()
        nodes = list(range(100))
        node_gen = (i**2 for i in nodes)
        res = graph.add_nodes_from(node_gen)
        self.assertEqual(len(res), 100)
        self.assertEqual(res, nodes)

    def test_add_node_from_empty(self):
        graph = rustworkx.PyGraph()
        res = graph.add_nodes_from([])
        self.assertEqual(len(res), 0)

    def test_get_node_data_getitem(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        self.assertEqual("b", graph[node_b])

    def test_get_node_data_getitem_bad_index(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        with self.assertRaises(IndexError):
            graph[42]

    def test_set_node_data_setitem(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        graph[node_b] = "Oh so cool"
        self.assertEqual("Oh so cool", graph[node_b])

    def test_set_node_data_setitem_bad_index(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        with self.assertRaises(IndexError):
            graph[42] = "Oh so cool"

    def test_remove_node_delitem(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Edgy_mk2")
        del graph[node_b]
        res = graph.nodes()
        self.assertEqual(["a", "c"], res)
        self.assertEqual([0, 2], graph.node_indexes())

    def test_remove_node_delitem_invalid_index(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        with self.assertRaises(IndexError):
            del graph[76]
        res = graph.nodes()
        self.assertEqual(["a", "b", "c"], res)
        self.assertEqual([0, 1, 2], graph.node_indexes())

    def test_has_node(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        self.assertTrue(graph.has_node(node_a))
        self.assertFalse(graph.has_node(node_a + 1))
