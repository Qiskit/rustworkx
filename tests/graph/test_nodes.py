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


class TestNodes(unittest.TestCase):

    def test_nodes(self):
        graph = retworkx.PyGraph()
        graph.add_node('a')
        graph.add_node('b')
        res = graph.nodes()
        self.assertEqual(['a', 'b'], res)
        self.assertEqual([0, 1], graph.node_indexes())

    def test_no_nodes(self):
        graph = retworkx.PyGraph()
        self.assertEqual([], graph.nodes())
        self.assertEqual([], graph.node_indexes())

    def test_remove_node(self):
        graph = retworkx.PyGraph()
        graph.add_node('a')
        node_b = graph.add_node('b')
        graph.add_node('c')
        graph.remove_node(node_b)
        res = graph.nodes()
        self.assertEqual(['a', 'c'], res)
        self.assertEqual([0, 2], graph.node_indexes())

    def test_remove_node_invalid_index(self):
        graph = retworkx.PyGraph()
        graph.add_node('a')
        graph.add_node('b')
        graph.add_node('c')
        graph.remove_node(76)
        res = graph.nodes()
        self.assertEqual(['a', 'b', 'c'], res)
        self.assertEqual([0, 1, 2], graph.node_indexes())

    def test_remove_nodes_from(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        node_c = dag.add_child(node_b, 'c', "Edgy_mk2")
        dag.remove_nodes_from([node_b, node_c])
        res = dag.nodes()
        self.assertEqual(['a'], res)
        self.assertEqual([0], dag.node_indexes())

    def test_remove_nodes_from_with_invalid_index(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        node_c = dag.add_child(node_b, 'c', "Edgy_mk2")
        dag.remove_nodes_from([node_b, node_c, 76])
        res = dag.nodes()
        self.assertEqual(['a'], res)
        self.assertEqual([0], dag.node_indexes())

    def test_get_node_data(self):
        graph = retworkx.PyGraph()
        graph.add_node('a')
        node_b = graph.add_node('b')
        self.assertEqual('b', graph.get_node_data(node_b))

    def test_get_node_data_bad_index(self):
        graph = retworkx.PyGraph()
        graph.add_node('a')
        graph.add_node('b')
        self.assertRaises(IndexError, graph.get_node_data, 42)

    def test_pygraph_length(self):
        graph = retworkx.PyGraph()
        node_a = graph.add_node('a')
        node_b = graph.add_node('b')
        graph.add_edge(node_a, node_b, 'An_edge')
        self.assertEqual(2, len(graph))

    def test_pygraph_length_empty(self):
        graph = retworkx.PyGraph()
        self.assertEqual(0, len(graph))

    def test_add_nodes_from(self):
        graph = retworkx.PyGraph()
        nodes = list(range(100))
        res = graph.add_nodes_from(nodes)
        self.assertEqual(len(res), 100)
        self.assertEqual(res, nodes)

    def test_add_node_from_empty(self):
        graph = retworkx.PyGraph()
        res = graph.add_nodes_from([])
        self.assertEqual(len(res), 0)
