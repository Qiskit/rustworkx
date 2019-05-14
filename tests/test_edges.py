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


class TestEdges(unittest.TestCase):

    def test_get_edge_data(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        res = dag.get_edge_data(node_a, node_b)
        self.assertEqual("Edgy", res)

    def test_no_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_node('b')
        self.assertRaises(Exception, dag.get_edge_data,
                          node_a, node_b)

    def test_edges(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        dag.add_child(node_b, 'c', "Super edgy")
        self.assertEqual(["Edgy", "Super edgy"], dag.edges())

    def test_edges_empty(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        self.assertEqual([], dag.edges())

    def test_add_duplicates(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'a', 'a')
        dag.add_edge(node_a, node_b, 'b')
        self.assertEqual(['a', 'b'], dag.edges())

    def test_remove_no_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_node('b')
        self.assertRaises(Exception, dag.remove_edge,
                          node_a, node_b)

    def test_remove_edge_single(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', 'edgy')
        dag.remove_edge(node_a, node_b)
        self.assertEqual([], dag.edges())

    def test_remove_multiple(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', 'edgy')
        dag.add_edge(node_a, node_b, 'super_edgy')
        dag.remove_edge_from_index(0)
        self.assertEqual(['super_edgy'], dag.edges())

    def test_remove_edge_from_index(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', 'edgy')
        dag.remove_edge_from_index(0)
        self.assertEqual([], dag.edges())

    def test_remove_edge_no_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.remove_edge_from_index(0)
        self.assertEqual([], dag.edges())
