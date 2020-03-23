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

    def test_get_all_edge_data(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        dag.add_edge(node_a, node_b, 'b')
        res = dag.get_all_edge_data(node_a, node_b)
        self.assertIn('b', res)
        self.assertIn('Edgy', res)

    def test_no_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_node('b')
        self.assertRaises(Exception, dag.get_edge_data,
                          node_a, node_b)

    def test_has_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        self.assertTrue(dag.has_edge(node_a, node_b))

    def test_has_edge_no_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_node('b')
        self.assertFalse(dag.has_edge(node_a, node_b))

    def test_edges(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        dag.add_child(node_b, 'c', "Super edgy")
        self.assertEqual(["Edgy", "Super edgy"], dag.edges())

    def test_edges_empty(self):
        dag = retworkx.PyDAG()
        dag.add_node('a')
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
        dag.add_child(node_a, 'b', 'edgy')
        dag.remove_edge_from_index(0)
        self.assertEqual([], dag.edges())

    def test_remove_edge_no_edge(self):
        dag = retworkx.PyDAG()
        dag.add_node('a')
        dag.remove_edge_from_index(0)
        self.assertEqual([], dag.edges())

    def test_add_cycle(self):
        dag = retworkx.PyDAG()
        dag.check_cycle = True
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        self.assertRaises(Exception, dag.add_edge, node_b,
                          node_a, {})

    def test_add_edge_with_cycle_check_enabled(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node('a')
        node_c = dag.add_node('c')
        node_b = dag.add_child(node_a, 'b', {})
        dag.add_edge(node_c, node_b, {})
        self.assertTrue(dag.has_edge(node_c, node_b))

    def test_enable_cycle_checking_after_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        dag.add_edge(node_b, node_a, {})
        with self.assertRaises(Exception):
            dag.check_cycle = True

    def test_cycle_checking_at_init(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        with self.assertRaises(Exception):
            dag.add_edge(node_b, node_a, {})
