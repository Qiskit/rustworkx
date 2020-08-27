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
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.add_child(node_a, 'b', "Edgy")
        res = dag.nodes()
        self.assertEqual(['a', 'b'], res)
        self.assertEqual([0, 1], dag.node_indexes())

    def test_no_nodes(self):
        dag = retworkx.PyDAG()
        self.assertEqual([], dag.nodes())
        self.assertEqual([], dag.node_indexes())

    def test_remove_node(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        dag.add_child(node_b, 'c', "Edgy_mk2")
        dag.remove_node(node_b)
        res = dag.nodes()
        self.assertEqual(['a', 'c'], res)
        self.assertEqual([0, 2], dag.node_indexes())

    def test_remove_node_invalid_index(self):
        graph = retworkx.PyDAG()
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

    def test_topo_sort_empty(self):
        dag = retworkx.PyDAG()
        self.assertEqual([], retworkx.topological_sort(dag))

    def test_topo_sort(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        for i in range(5):
            dag.add_child(node_a, i, None)
        dag.add_parent(3, 'A parent', None)
        res = retworkx.topological_sort(dag)
        self.assertEqual([6, 0, 5, 4, 3, 2, 1], res)

    def test_topo_sort_with_cycle(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        dag.add_edge(node_b, node_a, {})
        self.assertRaises(retworkx.DAGHasCycle, retworkx.topological_sort, dag)

    def test_lexicographical_topo_sort(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        for i in range(5):
            dag.add_child(node_a, i, None)
        dag.add_parent(3, 'A parent', None)
        res = retworkx.lexicographical_topological_sort(dag, lambda x: str(x))
        # Node values for nodes [6, 0, 5, 4, 3, 2, 1]
        expected = ['A parent', 'a', 0, 1, 2, 3, 4]
        self.assertEqual(expected, res)

    def test_lexicographical_topo_sort_qiskit(self):
        dag = retworkx.PyDAG()
        # inputs
        qr_0 = dag.add_node('qr[0]')
        qr_1 = dag.add_node('qr[1]')
        qr_2 = dag.add_node('qr[2]')
        cr_0 = dag.add_node('cr[0]')
        cr_1 = dag.add_node('cr[1]')

        # wires
        cx_1 = dag.add_node('cx_1')
        dag.add_edge(qr_0, cx_1, 'qr[0]')
        dag.add_edge(qr_1, cx_1, 'qr[1]')
        h_1 = dag.add_node('h_1')
        dag.add_edge(cx_1, h_1, 'qr[0]')
        cx_2 = dag.add_node('cx_2')
        dag.add_edge(cx_1, cx_2, 'qr[1]')
        dag.add_edge(qr_2, cx_2, 'qr[2]')
        cx_3 = dag.add_node('cx_3')
        dag.add_edge(h_1, cx_3, 'qr[0]')
        dag.add_edge(cx_2, cx_3, 'qr[2]')
        h_2 = dag.add_node('h_2')
        dag.add_edge(cx_3, h_2, 'qr[2]')

        # outputs
        qr_0_out = dag.add_node('qr[0]_out')
        dag.add_edge(cx_3, qr_0_out, 'qr[0]')
        qr_1_out = dag.add_node('qr[1]_out')
        dag.add_edge(cx_2, qr_1_out, 'qr[1]')
        qr_2_out = dag.add_node('qr[2]_out')
        dag.add_edge(h_2, qr_2_out, 'qr[2]')
        cr_0_out = dag.add_node('cr[0]_out')
        dag.add_edge(cr_0, cr_0_out, 'qr[2]')
        cr_1_out = dag.add_node('cr[1]_out')
        dag.add_edge(cr_1, cr_1_out, 'cr[1]')

        res = list(retworkx.lexicographical_topological_sort(dag,
                                                             lambda x: str(x)))
        expected = ['cr[0]', 'cr[0]_out', 'cr[1]', 'cr[1]_out', 'qr[0]',
                    'qr[1]', 'cx_1', 'h_1', 'qr[2]', 'cx_2', 'cx_3', 'h_2',
                    'qr[0]_out', 'qr[1]_out', 'qr[2]_out']
        self.assertEqual(expected, res)

    def test_get_node_data(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        self.assertEqual('b', dag.get_node_data(node_b))

    def test_get_node_data_bad_index(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.add_child(node_a, 'b', "Edgy")
        self.assertRaises(IndexError, dag.get_node_data, 42)

    def test_pydag_length(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.add_child(node_a, 'b', "Edgy")
        self.assertEqual(2, len(dag))

    def test_pydag_length_empty(self):
        dag = retworkx.PyDAG()
        self.assertEqual(0, len(dag))

    def test_add_nodes_from(self):
        dag = retworkx.PyDAG()
        nodes = list(range(100))
        res = dag.add_nodes_from(nodes)
        self.assertEqual(len(res), 100)
        self.assertEqual(res, nodes)

    def test_add_node_from_empty(self):
        dag = retworkx.PyDAG()
        res = dag.add_nodes_from([])
        self.assertEqual(len(res), 0)

    def test_get_node_data_getitem(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        self.assertEqual('b', dag[node_b])

    def test_get_node_data_getitem_bad_index(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.add_child(node_a, 'b', "Edgy")
        with self.assertRaises(IndexError):
            dag[42]

    def test_set_node_data_setitem(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        dag[node_b] = 'Oh so cool'
        self.assertEqual('Oh so cool', dag[node_b])

    def test_set_node_data_setitem_bad_index(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.add_child(node_a, 'b', "Edgy")
        with self.assertRaises(IndexError):
            dag[42] = 'Oh so cool'

    def test_remove_node_delitem(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        dag.add_child(node_b, 'c', "Edgy_mk2")
        del dag[node_b]
        res = dag.nodes()
        self.assertEqual(['a', 'c'], res)
        self.assertEqual([0, 2], dag.node_indexes())

    def test_remove_node_delitem_invalid_index(self):
        graph = retworkx.PyDAG()
        graph.add_node('a')
        graph.add_node('b')
        graph.add_node('c')
        with self.assertRaises(IndexError):
            del graph[76]
        res = graph.nodes()
        self.assertEqual(['a', 'b', 'c'], res)
        self.assertEqual([0, 1, 2], graph.node_indexes())
