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

import copy
import unittest

import retworkx


class TestIsomorphic(unittest.TestCase):

    def test_isomorphic_identical(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('a_1')
        dag_b.add_child(node_b, 'a_2', 'a_1')
        dag_b.add_child(node_b, 'a_3', 'a_2')
        self.assertTrue(retworkx.is_isomorphic(dag_a, dag_b))

    def test_isomorphic_mismatch_node_data(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('b_1')
        dag_b.add_child(node_b, 'b_2', 'b_1')
        dag_b.add_child(node_b, 'b_3', 'b_2')
        self.assertTrue(retworkx.is_isomorphic(dag_a, dag_b))

    def test_isomorphic_compare_nodes_mismatch_node_data(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('b_1')
        dag_b.add_child(node_b, 'b_2', 'b_1')
        dag_b.add_child(node_b, 'b_3', 'b_2')
        self.assertFalse(
            retworkx.is_isomorphic(
                dag_a, dag_b, lambda x, y: x == y))

    def test_is_isomorphic_nodes_compare_raises(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('b_1')
        dag_b.add_child(node_b, 'b_2', 'b_1')
        dag_b.add_child(node_b, 'b_3', 'b_2')

        def compare_nodes(a, b):
            raise TypeError("Failure")

        self.assertRaises(
            TypeError,
            retworkx.is_isomorphic,
            (dag_a, dag_b, compare_nodes))

    def test_isomorphic_compare_nodes_identical(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('a_1')
        dag_b.add_child(node_b, 'a_2', 'a_1')
        dag_b.add_child(node_b, 'a_3', 'a_2')
        self.assertTrue(
            retworkx.is_isomorphic(
                dag_a, dag_b, lambda x, y: x == y))

    def test_isomorphic_compare_edges_identical(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('a_1')
        dag_b.add_child(node_b, 'a_2', 'a_1')
        dag_b.add_child(node_b, 'a_3', 'a_2')
        self.assertTrue(
            retworkx.is_isomorphic(
                dag_a, dag_b, edge_matcher=lambda x, y: x == y))

    def test_isomorphic_compare_nodes_with_removals(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        qr_0_in = dag_a.add_node('qr[0]')
        qr_1_in = dag_a.add_node('qr[1]')
        cr_0_in = dag_a.add_node('cr[0]')
        qr_0_out = dag_a.add_node('qr[0]')
        qr_1_out = dag_a.add_node('qr[1]')
        cr_0_out = dag_a.add_node('qr[0]')
        cu1 = dag_a.add_child(qr_0_in, 'cu1', 'qr[0]')
        dag_a.add_edge(qr_1_in, cu1, 'qr[1]')
        measure_0 = dag_a.add_child(cr_0_in, 'measure', 'cr[0]')
        dag_a.add_edge(cu1, measure_0, 'qr[0]')
        measure_1 = dag_a.add_child(cu1, 'measure', 'qr[1]')
        dag_a.add_edge(measure_0, measure_1, 'cr[0]')
        dag_a.add_edge(measure_1, qr_1_out, 'qr[1]')
        dag_a.add_edge(measure_1, cr_0_out, 'cr[0]')
        dag_a.add_edge(measure_0, qr_0_out, 'qr[0]')
        dag_a.remove_node(cu1)
        dag_a.add_edge(qr_0_in, measure_0, 'qr[0]')
        dag_a.add_edge(qr_1_in, measure_1, 'qr[1]')

        qr_0_in = dag_b.add_node('qr[0]')
        qr_1_in = dag_b.add_node('qr[1]')
        cr_0_in = dag_b.add_node('cr[0]')
        qr_0_out = dag_b.add_node('qr[0]')
        qr_1_out = dag_b.add_node('qr[1]')
        cr_0_out = dag_b.add_node('qr[0]')
        measure_0 = dag_b.add_child(cr_0_in, 'measure', 'cr[0]')
        dag_b.add_edge(qr_0_in, measure_0, 'qr[0]')
        measure_1 = dag_b.add_child(qr_1_in, 'measure', 'qr[1]')
        dag_b.add_edge(measure_1, qr_1_out, 'qr[1]')
        dag_b.add_edge(measure_1, cr_0_out, 'cr[0]')
        dag_b.add_edge(measure_0, measure_1, 'cr[0]')
        dag_b.add_edge(measure_0, qr_0_out, 'qr[0]')

        self.assertTrue(
            retworkx.is_isomorphic(
                dag_a, dag_b, lambda x, y: x == y))

    def test_isomorphic_compare_nodes_with_removals_deepcopy(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        qr_0_in = dag_a.add_node('qr[0]')
        qr_1_in = dag_a.add_node('qr[1]')
        cr_0_in = dag_a.add_node('cr[0]')
        qr_0_out = dag_a.add_node('qr[0]')
        qr_1_out = dag_a.add_node('qr[1]')
        cr_0_out = dag_a.add_node('qr[0]')
        cu1 = dag_a.add_child(qr_0_in, 'cu1', 'qr[0]')
        dag_a.add_edge(qr_1_in, cu1, 'qr[1]')
        measure_0 = dag_a.add_child(cr_0_in, 'measure', 'cr[0]')
        dag_a.add_edge(cu1, measure_0, 'qr[0]')
        measure_1 = dag_a.add_child(cu1, 'measure', 'qr[1]')
        dag_a.add_edge(measure_0, measure_1, 'cr[0]')
        dag_a.add_edge(measure_1, qr_1_out, 'qr[1]')
        dag_a.add_edge(measure_1, cr_0_out, 'cr[0]')
        dag_a.add_edge(measure_0, qr_0_out, 'qr[0]')
        dag_a.remove_node(cu1)
        dag_a.add_edge(qr_0_in, measure_0, 'qr[0]')
        dag_a.add_edge(qr_1_in, measure_1, 'qr[1]')

        qr_0_in = dag_b.add_node('qr[0]')
        qr_1_in = dag_b.add_node('qr[1]')
        cr_0_in = dag_b.add_node('cr[0]')
        qr_0_out = dag_b.add_node('qr[0]')
        qr_1_out = dag_b.add_node('qr[1]')
        cr_0_out = dag_b.add_node('qr[0]')
        measure_0 = dag_b.add_child(cr_0_in, 'measure', 'cr[0]')
        dag_b.add_edge(qr_0_in, measure_0, 'qr[0]')
        measure_1 = dag_b.add_child(qr_1_in, 'measure', 'qr[1]')
        dag_b.add_edge(measure_1, qr_1_out, 'qr[1]')
        dag_b.add_edge(measure_1, cr_0_out, 'cr[0]')
        dag_b.add_edge(measure_0, measure_1, 'cr[0]')
        dag_b.add_edge(measure_0, qr_0_out, 'qr[0]')

        self.assertTrue(
            retworkx.is_isomorphic(
                copy.deepcopy(dag_a), copy.deepcopy(dag_b),
                lambda x, y: x == y))

    def test_isomorphic_identical_undirected(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_a.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])

        nodes = g_b.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_b.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])
        self.assertTrue(retworkx.is_isomorphic(g_a, g_b))

    def test_isomorphic_mismatch_node_data_undirected(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_a.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])

        nodes = g_b.add_nodes_from(['b_1', 'b_2', 'b_3'])
        g_b.add_edges_from([
            (nodes[0], nodes[1], 'b_1'), (nodes[1], nodes[2], 'b_2')
        ])
        self.assertTrue(retworkx.is_isomorphic(g_a, g_b))

    def test_isomorphic_compare_nodes_mismatch_node_data_undirected(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_a.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])

        nodes = g_b.add_nodes_from(['b_1', 'b_2', 'b_3'])
        g_b.add_edges_from([
            (nodes[0], nodes[1], 'b_1'), (nodes[1], nodes[2], 'b_2')
        ])
        self.assertFalse(
            retworkx.is_isomorphic(
                g_a, g_b, lambda x, y: x == y))

    def test_is_isomorphic_nodes_compare_raises_undirected(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_a.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])

        nodes = g_b.add_nodes_from(['b_1', 'b_2', 'b_3'])
        g_b.add_edges_from([
            (nodes[0], nodes[1], 'b_1'), (nodes[1], nodes[2], 'b_2')
        ])

        def compare_nodes(a, b):
            raise TypeError("Failure")

        self.assertRaises(
            TypeError,
            retworkx.is_isomorphic,
            (g_a, g_b, compare_nodes))

    def test_isomorphic_compare_nodes_identical_undirected(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_a.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])

        nodes = g_b.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_b.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])
        self.assertTrue(
            retworkx.is_isomorphic(
                g_a, g_b, lambda x, y: x == y))

    def test_isomorphic_compare_edges_identical_undirected(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_a.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])

        nodes = g_b.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_b.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])
        self.assertTrue(
            retworkx.is_isomorphic(
                g_a, g_b, edge_matcher=lambda x, y: x == y))

    def test_isomorphic_removed_nodes_in_second_graph(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_a.add_edges_from([
            (nodes[0], nodes[1], 'a_1'), (nodes[1], nodes[2], 'a_2')
        ])

        nodes = g_b.add_nodes_from(['a_0', 'a_2', 'a_1', 'a_3'])
        g_b.add_edges_from([
            (nodes[0], nodes[1], 'e_01'), (nodes[0], nodes[3], 'e_03'),
            (nodes[2], nodes[1], 'a_1'), (nodes[1], nodes[3], 'a_2')
        ])
        g_b.remove_node(nodes[0])
        self.assertTrue(
            retworkx.is_isomorphic(
                g_a, g_b, lambda x, y: x == y))

    def test_isomorphic_node_count_not_equal(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(['a_1', 'a_2', 'a_3'])
        g_a.add_edges_from([
            (nodes[0], nodes[1], 'a_1')
        ])

        nodes = g_b.add_nodes_from(['a_0', 'a_1'])
        g_b.add_edges_from([
            (nodes[0], nodes[1], 'a_1')
        ])
        g_b.remove_node(nodes[0])
        self.assertFalse(
            retworkx.is_isomorphic(
                g_a, g_b))

    def test_isomorphic_identical_default_order(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('a_1')
        dag_b.add_child(node_b, 'a_2', 'a_1')
        dag_b.add_child(node_b, 'a_3', 'a_2')
        self.assertTrue(
            retworkx.is_isomorphic(
                dag_a, dag_b, default_order=True))
