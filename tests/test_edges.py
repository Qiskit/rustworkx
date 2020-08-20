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
        self.assertRaises(retworkx.NoEdgeBetweenNodes, dag.get_edge_data,
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
        self.assertRaises(retworkx.NoEdgeBetweenNodes, dag.remove_edge,
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
        self.assertRaises(retworkx.DAGWouldCycle, dag.add_edge, node_b,
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
        with self.assertRaises(retworkx.DAGHasCycle):
            dag.check_cycle = True

    def test_cycle_checking_at_init(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        with self.assertRaises(retworkx.DAGWouldCycle):
            dag.add_edge(node_b, node_a, {})

    def test_find_adjacent_node_by_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.add_child(node_a, 'b', {'weights': [1, 2]})
        dag.add_child(node_a, 'c', {'weights': [3, 4]})

        def compare_edges(edge):
            return 4 in edge['weights']

        res = dag.find_adjacent_node_by_edge(node_a, compare_edges)
        self.assertEqual('c', res)

    def test_find_adjacent_node_by_edge_no_match(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        dag.add_child(node_a, 'b', {'weights': [1, 2]})
        dag.add_child(node_a, 'c', {'weights': [3, 4]})

        def compare_edges(edge):
            return 5 in edge['weights']

        with self.assertRaises(retworkx.NoSuitableNeighbors):
            dag.find_adjacent_node_by_edge(node_a, compare_edges)

    def test_add_edge_from(self):
        dag = retworkx.PyDAG()
        nodes = list(range(4))
        dag.add_nodes_from(nodes)
        edge_list = [(0, 1, 'a'), (1, 2, 'b'), (0, 2, 'c'), (2, 3, 'd'),
                     (0, 3, 'e')]
        res = dag.add_edges_from(edge_list)
        self.assertEqual(len(res), 5)
        self.assertEqual(['a', 'b', 'c', 'd', 'e'], dag.edges())
        self.assertEqual(3, dag.out_degree(0))
        self.assertEqual(0, dag.in_degree(0))
        self.assertEqual(1, dag.out_degree(1))
        self.assertEqual(1, dag.out_degree(2))
        self.assertEqual(2, dag.in_degree(3))

    def test_add_edge_from_empty(self):
        dag = retworkx.PyDAG()
        res = dag.add_edges_from([])
        self.assertEqual([], res)

    def test_cycle_checking_at_init_nodes_from(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        node_c = dag.add_child(node_b, 'c', {})
        with self.assertRaises(retworkx.DAGWouldCycle):
            dag.add_edges_from([(node_a, node_c, {}), (node_c, node_b, {})])

    def test_add_edge_from_no_data(self):
        dag = retworkx.PyDAG()
        nodes = list(range(4))
        dag.add_nodes_from(nodes)
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3),
                     (0, 3)]
        res = dag.add_edges_from_no_data(edge_list)
        self.assertEqual(len(res), 5)
        self.assertEqual([None, None, None, None, None], dag.edges())
        self.assertEqual(3, dag.out_degree(0))
        self.assertEqual(0, dag.in_degree(0))
        self.assertEqual(1, dag.out_degree(1))
        self.assertEqual(1, dag.out_degree(2))
        self.assertEqual(2, dag.in_degree(3))

    def test_add_edge_from_empty_no_data(self):
        dag = retworkx.PyDAG()
        res = dag.add_edges_from_no_data([])
        self.assertEqual([], res)

    def test_cycle_checking_at_init_nodes_from_no_data(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        node_c = dag.add_child(node_b, 'c', {})
        with self.assertRaises(retworkx.DAGWouldCycle):
            dag.add_edges_from_no_data([(node_a, node_c), (node_c, node_b)])

    def test_edge_list(self):
        dag = retworkx.PyDiGraph()
        dag.add_nodes_from(list(range(4)))
        edge_list = [(0, 1, 'a'), (1, 2, 'b'), (0, 2, 'c'), (2, 3, 'd'),
                     (0, 3, 'e')]
        dag.add_edges_from(edge_list)
        self.assertEqual([(x[0], x[1]) for x in edge_list], dag.edge_list())

    def test_edge_list_empty(self):
        dag = retworkx.PyDiGraph()
        self.assertEqual([], dag.edge_list())

    def test_weighted_edge_list(self):
        dag = retworkx.PyDiGraph()
        dag.add_nodes_from(list(range(4)))
        edge_list = [(0, 1, 'a'), (1, 2, 'b'), (0, 2, 'c'), (2, 3, 'd'),
                     (0, 3, 'e')]
        dag.add_edges_from(edge_list)
        self.assertEqual(edge_list, dag.weighted_edge_list())

    def test_edge_list_empty(self):
        dag = retworkx.PyDiGraph()
        self.assertEqual([], dag.weighted_edge_list())
