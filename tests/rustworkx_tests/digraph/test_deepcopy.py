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

import rustworkx


class TestDeepcopy(unittest.TestCase):
    def test_isomorphic_compare_nodes_identical(self):
        dag_a = rustworkx.PyDAG()
        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")
        dag_b = copy.deepcopy(dag_a)
        self.assertTrue(rustworkx.is_isomorphic_node_match(dag_a, dag_b, lambda x, y: x == y))

    def test_deepcopy_with_holes(self):
        dag_a = rustworkx.PyDAG()
        node_a = dag_a.add_node("a_1")
        node_b = dag_a.add_node("a_2")
        dag_a.add_edge(node_a, node_b, "edge_1")
        node_c = dag_a.add_node("a_3")
        dag_a.add_edge(node_b, node_c, "edge_2")
        dag_a.remove_node(node_b)
        dag_b = copy.deepcopy(dag_a)
        self.assertIsInstance(dag_b, rustworkx.PyDAG)
        self.assertEqual([node_a, node_c], dag_b.node_indexes())

    def test_deepcopy_empty(self):
        dag = rustworkx.PyDAG()
        empty_copy = copy.deepcopy(dag)
        self.assertEqual(len(empty_copy), 0)

    def test_deepcopy_attrs(self):
        graph = rustworkx.PyDiGraph(attrs="abc")
        graph_copy = copy.deepcopy(graph)
        self.assertEqual(graph.attrs, graph_copy.attrs)

    def test_deepcopy_check_cycle(self):
        graph_a = rustworkx.PyDiGraph(check_cycle=True)
        graph_b = copy.deepcopy(graph_a)
        graph_c = rustworkx.PyDiGraph(check_cycle=False)
        graph_d = copy.deepcopy(graph_c)
        self.assertTrue(graph_b.check_cycle)
        self.assertFalse(graph_d.check_cycle)

    def test_deepcopy_different_objects(self):
        graph_a = rustworkx.PyDiGraph(attrs=[1])
        node_a = graph_a.add_node([2])
        node_b = graph_a.add_child(node_a, [3], [4])
        graph_b = copy.deepcopy(graph_a)
        self.assertEqual(graph_a.attrs, graph_b.attrs)
        self.assertIsNot(graph_a.attrs, graph_b.attrs)
        self.assertEqual(graph_a[node_a], graph_b[node_a])
        self.assertIsNot(graph_a[node_a], graph_b[node_a])
        self.assertEqual(
            graph_a.get_edge_data(node_a, node_b), graph_b.get_edge_data(node_a, node_b)
        )
        self.assertIsNot(
            graph_a.get_edge_data(node_a, node_b), graph_b.get_edge_data(node_a, node_b)
        )

    def test_deepcopy_multinode_hole_in_middle(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(20))
        graph.remove_nodes_from([10, 11, 12, 13, 14])
        graph.add_edges_from_no_data(
            [
                (4, 5),
                (16, 18),
                (2, 19),
                (0, 15),
                (15, 16),
                (16, 17),
                (6, 17),
                (8, 18),
                (17, 1),
                (17, 7),
                (18, 3),
                (18, 9),
                (19, 16),
            ]
        )
        copied_graph = copy.deepcopy(graph)
        self.assertEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19], copied_graph.node_indices()
        )
