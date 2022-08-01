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


class TestUnion(unittest.TestCase):
    def test_union_merge_all(self):
        dag_a = rustworkx.PyDiGraph()
        dag_b = rustworkx.PyDiGraph()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "e_1")
        dag_a.add_child(node_a, "a_3", "e_2")

        node_b = dag_b.add_node("a_1")
        dag_b.add_child(node_b, "a_2", "e_1")
        dag_b.add_child(node_b, "a_3", "e_2")

        dag_c = rustworkx.digraph_union(dag_a, dag_b, True, True)

        self.assertTrue(rustworkx.is_isomorphic(dag_a, dag_c))

    def test_union_basic_merge_nodes_only(self):
        dag_a = rustworkx.PyDiGraph()
        dag_b = rustworkx.PyDiGraph()

        node_a = dag_a.add_node("a_1")
        child_a = dag_a.add_child(node_a, "a_2", "e_1")
        dag_a.add_child(node_a, "a_3", "e_2")

        node_b = dag_b.add_node("a_1")
        dag_b.add_child(node_b, "a_2", "e_1")
        dag_b.add_child(node_b, "a_3", "e_2")

        dag_c = rustworkx.digraph_union(dag_a, dag_b, True, False)

        self.assertTrue(len(dag_c.edge_list()) == 4)
        self.assertTrue(len(dag_c.get_all_edge_data(node_a, child_a)) == 2)
        self.assertTrue(len(dag_c.nodes()) == 3)

    def test_union_basic_merge_none(self):
        dag_a = rustworkx.PyDiGraph()
        dag_b = rustworkx.PyDiGraph()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "e_1")
        dag_a.add_child(node_a, "a_3", "r_2")

        node_b = dag_b.add_node("a_1")
        dag_b.add_child(node_b, "a_2", "e_1")
        dag_b.add_child(node_b, "a_3", "e_2")

        dag_c = rustworkx.digraph_union(dag_a, dag_b, False, False)

        self.assertTrue(len(dag_c.nodes()) == 6)
        self.assertTrue(len(dag_c.edge_list()) == 4)

    def test_union_mismatch_edge_weight(self):
        first = rustworkx.PyDiGraph()
        nodes = first.add_nodes_from([0, 1])
        first.add_edges_from([(nodes[0], nodes[1], "a")])

        second = rustworkx.PyDiGraph()
        nodes = second.add_nodes_from([0, 1])
        second.add_edges_from([(nodes[0], nodes[1], "b")])

        final = rustworkx.digraph_union(first, second, merge_nodes=True, merge_edges=True)
        self.assertEqual(final.weighted_edge_list(), [(0, 1, "a"), (0, 1, "b")])

    def test_union_node_hole(self):
        first = rustworkx.PyDiGraph()
        nodes = first.add_nodes_from([0, 1])
        first.add_edges_from([(nodes[0], nodes[1], "a")])

        second = rustworkx.PyDiGraph()
        dummy = second.add_node("dummy")
        nodes = second.add_nodes_from([0, 1])
        second.add_edges_from([(nodes[0], nodes[1], "a")])
        second.remove_node(dummy)

        final = rustworkx.digraph_union(first, second, merge_nodes=True, merge_edges=True)
        self.assertEqual(final.weighted_edge_list(), [(0, 1, "a")])

    def test_union_edge_between_merged_and_unmerged_nodes(self):
        first = rustworkx.PyDiGraph()
        nodes = first.add_nodes_from([0, 1])
        first.add_edges_from([(nodes[0], nodes[1], "a")])

        second = rustworkx.PyDiGraph()
        nodes = second.add_nodes_from([0, 2])
        second.add_edges_from([(nodes[0], nodes[1], "b")])

        final = rustworkx.digraph_union(first, second, merge_nodes=True, merge_edges=True)
        self.assertEqual(final.weighted_edge_list(), [(0, 1, "a"), (0, 2, "b")])
