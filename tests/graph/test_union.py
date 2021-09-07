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


class TestUnion(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyGraph()
        self.graph.add_nodes_from(["a_1", "a_2", "a_3"])
        self.graph.extend_from_weighted_edge_list([(0, 1, "e_1"), (1, 2, "e_2")])
    
    def test_union_basic_merge_none(self):
        final = retworkx.graph_union(
            self.graph, self.graph, merge_nodes=False, merge_edges=False
        )
        self.assertTrue(len(final.nodes()) == 6)
        self.assertTrue(len(final.edge_list()) == 4)

    def test_union_merge_all(self):
        final = retworkx.graph_union(
            self.graph, self.graph, merge_nodes=True, merge_edges=True
        )
        self.assertTrue(retworkx.is_isomorphic(final, self.graph))

    def test_union_basic_merge_nodes_only(self):
        final = retworkx.graph_union(
            self.graph, self.graph, merge_nodes=True, merge_edges=False
        )
        self.assertTrue(len(final.edge_list()) == 4)
        self.assertTrue(len(final.get_all_edge_data(0, 1)) == 2)
        self.assertTrue(len(final.nodes()) == 3)

    def test_union_mismatch_edge_weight(self):
        first = retworkx.PyGraph()
        nodes = first.add_nodes_from([0, 1])
        first.add_edges_from([(nodes[0], nodes[1], "a")])

        second = retworkx.PyGraph()
        nodes = second.add_nodes_from([0, 1])
        second.add_edges_from([(nodes[0], nodes[1], "b")])

        final = retworkx.graph_union(
            first, second, merge_nodes=True, merge_edges=True
        )
        self.assertEqual(final.weighted_edge_list(), [(0, 1, "a"), (0, 1, "b")])

    def test_union_node_hole(self):
        first = retworkx.PyGraph()
        nodes = first.add_nodes_from([0, 1])
        first.add_edges_from([(nodes[0], nodes[1], "a")])

        second = retworkx.PyGraph()
        dummy = second.add_node("dummy")
        nodes = second.add_nodes_from([0, 1])
        second.add_edges_from([(nodes[0], nodes[1], "a")])
        second.remove_node(dummy)

        final = retworkx.graph_union(
            first, second, merge_nodes=True, merge_edges=True
        )
        self.assertEqual(final.weighted_edge_list(), [(0, 1, "a")])

    def test_union_edge_between_merged_and_unmerged_nodes(self):
        first = retworkx.PyGraph()
        nodes = first.add_nodes_from([0, 1])
        first.add_edges_from([(nodes[0], nodes[1], "a")])

        second = retworkx.PyGraph()
        nodes = second.add_nodes_from([0, 2])
        second.add_edges_from([(nodes[0], nodes[1], "b")])

        final = retworkx.graph_union(
            first, second, merge_nodes=True, merge_edges=True
        )
        self.assertEqual(final.weighted_edge_list(), [(0, 1, "a"), (0, 2, "b")])