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
    def test_deepcopy_returns_graph(self):
        dag_a = rustworkx.PyGraph()
        node_a = dag_a.add_node("a_1")
        node_b = dag_a.add_node("a_2")
        dag_a.add_edge(node_a, node_b, "edge_1")
        node_c = dag_a.add_node("a_3")
        dag_a.add_edge(node_b, node_c, "edge_2")
        dag_b = copy.deepcopy(dag_a)
        self.assertIsInstance(dag_b, rustworkx.PyGraph)

    def test_deepcopy_with_holes_returns_graph(self):
        dag_a = rustworkx.PyGraph()
        node_a = dag_a.add_node("a_1")
        node_b = dag_a.add_node("a_2")
        dag_a.add_edge(node_a, node_b, "edge_1")
        node_c = dag_a.add_node("a_3")
        dag_a.add_edge(node_b, node_c, "edge_2")
        dag_a.remove_node(node_b)
        dag_b = copy.deepcopy(dag_a)
        self.assertIsInstance(dag_b, rustworkx.PyGraph)
        self.assertEqual([node_a, node_c], dag_b.node_indexes())

    def test_deepcopy_empty(self):
        dag = rustworkx.PyGraph()
        empty_copy = copy.deepcopy(dag)
        self.assertEqual(len(empty_copy), 0)

    def test_deepcopy_attrs(self):
        graph = rustworkx.PyGraph(attrs="abc")
        graph_copy = copy.deepcopy(graph)
        self.assertEqual(graph.attrs, graph_copy.attrs)

    def test_deepcopy_multinode_hole_in_middle(self):
        graph = rustworkx.PyGraph()
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
