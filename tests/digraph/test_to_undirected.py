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

import operator as op
import unittest

import rustworkx


class TestToUndirected(unittest.TestCase):
    def test_to_undirected_empty_graph(self):
        digraph = rustworkx.PyDiGraph()
        graph = digraph.to_undirected()
        self.assertEqual(0, len(graph))

    def test_single_direction_graph(self):
        digraph = rustworkx.generators.directed_path_graph(5)
        graph = digraph.to_undirected()
        self.assertEqual(digraph.weighted_edge_list(), graph.weighted_edge_list())

    def test_bidirectional_graph(self):
        digraph = rustworkx.generators.directed_path_graph(5)
        for i in range(0, 4):
            digraph.add_edge(i + 1, i, None)
        graph = digraph.to_undirected()
        self.assertEqual(digraph.weighted_edge_list(), graph.weighted_edge_list())

    def test_bidirectional_not_multigraph(self):
        digraph = rustworkx.generators.directed_path_graph(5)
        for i in range(0, 4):
            digraph.add_edge(i + 1, i, None)
        graph = digraph.to_undirected(multigraph=False)
        self.assertEqual(graph.edge_list(), [(0, 1), (1, 2), (2, 3), (3, 4)])

    def test_multiple_edges_combo_weight_not_multigraph(self):
        digraph = rustworkx.PyDiGraph()
        digraph.add_nodes_from([0, 1])
        digraph.add_edges_from([(0, 1, "a"), (0, 1, "b")])
        graph = digraph.to_undirected(multigraph=False, weight_combo_fn=op.add)
        self.assertEqual(graph.weighted_edge_list(), [(0, 1, "ab")])

    def test_shared_ref(self):
        digraph = rustworkx.PyDiGraph()
        node_weight = {"a": 1}
        node_a = digraph.add_node(node_weight)
        edge_weight = {"a": 1}
        digraph.add_child(node_a, "b", edge_weight)
        graph = digraph.to_undirected()
        self.assertEqual(digraph[node_a], {"a": 1})
        self.assertEqual(graph[node_a], {"a": 1})
        node_weight["b"] = 2
        self.assertEqual(digraph[node_a], {"a": 1, "b": 2})
        self.assertEqual(graph[node_a], {"a": 1, "b": 2})
        self.assertEqual(digraph.get_edge_data(0, 1), {"a": 1})
        self.assertEqual(graph.get_edge_data(0, 1), {"a": 1})
        edge_weight["b"] = 2
        self.assertEqual(digraph.get_edge_data(0, 1), {"a": 1, "b": 2})
        self.assertEqual(graph.get_edge_data(0, 1), {"a": 1, "b": 2})
