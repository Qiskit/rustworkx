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


class TestToDirected(unittest.TestCase):
    def test_to_undirected_empty_graph(self):
        graph = rustworkx.PyGraph()
        digraph = graph.to_directed()
        self.assertEqual(0, len(digraph))

    def test_path_graph(self):
        graph = rustworkx.generators.path_graph(5)
        digraph = graph.to_directed()
        expected = [
            (0, 1, None),
            (1, 0, None),
            (1, 2, None),
            (2, 1, None),
            (2, 3, None),
            (3, 2, None),
            (3, 4, None),
            (4, 3, None),
        ]
        self.assertEqual(digraph.weighted_edge_list(), expected)

    def test_parallel_edge_graph(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (0, 1, "A"),
                (0, 1, "B"),
                (0, 2, "C"),
                (0, 3, "D"),
            ]
        )
        digraph = graph.to_directed()
        expected = [
            (0, 1, "A"),
            (1, 0, "A"),
            (0, 1, "B"),
            (1, 0, "B"),
            (0, 2, "C"),
            (2, 0, "C"),
            (0, 3, "D"),
            (3, 0, "D"),
        ]
        self.assertEqual(digraph.weighted_edge_list(), expected)

    def test_shared_ref(self):
        graph = rustworkx.PyGraph()
        node_weight = {"a": 1}
        node_a = graph.add_node(node_weight)
        edge_weight = {"a": 1}
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, edge_weight)
        digraph = graph.to_directed()
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
