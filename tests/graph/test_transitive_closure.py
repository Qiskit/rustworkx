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


class TestTransitive(unittest.TestCase):
    def test_transitive_closure(self):

        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.add_edge(0, 1, ())
        graph.add_edge(1, 2, ())
        graph.add_edge(2, 0, ())
        graph.add_edge(2, 3, ())

        closure_graph = rustworkx.transitive_closure(graph)
        self.expected_edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 3),
        ]

        self.assertEqualEdgeList(self.expected_edges, closure_graph.edge_list())

    def test_transitive_closure_single_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(())
        closure_graph = rustworkx.transitive_closure(graph)
        expected_edges = []
        self.assertEqualEdgeList(expected_edges, closure_graph.edge_list())

    def test_transitive_closure_no_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        closure_graph = rustworkx.transitive_closure(graph)
        expected_edges = []
        self.assertEqualEdgeList(expected_edges, closure_graph.edge_list())

    def test_transitive_closure_complete_graph(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        for i in range(4):
            for j in range(4):
                if i != j:
                    graph.add_edge(i, j, ())
        closure_graph = rustworkx.transitive_closure(graph)
        expected_edges = [(i, j) for i in range(4) for j in range(4) if i != j]
        self.assertEqualEdgeList(expected_edges, closure_graph.edge_list())

    def assertEqualEdgeList(self, expected, actual):
        for edge in actual:
            self.assertTrue(edge in expected)