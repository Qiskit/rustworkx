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
from copy import copy

import rustworkx


class TestGraphAllConnectedSubgraphs(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 3),
            (0, 4),
            (4, 5),
            (4, 7),
            (7, 6),
            (5, 6)
        ]
        self.nodes = list(range(8))
        self.expected_subgraph_nodes = [[0, 1],
                                        [0, 3],
                                        [0, 4],
                                        [1, 2],
                                        [2, 3],
                                        [4, 5],
                                        [4, 7],
                                        [5, 6],
                                        [6, 7],
                                        [0, 1, 2],
                                        [0, 1, 3],
                                        [0, 1, 4],
                                        [0, 2, 3],
                                        [0, 3, 4],
                                        [0, 4, 5],
                                        [0, 4, 7],
                                        [1, 2, 3],
                                        [4, 5, 7],
                                        [4, 5, 6],
                                        [4, 6, 7],
                                        [5, 6, 7],
                                        [0, 1, 2, 3],
                                        [0, 1, 2, 4],
                                        [0, 1, 3, 4],
                                        [0, 1, 4, 5],
                                        [0, 1, 4, 7],
                                        [0, 2, 3, 4],
                                        [0, 3, 4, 5],
                                        [0, 3, 4, 7],
                                        [0, 4, 5, 7],
                                        [0, 4, 5, 6],
                                        [0, 4, 7, 6],
                                        [4, 5, 7, 6],
                                        [0, 1, 2, 3, 4],
                                        [0, 1, 2, 4, 5],
                                        [0, 1, 2, 4, 7],
                                        [0, 1, 3, 4, 5],
                                        [0, 1, 3, 4, 7],
                                        [0, 1, 4, 5, 7],
                                        [0, 1, 4, 5, 6],
                                        [0, 1, 4, 7, 6],
                                        [0, 2, 3, 4, 5],
                                        [0, 2, 3, 4, 7],
                                        [0, 3, 4, 5, 7],
                                        [0, 3, 4, 5, 6],
                                        [0, 3, 4, 7, 6],
                                        [0, 4, 5, 7, 6],
                                        [0, 1, 2, 3, 4, 5],
                                        [0, 1, 2, 3, 4, 7],
                                        [0, 1, 2, 4, 5, 7],
                                        [0, 1, 2, 4, 5, 6],
                                        [0, 1, 2, 4, 7, 6],
                                        [0, 1, 3, 4, 5, 7],
                                        [0, 1, 3, 4, 5, 6],
                                        [0, 1, 3, 4, 7, 6],
                                        [0, 1, 4, 5, 7, 6],
                                        [0, 2, 3, 4, 5, 7],
                                        [0, 2, 3, 4, 5, 6],
                                        [0, 2, 3, 4, 7, 6],
                                        [0, 3, 4, 5, 7, 6],
                                        [0, 1, 2, 3, 4, 5, 7],
                                        [0, 1, 2, 3, 4, 5, 6],
                                        [0, 1, 2, 3, 4, 7, 6],
                                        [0, 1, 2, 4, 5, 7, 6],
                                        [0, 1, 3, 4, 5, 7, 6],
                                        [0, 2, 3, 4, 5, 7, 6]]
        self.expected_subgraphs = {}
        for node_list in self.expected_subgraph_nodes:
            graph = rustworkx.PyGraph()
            graph.add_nodes_from(node_list)
            graph.nodes()
            node_index_map = {n: i for i, n in enumerate(node_list)}
            for e in self.edges:
                if e[0] in graph.nodes() and e[1] in graph.nodes():
                    graph.add_edge(node_index_map[e[0]], node_index_map[e[1]], None)
            self.expected_subgraphs.setdefault(len(node_list), list()).append(graph.nodes())

        for n in self.nodes:
            g = rustworkx.PyGraph()
            g.add_node(n)
            self.expected_subgraphs.setdefault(1, list()).append(g.nodes())

        for e in enumerate(self.edges):
            g = rustworkx.PyGraph()
            g.add_nodes_from(e)
            node_index_map = {n: i for i, n in enumerate(e)}
            g.add_edge(node_index_map[e[0]], node_index_map[e[1]], None)
            self.expected_subgraphs.setdefault(2, list()).append(g.nodes())

    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        subgraphs = rustworkx.connected_subgraphs(graph, 0)
        expected = []
        self.assertEqual(subgraphs, expected)

    def test_empty_graph_2(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from_no_data(self.edges)
        subgraphs = rustworkx.connected_subgraphs(graph, 0)
        expected = []
        self.assertEqual(subgraphs, expected)

    def test_size_one_subgraphs(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from_no_data(self.edges)
        subgraphs = rustworkx.connected_subgraphs(graph, 1)
        self.assertEqual(subgraphs, self.expected_subgraphs[1])

    def test_size_two_subgraphs(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from_no_data(self.edges)
        subgraphs = rustworkx.connected_subgraphs(graph, 2)
        self.assertEqual(subgraphs, self.expected_subgraphs[1]+self.expected_subgraphs[2])

    def test_size_three_subgraphs(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from_no_data(self.edges)
        subgraphs = rustworkx.connected_subgraphs(graph, 3)
        self.assertEqual(subgraphs, self.expected_subgraphs[1]+self.expected_subgraphs[2]+self.expected_subgraphs[3])

    def test_full_graph(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from_no_data(self.edges)
        subgraphs = rustworkx.connected_subgraphs(graph, 8)
        self.assertEqual(subgraphs, list(self.expected_subgraphs.values()))