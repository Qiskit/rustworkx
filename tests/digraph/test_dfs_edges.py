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


class TestDfsEdges(unittest.TestCase):
    def test_digraph_disconnected_dfs_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.extend_from_edge_list([(0, 1), (2, 3)])
        edges = rustworkx.digraph_dfs_edges(graph)
        expected = [(0, 1), (2, 3)]
        self.assertEqual(expected, edges)

    def test_digraph_dfs_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.extend_from_edge_list([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
        edges = rustworkx.digraph_dfs_edges(graph, 0)
        expected = [(0, 1), (1, 2), (2, 4), (1, 3)]
        self.assertEqual(expected, edges)

    def test_digraph_dfs_edges_empty(self):
        graph = rustworkx.PyDiGraph()
        edges = rustworkx.digraph_dfs_edges(graph)
        self.assertEqual([], edges)

    def test_digraph_dfs_edges_single_node(self):
        graph = rustworkx.generators.directed_empty_graph(1)
        edges = rustworkx.digraph_dfs_edges(graph, 0)
        self.assertEqual([], edges)

    def test_digraph_dfs_edges_node_gaps(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(5))
        graph.add_edge(0, 2, None)
        graph.add_edge(2, 4, None)
        graph.remove_node(1)
        graph.remove_node(3)
        edges = rustworkx.digraph_dfs_edges(graph, 0)
        self.assertEqual([(0, 2), (2, 4)], edges)

    def test_digraph_dfs_edges_star(self):
        graph = rustworkx.generators.directed_star_graph(101)
        hub = 0
        spokes = list(range(1, 101))
        edges = rustworkx.digraph_dfs_edges(graph, hub)
        # Should visit all spokes exactly once
        self.assertEqual(len(edges), 100)
        # All edges should originate from hub
        for src, _ in edges:
            self.assertEqual(src, hub)
        # All spokes should be visited
        visited = {tgt for _, tgt in edges}
        self.assertEqual(visited, set(spokes))
