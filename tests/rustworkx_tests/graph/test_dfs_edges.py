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
    def test_graph_dfs_edges(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
        edges = rustworkx.graph_dfs_edges(graph, 0)
        expected = [(0, 1), (1, 2), (2, 4), (4, 3)]
        self.assertEqual(expected, edges)

    def test_graph_disconnected_dfs_edges(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list([(0, 1), (2, 3)])
        edges = rustworkx.graph_dfs_edges(graph)
        expected = [(0, 1), (2, 3)]
        self.assertEqual(expected, edges)
