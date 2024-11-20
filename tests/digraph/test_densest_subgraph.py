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

import rustworkx as rx


class TestDensestSubgraph(unittest.TestCase):
    def test_simple_grid_three_nodes(self):
        graph = rx.generators.grid_graph(3, 3)
        subgraph, node_map = rx.densest_subgraph_of_size(graph, 3)
        expected_subgraph_edge_list = [(0, 2), (0, 1)]
        self.assertEqual(expected_subgraph_edge_list, subgraph.edge_list())
        self.assertEqual(node_map, {0: 0, 1: 1, 3: 2})

    def test_simple_grid_six_nodes(self):
        graph = rx.generators.grid_graph(3, 3)
        subgraph, node_map = rx.densest_subgraph_of_size(graph, 6)
        expected_subgraph_edge_list = [(5, 2), (5, 3), (3, 0), (3, 4), (4, 1), (2, 0), (0, 1)]
        self.assertEqual(expected_subgraph_edge_list, subgraph.edge_list())
        self.assertEqual(node_map, {7: 0, 8: 1, 6: 2, 4: 3, 5: 4, 3: 5})
