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
import itertools
import unittest
import rustworkx


def bruteforce(g, k):
    connected_subgraphs = []
    for sg in (
        g.subgraph(selected_nodes) for selected_nodes in itertools.combinations(g.node_indices(), k)
    ):
        if rustworkx.is_connected(sg):
            connected_subgraphs.append(list(sg.nodes()))
    return connected_subgraphs


class TestGraphAllConnectedSubgraphs(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (4, 5), (4, 7), (7, 6), (5, 6)]
        self.nodes = list(range(8))
        g = rustworkx.PyGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from_no_data(self.edges)
        self.expected_subgraphs = {k: list(bruteforce(g, k)) for k in range(1, 9)}

    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        subgraphs = rustworkx.connected_subgraphs(graph, 0)
        expected = []
        self.assertConnectedSubgraphsEqual(subgraphs, expected)

    def test_empty_graph_2(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from_no_data(self.edges)
        subgraphs = rustworkx.connected_subgraphs(graph, 0)
        expected = []
        self.assertConnectedSubgraphsEqual(subgraphs, expected)

    def test_size_one_subgraphs(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from_no_data(self.edges)
        subgraphs = rustworkx.connected_subgraphs(graph, 1)
        self.assertConnectedSubgraphsEqual(subgraphs, self.expected_subgraphs[1])

    def test_sized_subgraphs(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from_no_data(self.edges)
        for i in range(2, 9):
            with self.subTest(subgraph_size=i):
                subgraphs = rustworkx.connected_subgraphs(graph, i)
                self.assertConnectedSubgraphsEqual(subgraphs, self.expected_subgraphs[i])

    def test_unique_subgraphs(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from_no_data(self.edges)
        for i in range(2, 9):
            with self.subTest(subgraph_size=i):
                subgraphs = rustworkx.connected_subgraphs(graph, i)
                self.assertEqual(len(subgraphs), len({tuple(sorted(el)) for el in subgraphs}))

    def test_disconnected_graph(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from([0, 1, 2, 3, 4])
        graph.add_edge(0, 1, None)
        graph.add_edge(1, 2, None)
        graph.add_edge(0, 2, None)

        graph.add_edge(3, 4, None)

        self.assertConnectedSubgraphsEqual(
            rustworkx.connected_subgraphs(graph, 1), [[n] for n in graph.nodes()]
        )
        self.assertConnectedSubgraphsEqual(
            rustworkx.connected_subgraphs(graph, 2), graph.edge_list()
        )
        self.assertConnectedSubgraphsEqual(rustworkx.connected_subgraphs(graph, 3), [[0, 1, 2]])
        self.assertConnectedSubgraphsEqual(rustworkx.connected_subgraphs(graph, 4), [])

    def assertConnectedSubgraphsEqual(self, subgraphs, expected):
        self.assertEqual(
            {tuple(sorted(el)) for el in subgraphs}, {tuple(sorted(el)) for el in expected}
        )
