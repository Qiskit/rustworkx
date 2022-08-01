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
import networkx


class TestNetworkxConverter(unittest.TestCase):
    def test_undirected_gnm_graph(self):
        g = networkx.gnm_random_graph(10, 10, seed=42)
        for keep_attributes in [True, False]:
            with self.subTest(keep_attributes=keep_attributes):
                out_graph = rustworkx.networkx_converter(g, keep_attributes=keep_attributes)
                self.assertIsInstance(out_graph, rustworkx.PyGraph)
                self.assertEqual(list(out_graph.node_indexes()), list(g.nodes))
                self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
                self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_directed_gnm_graph(self):
        g = networkx.gnm_random_graph(10, 10, seed=42, directed=True)
        for keep_attributes in [True, False]:
            with self.subTest(keep_attributes=keep_attributes):
                out_graph = rustworkx.networkx_converter(g, keep_attributes=keep_attributes)
                self.assertIsInstance(out_graph, rustworkx.PyDiGraph)
                self.assertEqual(list(out_graph.node_indexes()), list(g.nodes))
                self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
                self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_empty_graph(self):
        g = networkx.Graph()
        for keep_attributes in [True, False]:
            with self.subTest(keep_attributes=keep_attributes):
                out_graph = rustworkx.networkx_converter(g, keep_attributes=keep_attributes)
                self.assertIsInstance(out_graph, rustworkx.PyGraph)
                self.assertEqual(list(out_graph.node_indexes()), list(g.nodes))
                self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
                self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_empty_multigraph(self):
        g = networkx.MultiGraph()
        for keep_attributes in [True, False]:
            with self.subTest(keep_attributes=keep_attributes):
                out_graph = rustworkx.networkx_converter(g, keep_attributes=keep_attributes)
                self.assertIsInstance(out_graph, rustworkx.PyGraph)
                self.assertEqual(list(out_graph.node_indexes()), list(g.nodes))
                self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
                self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_empty_directed_graph(self):
        g = networkx.DiGraph()
        for keep_attributes in [True, False]:
            with self.subTest(keep_attributes=keep_attributes):
                out_graph = rustworkx.networkx_converter(g, keep_attributes=keep_attributes)
                self.assertIsInstance(out_graph, rustworkx.PyDiGraph)
                self.assertEqual(list(out_graph.node_indexes()), list(g.nodes))
                self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
                self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_empty_directed_multigraph(self):
        g = networkx.MultiDiGraph()
        for keep_attributes in [True, False]:
            with self.subTest(keep_attributes=keep_attributes):
                out_graph = rustworkx.networkx_converter(g, keep_attributes=keep_attributes)
                self.assertIsInstance(out_graph, rustworkx.PyDiGraph)
                self.assertEqual(list(out_graph.node_indexes()), list(g.nodes))
                self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
                self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_cubical_graph(self):
        g = networkx.cubical_graph(networkx.Graph)
        for keep_attributes in [True, False]:
            with self.subTest(keep_attributes=keep_attributes):
                out_graph = rustworkx.networkx_converter(g, keep_attributes=keep_attributes)
                self.assertIsInstance(out_graph, rustworkx.PyGraph)
                self.assertEqual(list(out_graph.node_indexes()), list(g.nodes))
                self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
                self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_cubical_multigraph(self):
        g = networkx.cubical_graph(networkx.MultiGraph)
        for keep_attributes in [True, False]:
            with self.subTest(keep_attributes=keep_attributes):
                out_graph = rustworkx.networkx_converter(g, keep_attributes=keep_attributes)
                self.assertIsInstance(out_graph, rustworkx.PyGraph)
                self.assertEqual(list(out_graph.node_indexes()), list(g.nodes))
                self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
                self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_random_k_out_graph(self):
        g = networkx.random_k_out_graph(100, 50, 3.14159, True, 42)
        for keep_attributes in [True, False]:
            with self.subTest(keep_attributes=keep_attributes):
                out_graph = rustworkx.networkx_converter(g, keep_attributes=keep_attributes)
                self.assertIsInstance(out_graph, rustworkx.PyDiGraph)
                self.assertEqual(list(out_graph.node_indexes()), list(g.nodes))
                self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
                self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_networkx_graph_attributes_are_converted(self):
        g = networkx.Graph()
        for node in range(100):
            g.add_node(str(node), test=True)

        out_graph = rustworkx.networkx_converter(g, keep_attributes=True)
        for node in out_graph.node_indexes():
            self.assertEqual(out_graph[node]["test"], True)
            self.assertEqual(out_graph[node]["__networkx_node__"], str(node))
