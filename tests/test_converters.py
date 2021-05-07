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
import retworkx
import networkx


class TestNetworkxConverter(unittest.TestCase):
    def test_undirected_gnm_graph(self):
        g = networkx.gnm_random_graph(10, 10, seed=42)
        out_graph = retworkx.networkx_converter(g)
        self.assertIsInstance(out_graph, retworkx.PyGraph)
        self.assertEqual(out_graph.nodes(), list(g.nodes))
        self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
        self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_directed_gnm_graph(self):
        g = networkx.gnm_random_graph(10, 10, seed=42, directed=True)
        out_graph = retworkx.networkx_converter(g)
        self.assertIsInstance(out_graph, retworkx.PyDiGraph)
        self.assertEqual(out_graph.nodes(), list(g.nodes))
        self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
        self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_empty_graph(self):
        g = networkx.Graph()
        out_graph = retworkx.networkx_converter(g)
        self.assertIsInstance(out_graph, retworkx.PyGraph)
        self.assertEqual(out_graph.nodes(), list(g.nodes))
        self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
        self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_empty_multigraph(self):
        g = networkx.MultiGraph()
        out_graph = retworkx.networkx_converter(g)
        self.assertIsInstance(out_graph, retworkx.PyGraph)
        self.assertEqual(out_graph.nodes(), list(g.nodes))
        self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
        self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_empty_directed_graph(self):
        g = networkx.DiGraph()
        out_graph = retworkx.networkx_converter(g)
        self.assertIsInstance(out_graph, retworkx.PyDiGraph)
        self.assertEqual(out_graph.nodes(), list(g.nodes))
        self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
        self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_empty_directed_multigraph(self):
        g = networkx.MultiDiGraph()
        out_graph = retworkx.networkx_converter(g)
        self.assertIsInstance(out_graph, retworkx.PyDiGraph)
        self.assertEqual(out_graph.nodes(), list(g.nodes))
        self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
        self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_cubical_graph(self):
        g = networkx.cubical_graph(networkx.Graph)
        out_graph = retworkx.networkx_converter(g)
        self.assertIsInstance(out_graph, retworkx.PyGraph)
        self.assertEqual(out_graph.nodes(), list(g.nodes))
        self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
        self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_cubical_multigraph(self):
        g = networkx.cubical_graph(networkx.MultiGraph)
        out_graph = retworkx.networkx_converter(g)
        self.assertIsInstance(out_graph, retworkx.PyGraph)
        self.assertEqual(out_graph.nodes(), list(g.nodes))
        self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
        self.assertEqual(out_graph.multigraph, g.is_multigraph())

    def test_random_k_out_graph(self):
        g = networkx.random_k_out_graph(100, 50, 3.14159, True, 42)
        out_graph = retworkx.networkx_converter(g)
        self.assertIsInstance(out_graph, retworkx.PyDiGraph)
        self.assertEqual(out_graph.nodes(), list(g.nodes))
        self.assertEqual(out_graph.weighted_edge_list(), list(g.edges(data=True)))
        self.assertEqual(out_graph.multigraph, g.is_multigraph())
