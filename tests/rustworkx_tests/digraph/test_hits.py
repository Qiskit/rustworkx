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

# These tests are adapated from the networkx test cases:
# https://github.com/networkx/networkx/blob/cea310f9066efc0d5ff76f63d33dbc3eefe61f6b/networkx/algorithms/link_analysis/tests/test_pagerank.py

import unittest

import rustworkx
import networkx as nx


class TestHits(unittest.TestCase):
    def setUp(self):
        try:
            # required for networkx.hits to work
            import scipy

            self.assertIsNotNone(scipy.__version__)
        except ModuleNotFoundError:
            self.skipTest("SciPy is not installed, skipping HITS tests")

    def test_hits(self):
        edges = [(0, 2), (0, 4), (1, 0), (2, 4), (4, 3), (4, 2), (5, 4)]

        rx_graph = rustworkx.PyDiGraph()
        rx_graph.extend_from_edge_list(edges)

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(edges)

        rx_h, rx_a = rustworkx.hits(rx_graph)
        nx_h, nx_a = nx.hits(nx_graph)

        for v in rx_graph.node_indices():
            self.assertAlmostEqual(rx_h[v], nx_h[v], delta=1.0e-4)
            self.assertAlmostEqual(rx_a[v], nx_a[v], delta=1.0e-4)

    def test_no_convergence(self):
        graph = rustworkx.generators.directed_path_graph(4)
        with self.assertRaises(rustworkx.FailedToConverge):
            rustworkx.hits(graph, max_iter=0)

    def test_normalized(self):
        graph = rustworkx.generators.directed_complete_graph(2)
        h, a = rustworkx.hits(graph, normalized=False)
        self.assertEqual({0: 1, 1: 1}, h)
        self.assertEqual({0: 1, 1: 1}, a)

    def test_multi_digraph_versus_weighted(self):
        multi_graph = rustworkx.PyDiGraph()
        multi_graph.extend_from_edge_list(
            [
                (0, 1),
                (1, 0),
                (0, 1),
                (1, 0),
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 2),
                (2, 3),
                (3, 2),
            ]
        )

        weighted_graph = rustworkx.PyDiGraph()
        weighted_graph.extend_from_weighted_edge_list(
            [(0, 1, 3), (1, 0, 3), (1, 2, 2), (2, 1, 2), (2, 3, 2), (3, 2, 2)]
        )

        h_multi, a_multi = rustworkx.hits(multi_graph, weight_fn=lambda _: 1.0)
        h_weight, a_weight = rustworkx.hits(weighted_graph, weight_fn=float)

        for v in multi_graph.node_indices():
            self.assertAlmostEqual(h_multi[v], h_weight[v], delta=1.0e-4)
            self.assertAlmostEqual(a_multi[v], a_weight[v], delta=1.0e-4)

    def test_nstart(self):
        graph = rustworkx.generators.directed_complete_graph(10)
        nstart = {5: 1, 6: 1}  # this guess is worse than the uniform guess =)
        h, a = rustworkx.hits(graph, nstart=nstart)

        for v in graph.node_indices():
            self.assertAlmostEqual(h[v], 1 / 10.0, delta=1.0e-4)
            self.assertAlmostEqual(a[v], 1 / 10.0, delta=1.0e-4)
