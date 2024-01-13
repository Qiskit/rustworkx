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


class TestPageRank(unittest.TestCase):
    def setUp(self) -> None:
        try:
            # required for networkx.pagerank to work
            import scipy

            self.assertIsNotNone(scipy.__version__)
        except ModuleNotFoundError:
            self.skipTest("SciPy is not installed, skipping PageRank tests")

    def test_with_dangling_node(self):
        edges = [
            (0, 1),
            (0, 2),
            (2, 0),
            (2, 1),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 3),
            (4, 5),
            (5, 4),
        ]  # node 1 is dangling because it does not point to anyone

        rx_graph = rustworkx.PyDiGraph()
        nx_graph = nx.DiGraph()

        rx_graph.extend_from_edge_list(edges)
        nx_graph.add_edges_from(edges)

        alpha = 0.9
        tol = 1.0e-8

        rx_ranks = rustworkx.pagerank(rx_graph, alpha=alpha, tol=tol)
        nx_ranks = nx.pagerank(nx_graph, alpha=alpha, tol=tol)

        for v in rx_graph.node_indices():
            self.assertAlmostEqual(rx_ranks[v], nx_ranks[v], delta=1.0e-4)

    def test_with_dangling_node_and_argument(self):
        edges = [
            (0, 1),
            (0, 2),
            (2, 0),
            (2, 1),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 3),
            (4, 5),
            (5, 4),
        ]  # node 1 is dangling because it does not point to anyone

        rx_graph = rustworkx.PyDiGraph()
        nx_graph = nx.DiGraph()

        rx_graph.extend_from_edge_list(edges)
        nx_graph.add_edges_from(edges)

        dangling = {0: 0, 1: 1, 2: 2, 3: 0, 5: 0}

        alpha = 0.85
        tol = 1.0e-8

        rx_ranks = rustworkx.pagerank(rx_graph, alpha=alpha, tol=tol, dangling=dangling)
        nx_ranks = nx.pagerank(nx_graph, alpha=alpha, tol=tol, dangling=dangling)

        for v in rx_graph.node_indices():
            self.assertAlmostEqual(rx_ranks[v], nx_ranks[v], delta=1.0e-4)

    def test_empty(self):
        graph = rustworkx.PyDiGraph()
        ranks = rustworkx.pagerank(graph)
        self.assertEqual({}, ranks)

    def test_one_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        ranks = rustworkx.pagerank(graph)
        self.assertEqual({0: 1}, ranks)

    def test_cycle_graph(self):
        graph = rustworkx.generators.directed_cycle_graph(100)
        ranks = rustworkx.pagerank(graph)

        for v in graph.node_indices():
            self.assertAlmostEqual(ranks[v], 1 / 100.0, delta=1.0e-4)

    def test_with_removed_node(self):
        graph = rustworkx.PyDiGraph()

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 0),
            (4, 1),
            (4, 2),
            (0, 4),
        ]
        graph.extend_from_edge_list(edges)
        graph.remove_node(3)

        ranks = rustworkx.pagerank(graph)

        expected_ranks = {
            0: 0.17401467654615052,
            1: 0.2479710438690554,
            2: 0.3847906219106203,
            4: 0.19322365767417365,
        }

        for v in graph.node_indices():
            self.assertAlmostEqual(ranks[v], expected_ranks[v], delta=1.0e-4)

    def test_pagerank_with_nstart(self):
        rx_graph = rustworkx.generators.directed_complete_graph(4)
        nstart = {0: 0.5, 1: 0.5, 2: 0, 3: 0}
        alpha = 0.85
        rx_ranks = rustworkx.pagerank(rx_graph, alpha=alpha, nstart=nstart)
        nx_graph = nx.DiGraph(list(rx_graph.edge_list()))
        nx_ranks = nx.pagerank(nx_graph, alpha=alpha, nstart=nstart)

        for v in rx_graph.node_indices():
            self.assertAlmostEqual(rx_ranks[v], nx_ranks[v], delta=1.0e-4)

    def test_pagerank_with_personalize(self):
        rx_graph = rustworkx.generators.directed_complete_graph(4)
        personalize = {0: 0, 1: 0, 2: 0, 3: 1}
        alpha = 0.85
        rx_ranks = rustworkx.pagerank(rx_graph, alpha=alpha, personalization=personalize)
        nx_graph = nx.DiGraph(list(rx_graph.edge_list()))
        nx_ranks = nx.pagerank(nx_graph, alpha=alpha, personalization=personalize)

        for v in rx_graph.node_indices():
            self.assertAlmostEqual(rx_ranks[v], nx_ranks[v], delta=1.0e-4)

    def test_pagerank_with_personalize_missing(self):
        rx_graph = rustworkx.generators.directed_complete_graph(4)
        personalize = {3: 1}
        alpha = 0.85
        rx_ranks = rustworkx.pagerank(rx_graph, alpha=alpha, personalization=personalize)
        nx_graph = nx.DiGraph(list(rx_graph.edge_list()))
        nx_ranks = nx.pagerank(nx_graph, alpha=alpha, personalization=personalize)

        for v in rx_graph.node_indices():
            self.assertAlmostEqual(rx_ranks[v], nx_ranks[v], delta=1.0e-4)

    def test_multi_digraph(self):
        rx_graph = rustworkx.PyDiGraph()
        rx_graph.extend_from_edge_list(
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
        nx_graph = nx.MultiDiGraph(list(rx_graph.edge_list()))

        alpha = 0.9
        rx_ranks = rustworkx.pagerank(rx_graph, alpha=alpha)
        nx_ranks = nx.pagerank(nx_graph, alpha=alpha)

        for v in rx_graph.node_indices():
            self.assertAlmostEqual(rx_ranks[v], nx_ranks[v], delta=1.0e-4)

    def test_no_convergence(self):
        graph = rustworkx.generators.directed_complete_graph(4)
        with self.assertRaises(rustworkx.FailedToConverge):
            rustworkx.pagerank(graph, max_iter=0)

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

        alpha = 0.85
        ranks_multi = rustworkx.pagerank(multi_graph, alpha=alpha, weight_fn=lambda _: 1.0)
        ranks_weight = rustworkx.pagerank(weighted_graph, alpha=alpha, weight_fn=float)

        for v in multi_graph.node_indices():
            self.assertAlmostEqual(ranks_multi[v], ranks_weight[v], delta=1.0e-4)
