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
