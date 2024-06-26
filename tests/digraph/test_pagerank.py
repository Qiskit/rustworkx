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
#
# NetworkX is distributed with the 3-clause BSD license.
#
#   Copyright (C) 2004-2020, NetworkX Developers
#   Aric Hagberg <hagberg@lanl.gov>
#   Dan Schult <dschult@colgate.edu>
#   Pieter Swart <swart@lanl.gov>
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of the NetworkX Developers nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# These tests are adapated from the networkx test cases:
# https://github.com/networkx/networkx/blob/cea310f9066efc0d5ff76f63d33dbc3eefe61f6b/networkx/algorithms/link_analysis/tests/test_pagerank.py

import unittest

import rustworkx
import networkx as nx


def pagerank_python(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    if len(G) == 0:
        return {}

    D = G.to_directed()

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = sum(nstart.values())
        x = {k: v / s for k, v in nstart.items()}

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = sum(personalization.values())
        p = {k: v / s for k, v in personalization.items()}

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        s = sum(dangling.values())
        dangling_weights = {k: v / s for k, v in dangling.items()}
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for _, nbr, wt in W.edges(n, data=weight):
                x[nbr] += alpha * xlast[n] * wt
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        # check convergence, l1 norm
        err = sum(abs(x[n] - xlast[n]) for n in x)
        if err < N * tol:
            return x
    raise ValueError(max_iter)


class TestPageRank(unittest.TestCase):
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
        nx_ranks = pagerank_python(nx_graph, alpha=alpha, tol=tol)

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
        nx_ranks = pagerank_python(nx_graph, alpha=alpha, tol=tol, dangling=dangling)

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
        nx_ranks = pagerank_python(nx_graph, alpha=alpha, nstart=nstart)

        for v in rx_graph.node_indices():
            self.assertAlmostEqual(rx_ranks[v], nx_ranks[v], delta=1.0e-4)

    def test_pagerank_with_personalize(self):
        rx_graph = rustworkx.generators.directed_complete_graph(4)
        personalize = {0: 0, 1: 0, 2: 0, 3: 1}
        alpha = 0.85
        rx_ranks = rustworkx.pagerank(rx_graph, alpha=alpha, personalization=personalize)
        nx_graph = nx.DiGraph(list(rx_graph.edge_list()))
        nx_ranks = pagerank_python(nx_graph, alpha=alpha, personalization=personalize)

        for v in rx_graph.node_indices():
            self.assertAlmostEqual(rx_ranks[v], nx_ranks[v], delta=1.0e-4)

    def test_pagerank_with_personalize_missing(self):
        rx_graph = rustworkx.generators.directed_complete_graph(4)
        personalize = {3: 1}
        alpha = 0.85
        rx_ranks = rustworkx.pagerank(rx_graph, alpha=alpha, personalization=personalize)
        nx_graph = nx.DiGraph(list(rx_graph.edge_list()))
        nx_ranks = pagerank_python(nx_graph, alpha=alpha, personalization=personalize)

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
        nx_ranks = pagerank_python(nx_graph, alpha=alpha)

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
