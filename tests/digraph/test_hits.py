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


def hits_python(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    if len(G) == 0:
        return {}, {}
    # choose fixed starting vector if not given
    if nstart is None:
        h = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    else:
        h = nstart
        # normalize starting vector
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s
    for _ in range(max_iter):  # power iteration: make up to max_iter iterations
        hlast = h
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)
        # this "matrix multiply" looks odd because it is
        # doing a left multiply a^T=hlast^T*G
        for n in h:
            for nbr in G[n]:
                a[nbr] += hlast[n] * G[n][nbr].get("weight", 1)
        # now multiply h=Ga
        for n in h:
            for nbr in G[n]:
                h[n] += a[nbr] * G[n][nbr].get("weight", 1)
        # normalize vector
        s = 1.0 / max(h.values())
        for n in h:
            h[n] *= s
        # normalize vector
        s = 1.0 / max(a.values())
        for n in a:
            a[n] *= s
        # check convergence, l1 norm
        err = sum(abs(h[n] - hlast[n]) for n in h)
        if err < tol:
            break
    else:
        raise ValueError(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return h, a


class TestHits(unittest.TestCase):
    def test_hits(self):
        edges = [(0, 2), (0, 4), (1, 0), (2, 4), (4, 3), (4, 2), (5, 4)]

        rx_graph = rustworkx.PyDiGraph()
        rx_graph.extend_from_edge_list(edges)

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(edges)

        rx_h, rx_a = rustworkx.hits(rx_graph)
        nx_h, nx_a = hits_python(nx_graph)

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
