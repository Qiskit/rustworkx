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


class TestMinimumSpanningTree(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
        self.a = self.graph.add_node("A")
        self.b = self.graph.add_node("B")
        self.c = self.graph.add_node("C")
        self.d = self.graph.add_node("D")
        self.e = self.graph.add_node("E")
        self.f = self.graph.add_node("F")

        edge_list = [
            (self.a, self.b, 3),
            (self.a, self.d, 2),
            (self.b, self.c, 4),
            (self.c, self.d, 1),
            (self.a, self.f, 1),
            (self.b, self.f, 6),
            (self.d, self.e, 5),
            (self.c, self.e, 7),
        ]
        self.graph.add_edges_from(edge_list)

        self.expected_edges = [
            (self.a, self.b, 3),
            (self.a, self.d, 2),
            (self.c, self.d, 1),
            (self.a, self.f, 1),
            (self.d, self.e, 5),
        ]

    def assertEqualEdgeList(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for edge in actual:
            self.assertTrue(edge in expected)

    def test_edges(self):
        mst_edges = rustworkx.minimum_spanning_edges(self.graph, weight_fn=lambda x: x)
        self.assertEqual(len(self.graph.nodes()) - 1, len(mst_edges))
        for edge in mst_edges:
            self.assertTrue(edge in self.expected_edges)

    def test_tree(self):
        mst_graph = rustworkx.minimum_spanning_tree(self.graph, weight_fn=lambda x: x)
        self.assertEqual(self.graph.nodes(), mst_graph.nodes())
        self.assertEqual(len(self.graph.nodes()) - 1, len(mst_graph.edge_list()))
        self.assertEqualEdgeList(self.expected_edges, mst_graph.weighted_edge_list())

    def test_forest(self):
        s = self.graph.add_node("S")
        t = self.graph.add_node("T")
        u = self.graph.add_node("U")
        self.graph.add_edges_from([(s, t, 10), (t, u, 9), (s, u, 8)])
        forest_expected_edges = self.expected_edges + [(s, u, 8), (t, u, 9)]

        msf_graph = rustworkx.minimum_spanning_tree(self.graph, weight_fn=lambda x: x)
        self.assertEqual(self.graph.nodes(), msf_graph.nodes())
        self.assertEqual(len(self.graph.nodes()) - 2, len(msf_graph.edge_list()))
        self.assertEqualEdgeList(forest_expected_edges, msf_graph.weighted_edge_list())

    def test_isolated(self):
        s = self.graph.add_node("S")

        msf_graph = rustworkx.minimum_spanning_tree(self.graph, weight_fn=lambda x: x)
        self.assertEqual("S", msf_graph.nodes()[s])
        self.assertEqual(self.graph.nodes(), msf_graph.nodes())
        self.assertEqual(len(self.graph.nodes()) - 2, len(msf_graph.edge_list()))
        self.assertEqualEdgeList(self.expected_edges, msf_graph.weighted_edge_list())

    def test_multigraph(self):
        mutligraph = rustworkx.PyGraph(multigraph=True)
        mutligraph.extend_from_weighted_edge_list(
            [(0, 1, 1), (0, 2, 3), (1, 2, 2), (0, 0, -10), (1, 2, 1)]
        )

        mst_graph = rustworkx.minimum_spanning_tree(mutligraph, weight_fn=lambda x: x)
        self.assertEqualEdgeList([(0, 1, 1), (1, 2, 1)], mst_graph.weighted_edge_list())

    def test_default_weight(self):
        weightless_graph = rustworkx.PyGraph()
        weightless_graph.extend_from_edge_list(
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8)]
        )  # MST of the graph is itself

        mst_graph_default_weight = rustworkx.minimum_spanning_tree(weightless_graph)
        mst_graph_weight_2 = rustworkx.minimum_spanning_tree(weightless_graph, default_weight=2.0)

        self.assertTrue(
            rustworkx.is_isomorphic(
                weightless_graph,
                mst_graph_default_weight,
            )
        )
        self.assertTrue(
            rustworkx.is_isomorphic(
                weightless_graph,
                mst_graph_weight_2,
            )
        )

    def test_nan_weight(self):
        invalid_graph = rustworkx.PyGraph()
        invalid_graph.extend_from_weighted_edge_list([(0, 1, 0.5), (0, 2, float("nan"))])

        with self.assertRaises(ValueError):
            rustworkx.minimum_spanning_tree(invalid_graph, lambda x: x)
