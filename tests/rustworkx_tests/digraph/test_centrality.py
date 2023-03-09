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

import math
import unittest

import rustworkx


class TestCentralityDiGraph(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyDiGraph()
        self.a = self.graph.add_node("A")
        self.b = self.graph.add_node("B")
        self.c = self.graph.add_node("C")
        self.d = self.graph.add_node("D")
        edge_list = [
            (self.a, self.b, 1),
            (self.b, self.c, 1),
            (self.c, self.d, 1),
        ]
        self.graph.add_edges_from(edge_list)

    def test_betweenness_centrality(self):
        betweenness = rustworkx.digraph_betweenness_centrality(self.graph)
        expected = {
            0: 0.0,
            1: 0.3333333333333333,
            2: 0.3333333333333333,
            3: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints(self):
        betweenness = rustworkx.digraph_betweenness_centrality(self.graph, endpoints=True)
        expected = {
            0: 0.25,
            1: 0.41666666666666663,
            2: 0.41666666666666663,
            3: 0.25,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_unnormalized(self):
        betweenness = rustworkx.digraph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 3: 0.0}
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_parallel(self):
        betweenness = rustworkx.digraph_betweenness_centrality(self.graph, parallel_threshold=1)
        expected = {
            0: 0.0,
            1: 0.3333333333333333,
            2: 0.3333333333333333,
            3: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints_parallel(self):
        betweenness = rustworkx.digraph_betweenness_centrality(
            self.graph, endpoints=True, parallel_threshold=1
        )
        expected = {
            0: 0.25,
            1: 0.41666666666666663,
            2: 0.41666666666666663,
            3: 0.25,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_unnormalized_parallel(self):
        betweenness = rustworkx.digraph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False, parallel_threshold=1
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 3: 0.0}
        self.assertEqual(expected, betweenness)


class TestCentralityDiGraphDeletedNode(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyDiGraph()
        self.a = self.graph.add_node("A")
        self.b = self.graph.add_node("B")
        self.c = self.graph.add_node("C")
        c0 = self.graph.add_node("C0")
        self.d = self.graph.add_node("D")
        edge_list = [
            (self.a, self.b, 1),
            (self.b, self.c, 1),
            (self.c, self.d, 1),
        ]
        self.graph.add_edges_from(edge_list)
        self.graph.remove_node(c0)

    def test_betweenness_centrality(self):
        betweenness = rustworkx.digraph_betweenness_centrality(self.graph)
        expected = {
            0: 0.0,
            1: 0.3333333333333333,
            2: 0.3333333333333333,
            4: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints(self):
        betweenness = rustworkx.digraph_betweenness_centrality(self.graph, endpoints=True)
        expected = {
            0: 0.25,
            1: 0.41666666666666663,
            2: 0.41666666666666663,
            4: 0.25,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_unnormalized(self):
        betweenness = rustworkx.digraph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 4: 0.0}
        self.assertEqual(expected, betweenness)


class TestEigenvectorCentrality(unittest.TestCase):
    def test_complete_graph(self):
        graph = rustworkx.generators.directed_mesh_graph(5)
        centrality = rustworkx.eigenvector_centrality(graph)
        expected_value = math.sqrt(1.0 / 5.0)
        for value in centrality.values():
            self.assertAlmostEqual(value, expected_value)

    def test_path_graph(self):
        graph = rustworkx.generators.directed_path_graph(3, bidirectional=True)
        centrality = rustworkx.eigenvector_centrality(graph)
        expected = [0.5, 0.7071, 0.5]
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k], 4)

    def test_no_convergence(self):
        graph = rustworkx.PyDiGraph()
        with self.assertRaises(rustworkx.FailedToConverge):
            rustworkx.eigenvector_centrality(graph, max_iter=0)


class TestEdgeBetweennessCentrality(unittest.TestCase):
    def test_complete_graph(self):
        graph = rustworkx.generators.directed_mesh_graph(5)
        centrality = rustworkx.edge_betweenness_centrality(graph)
        for value in centrality.values():
            self.assertAlmostEqual(value, 0.05)

    def test_path_graph(self):
        graph = rustworkx.generators.directed_path_graph(5)
        centrality = rustworkx.edge_betweenness_centrality(graph)
        expected = {0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2}
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_cycle_graph(self):
        graph = rustworkx.generators.directed_cycle_graph(5)
        centrality = rustworkx.edge_betweenness_centrality(graph)
        for k, v in centrality.items():
            self.assertAlmostEqual(v, 0.5)

    def test_tree_unnormalized(self):
        graph = rustworkx.generators.full_rary_tree(2, 7).to_directed()
        centrality = rustworkx.edge_betweenness_centrality(graph, normalized=False)
        expected = {0: 12, 1: 12, 2: 12, 3: 12, 4: 6, 5: 6, 6: 6, 7: 6, 8: 6, 9: 6, 10: 6, 11: 6}
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_path_graph_unnormalized(self):
        graph = rustworkx.generators.directed_path_graph(5)
        centrality = rustworkx.edge_betweenness_centrality(graph, normalized=False)
        expected = {0: 4.0, 1: 6.0, 2: 6.0, 3: 4.0}
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])
