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
import networkx as nx


class TestCentralityGraph(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
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
        betweenness = rustworkx.graph_betweenness_centrality(self.graph)
        expected = {
            0: 0.0,
            1: 0.6666666666666666,
            2: 0.6666666666666666,
            3: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints(self):
        betweenness = rustworkx.graph_betweenness_centrality(self.graph, endpoints=True)
        expected = {
            0: 0.5,
            1: 0.8333333333333333,
            2: 0.8333333333333333,
            3: 0.5,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_unnormalized(self):
        betweenness = rustworkx.graph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 3: 0.0}
        self.assertEqual(expected, betweenness)


class TestCentralityGraphDeletedNode(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
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
        betweenness = rustworkx.graph_betweenness_centrality(self.graph)
        expected = {
            0: 0.0,
            1: 0.6666666666666666,
            2: 0.6666666666666666,
            4: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints(self):
        betweenness = rustworkx.graph_betweenness_centrality(self.graph, endpoints=True)
        expected = {
            0: 0.5,
            1: 0.8333333333333333,
            2: 0.8333333333333333,
            4: 0.5,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_unnormalized(self):
        betweenness = rustworkx.graph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 4: 0.0}
        self.assertEqual(expected, betweenness)

    def test_closeness_centrality(self):
        closeness = rustworkx.graph_closeness_centrality(self.graph)
        expected = {0: 0.5, 1: 0.75, 2: 0.75, 4: 0.5}
        self.assertEqual(expected, closeness)

    def test_closeness_centrality_parallel(self):
        closeness = rustworkx.graph_closeness_centrality(
            self.graph, parallel_threshold=1
        )  # force parallelism
        expected = {0: 0.5, 1: 0.75, 2: 0.75, 4: 0.5}
        self.assertEqual(expected, closeness)

    def test_closeness_centrality_wf_improved(self):
        closeness = rustworkx.graph_closeness_centrality(self.graph, wf_improved=False)
        expected = {0: 0.5, 1: 0.75, 2: 0.75, 4: 0.5}
        self.assertEqual(expected, closeness)


class TestEigenvectorCentrality(unittest.TestCase):
    def test_complete_graph(self):
        graph = rustworkx.generators.mesh_graph(5)
        centrality = rustworkx.eigenvector_centrality(graph)
        expected_value = math.sqrt(1.0 / 5.0)
        for value in centrality.values():
            self.assertAlmostEqual(value, expected_value)

    def test_path_graph(self):
        graph = rustworkx.generators.path_graph(3)
        centrality = rustworkx.eigenvector_centrality(graph)
        expected = [0.5, 0.7071, 0.5]
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k], 4)

    def test_no_convergence(self):
        graph = rustworkx.PyGraph()
        with self.assertRaises(rustworkx.FailedToConverge):
            rustworkx.eigenvector_centrality(graph, max_iter=0)


class TestKatzCentrality(unittest.TestCase):
    def test_complete_graph(self):
        graph = rustworkx.generators.complete_graph(5)
        centrality = rustworkx.graph_katz_centrality(graph)
        expected_value = math.sqrt(1.0 / 5.0)
        for value in centrality.values():
            self.assertAlmostEqual(value, expected_value, delta=1e-4)

    def test_no_convergence(self):
        graph = rustworkx.generators.complete_graph(5)
        with self.assertRaises(rustworkx.FailedToConverge):
            rustworkx.katz_centrality(graph, max_iter=0)

    def test_beta_scalar(self):
        graph = rustworkx.generators.generalized_petersen_graph(5, 2)
        expected_value = 0.31622776601683794

        centrality = rustworkx.katz_centrality(graph, alpha=0.1, beta=0.1, tol=1e-8)

        for value in centrality.values():
            self.assertAlmostEqual(value, expected_value, delta=1e-4)

    def test_beta_dictionary(self):
        rx_graph = rustworkx.generators.generalized_petersen_graph(5, 2)
        beta = {i: 0.1 * i**2 for i in range(10)}

        rx_centrality = rustworkx.katz_centrality(rx_graph, alpha=0.25, beta=beta)

        nx_graph = nx.Graph()
        nx_graph.add_edges_from(rx_graph.edge_list())
        nx_centrality = nx.katz_centrality(nx_graph, alpha=0.25, beta=beta)

        for key in rx_centrality.keys():
            self.assertAlmostEqual(rx_centrality[key], nx_centrality[key], delta=1e-4)

    def test_beta_incomplete(self):
        graph = rustworkx.generators.generalized_petersen_graph(5, 2)
        with self.assertRaises(ValueError):
            rustworkx.katz_centrality(graph, beta={0: 0.25})


class TestEdgeBetweennessCentrality(unittest.TestCase):
    def test_complete_graph(self):
        graph = rustworkx.generators.mesh_graph(5)
        centrality = rustworkx.edge_betweenness_centrality(graph)
        for value in centrality.values():
            self.assertAlmostEqual(value, 0.1)

    def test_path_graph(self):
        graph = rustworkx.generators.path_graph(5)
        centrality = rustworkx.edge_betweenness_centrality(graph)
        expected = {0: 0.4, 1: 0.6, 2: 0.6, 3: 0.4}
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_cycle_graph(self):
        graph = rustworkx.generators.cycle_graph(5)
        centrality = rustworkx.edge_betweenness_centrality(graph)
        for k, v in centrality.items():
            self.assertAlmostEqual(v, 0.3)

    def test_tree_unnormalized(self):
        graph = rustworkx.generators.full_rary_tree(2, 7)
        centrality = rustworkx.edge_betweenness_centrality(graph, normalized=False)
        expected = {0: 12.0, 1: 12.0, 2: 6.0, 3: 6.0, 4: 6.0, 5: 6.0}
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_path_graph_unnormalized(self):
        graph = rustworkx.generators.path_graph(5)
        centrality = rustworkx.edge_betweenness_centrality(graph, normalized=False)
        expected = {0: 4.0, 1: 6.0, 2: 6.0, 3: 4.0}
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_custom_graph_unnormalized(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(range(10))
        graph.add_edges_from(
            [
                (0, 1, 1),
                (0, 2, 1),
                (0, 3, 1),
                (0, 4, 1),
                (3, 5, 1),
                (4, 6, 1),
                (5, 7, 1),
                (6, 8, 1),
                (7, 8, 1),
                (8, 9, 1),
            ]
        )
        centrality = rustworkx.edge_betweenness_centrality(graph, normalized=False)
        expected = {0: 9, 1: 9, 2: 12, 3: 15, 4: 11, 5: 14, 6: 10, 7: 13, 8: 9, 9: 9}
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])


class TestGraphDegreeCentrality(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
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

    def test_degree_centrality(self):
        centrality = rustworkx.degree_centrality(self.graph)
        expected = {
            0: 1 / 3,  # Node A has 1 edge, normalized by (n-1) = 3
            1: 2 / 3,  # Node B has 2 edges
            2: 2 / 3,  # Node C has 2 edges
            3: 1 / 3,  # Node D has 1 edge
        }
        self.assertEqual(expected, centrality)

    def test_degree_centrality_complete_graph(self):
        graph = rustworkx.generators.complete_graph(5)
        centrality = rustworkx.degree_centrality(graph)
        expected = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        self.assertEqual(expected, centrality)

    def test_degree_centrality_star_graph(self):
        graph = rustworkx.generators.star_graph(5)
        centrality = rustworkx.degree_centrality(graph)
        expected = {0: 1.0, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
        self.assertEqual(expected, centrality)

    def test_degree_centrality_empty_graph(self):
        graph = rustworkx.PyGraph()
        centrality = rustworkx.degree_centrality(graph)
        expected = {}
        self.assertEqual(expected, centrality)

    def test_degree_centrality_multigraph(self):
        graph = rustworkx.PyGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")
        edge_list = [
            (a, b, 1),  # First edge between A-B
            (a, b, 2),  # Second edge between A-B (parallel edge)
            (b, c, 1),  # Edge between B-C
        ]
        graph.add_edges_from(edge_list)

        centrality = rustworkx.degree_centrality(graph)
        expected = {
            0: 1.0,  # Node A has 2 edges (counting parallel edges), normalized by (n-1) = 2
            1: 1.5,  # Node B has 3 edges total (2 to A, 1 to C)
            2: 0.5,  # Node C has 1 edge
        }
        self.assertEqual(expected, dict(centrality))
