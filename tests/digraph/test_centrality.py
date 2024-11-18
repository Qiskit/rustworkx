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

    def test_closeness_centrality(self):
        closeness = rustworkx.digraph_closeness_centrality(self.graph)
        expected = {0: 0.0, 1: 1.0 / 3.0, 2: 4.0 / 9.0, 4: 0.5}
        self.assertEqual(expected, closeness)

    def test_closeness_centrality_wf_improved(self):
        closeness = rustworkx.digraph_closeness_centrality(self.graph, wf_improved=False)
        expected = {0: 0.0, 1: 1.0, 2: 2.0 / 3.0, 4: 0.5}
        self.assertEqual(expected, closeness)


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


class TestKatzCentrality(unittest.TestCase):
    def test_complete_graph(self):
        graph = rustworkx.generators.directed_complete_graph(5)
        centrality = rustworkx.digraph_katz_centrality(graph)
        expected_value = math.sqrt(1.0 / 5.0)
        for value in centrality.values():
            self.assertAlmostEqual(value, expected_value, delta=1e-4)

    def test_no_convergence(self):
        graph = rustworkx.generators.directed_complete_graph(5)
        with self.assertRaises(rustworkx.FailedToConverge):
            rustworkx.katz_centrality(graph, max_iter=0)

    def test_beta_scalar(self):
        rx_graph = rustworkx.generators.directed_grid_graph(5, 2)
        beta = 0.3

        rx_centrality = rustworkx.katz_centrality(rx_graph, alpha=0.25, beta=beta)

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(rx_graph.edge_list())
        nx_centrality = nx.katz_centrality(nx_graph, alpha=0.25, beta=beta)

        for key in rx_centrality.keys():
            self.assertAlmostEqual(rx_centrality[key], nx_centrality[key], delta=1e-4)

    def test_beta_dictionary(self):
        rx_graph = rustworkx.generators.directed_grid_graph(5, 2)
        beta = {i: 0.1 * i**2 for i in range(10)}

        rx_centrality = rustworkx.katz_centrality(rx_graph, alpha=0.25, beta=beta)

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(rx_graph.edge_list())
        nx_centrality = nx.katz_centrality(nx_graph, alpha=0.25, beta=beta)

        for key in rx_centrality.keys():
            self.assertAlmostEqual(rx_centrality[key], nx_centrality[key], delta=1e-4)

    def test_beta_incomplete(self):
        graph = rustworkx.generators.directed_grid_graph(5, 2)
        with self.assertRaises(ValueError):
            rustworkx.katz_centrality(graph, beta={0: 0.25})


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


class TestDiGraphDegreeCentrality(unittest.TestCase):
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
            (self.a, self.c, 1),  # Additional edge
        ]
        self.graph.add_edges_from(edge_list)

    def test_degree_centrality(self):
        centrality = rustworkx.degree_centrality(self.graph)
        expected = {
            0: 2 / 3,  # 2 total edges / 3
            1: 2 / 3,  # 2 total edges / 3
            2: 1.0,  # 3 total edges / 3
            3: 1 / 3,  # 1 total edge / 3
        }
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_in_degree_centrality(self):
        centrality = rustworkx.in_degree_centrality(self.graph)
        expected = {
            0: 0.0,  # 0 incoming edges
            1: 1 / 3,  # 1 incoming edge
            2: 2 / 3,  # 2 incoming edges
            3: 1 / 3,  # 1 incoming edge
        }
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_out_degree_centrality(self):
        centrality = rustworkx.out_degree_centrality(self.graph)
        expected = {
            0: 2 / 3,  # 2 outgoing edges
            1: 1 / 3,  # 1 outgoing edge
            2: 1 / 3,  # 1 outgoing edge
            3: 0.0,  # 0 outgoing edges
        }
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_degree_centrality_complete_digraph(self):
        graph = rustworkx.generators.directed_complete_graph(5)
        centrality = rustworkx.degree_centrality(graph)
        expected = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_degree_centrality_directed_path(self):
        graph = rustworkx.generators.directed_path_graph(5)
        centrality = rustworkx.degree_centrality(graph)
        expected = {
            0: 1 / 4,  # 1 total edge (out only) / 4
            1: 2 / 4,  # 2 total edges (1 in + 1 out) / 4
            2: 2 / 4,  # 2 total edges (1 in + 1 out) / 4
            3: 2 / 4,  # 2 total edges (1 in + 1 out) / 4
            4: 1 / 4,  # 1 total edge (in only) / 4
        }
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_in_degree_centrality_directed_path(self):
        graph = rustworkx.generators.directed_path_graph(5)
        centrality = rustworkx.in_degree_centrality(graph)
        expected = {
            0: 0.0,  # 0 incoming edges
            1: 1 / 4,  # 1 incoming edge
            2: 1 / 4,  # 1 incoming edge
            3: 1 / 4,  # 1 incoming edge
            4: 1 / 4,  # 1 incoming edge
        }
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])

    def test_out_degree_centrality_directed_path(self):
        graph = rustworkx.generators.directed_path_graph(5)
        centrality = rustworkx.out_degree_centrality(graph)
        expected = {
            0: 1 / 4,  # 1 outgoing edge
            1: 1 / 4,  # 1 outgoing edge
            2: 1 / 4,  # 1 outgoing edge
            3: 1 / 4,  # 1 outgoing edge
            4: 0.0,  # 0 outgoing edges
        }
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k])
