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

    def test_closeness_weighted_with_default_weight(self):
        for parallel_threshold in [1, 200]:
            with self.subTest(parallel_threshold=parallel_threshold):
                closeness = rustworkx.closeness_centrality(self.graph, parallel_threshold=1)
                weighted_closeness = rustworkx.newman_weighted_closeness_centrality(
                    self.graph, default_weight=1.0, parallel_threshold=1
                )
                self.assertEqual(closeness, weighted_closeness)


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


class TestGroupDegreeCentralityGraph(unittest.TestCase):
    def test_path_graph(self):
        graph = rustworkx.generators.path_graph(5)
        result = rustworkx.graph_group_degree_centrality(graph, [0, 1])
        self.assertAlmostEqual(result, 1.0 / 3.0)

    def test_complete_graph(self):
        graph = rustworkx.generators.complete_graph(4)
        result = rustworkx.graph_group_degree_centrality(graph, [0])
        self.assertAlmostEqual(result, 1.0)

    def test_single_node_group(self):
        graph = rustworkx.generators.path_graph(3)
        result = rustworkx.graph_group_degree_centrality(graph, [1])
        self.assertAlmostEqual(result, 1.0)

    def test_dispatch(self):
        graph = rustworkx.generators.path_graph(5)
        result = rustworkx.group_degree_centrality(graph, [0, 1])
        self.assertAlmostEqual(result, 1.0 / 3.0)

    def test_invalid_node(self):
        graph = rustworkx.generators.path_graph(3)
        with self.assertRaises(ValueError):
            rustworkx.graph_group_degree_centrality(graph, [10])


class TestGroupClosenessCentralityGraph(unittest.TestCase):
    def test_path_graph(self):
        graph = rustworkx.generators.path_graph(5)
        result = rustworkx.graph_group_closeness_centrality(graph, [0, 1])
        self.assertAlmostEqual(result, 0.5)

    def test_star_center(self):
        graph = rustworkx.PyGraph()
        center = graph.add_node("center")
        for _ in range(4):
            leaf = graph.add_node("leaf")
            graph.add_edge(center, leaf, None)
        result = rustworkx.graph_group_closeness_centrality(graph, [center])
        self.assertAlmostEqual(result, 1.0)

    def test_dispatch(self):
        graph = rustworkx.generators.path_graph(5)
        result = rustworkx.group_closeness_centrality(graph, [0, 1])
        self.assertAlmostEqual(result, 0.5)

    def test_invalid_node(self):
        graph = rustworkx.generators.path_graph(3)
        with self.assertRaises(ValueError):
            rustworkx.graph_group_closeness_centrality(graph, [10])


class TestGroupBetweennessCentralityGraph(unittest.TestCase):
    def test_path_center(self):
        graph = rustworkx.generators.path_graph(5)
        result = rustworkx.graph_group_betweenness_centrality(graph, [2], normalized=False)
        self.assertAlmostEqual(result, 4.0)

    def test_path_center_normalized(self):
        graph = rustworkx.generators.path_graph(5)
        result = rustworkx.graph_group_betweenness_centrality(graph, [2], normalized=True)
        self.assertAlmostEqual(result, 2.0 / 3.0)

    def test_star_center(self):
        graph = rustworkx.PyGraph()
        center = graph.add_node("center")
        for _ in range(4):
            leaf = graph.add_node("leaf")
            graph.add_edge(center, leaf, None)
        result = rustworkx.graph_group_betweenness_centrality(
            graph, [center], normalized=False
        )
        self.assertAlmostEqual(result, 6.0)

    def test_empty_group(self):
        graph = rustworkx.generators.path_graph(3)
        result = rustworkx.graph_group_betweenness_centrality(graph, [], normalized=False)
        self.assertAlmostEqual(result, 0.0)

    def test_dispatch(self):
        graph = rustworkx.generators.path_graph(5)
        result = rustworkx.group_betweenness_centrality(graph, [2])
        self.assertAlmostEqual(result, 2.0 / 3.0)

    def test_invalid_node(self):
        graph = rustworkx.generators.path_graph(3)
        with self.assertRaises(ValueError):
            rustworkx.graph_group_betweenness_centrality(graph, [10])


class TestGroupCentralityNetworkXComparisonGraph(unittest.TestCase):
    """Cross-validate group centrality results against NetworkX."""

    def _build_graphs(self, nx_graph):
        rx_graph = rustworkx.PyGraph()
        node_map = {}
        for node in nx_graph.nodes():
            node_map[node] = rx_graph.add_node(node)
        for u, v in nx_graph.edges():
            rx_graph.add_edge(node_map[u], node_map[v], None)
        return rx_graph, node_map

    def test_degree_path_graph(self):
        g_nx = nx.path_graph(5)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{0}, {2}, {0, 1}, {1, 3}, {0, 2, 4}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_degree_centrality(g_nx, group_nodes)
            result = rustworkx.graph_group_degree_centrality(g_rx, rx_group)
            self.assertAlmostEqual(result, expected, places=10)

    def test_degree_complete_graph(self):
        g_nx = nx.complete_graph(6)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{0}, {0, 1}, {0, 2, 4}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_degree_centrality(g_nx, group_nodes)
            result = rustworkx.graph_group_degree_centrality(g_rx, rx_group)
            self.assertAlmostEqual(result, expected, places=10)

    def test_degree_cycle_graph(self):
        g_nx = nx.cycle_graph(8)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{0}, {0, 4}, {0, 2, 4, 6}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_degree_centrality(g_nx, group_nodes)
            result = rustworkx.graph_group_degree_centrality(g_rx, rx_group)
            self.assertAlmostEqual(result, expected, places=10)

    def test_closeness_path_graph(self):
        g_nx = nx.path_graph(5)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{0}, {2}, {0, 1}, {1, 3}, {0, 2, 4}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_closeness_centrality(g_nx, group_nodes)
            result = rustworkx.graph_group_closeness_centrality(g_rx, rx_group)
            self.assertAlmostEqual(result, expected, places=10)

    def test_closeness_complete_graph(self):
        g_nx = nx.complete_graph(6)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{0}, {0, 1}, {0, 2, 4}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_closeness_centrality(g_nx, group_nodes)
            result = rustworkx.graph_group_closeness_centrality(g_rx, rx_group)
            self.assertAlmostEqual(result, expected, places=10)

    def test_closeness_cycle_graph(self):
        g_nx = nx.cycle_graph(8)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{0}, {0, 4}, {0, 2, 4, 6}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_closeness_centrality(g_nx, group_nodes)
            result = rustworkx.graph_group_closeness_centrality(g_rx, rx_group)
            self.assertAlmostEqual(result, expected, places=10)

    def test_betweenness_path_graph(self):
        g_nx = nx.path_graph(5)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{2}, {1, 3}, {0, 4}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_betweenness_centrality(g_nx, [group_nodes])[0]
            result = rustworkx.graph_group_betweenness_centrality(
                g_rx, rx_group, normalized=True
            )
            self.assertAlmostEqual(result, expected, places=10)

    def test_betweenness_complete_graph(self):
        g_nx = nx.complete_graph(6)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{0}, {0, 1}, {0, 2, 4}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_betweenness_centrality(g_nx, [group_nodes])[0]
            result = rustworkx.graph_group_betweenness_centrality(
                g_rx, rx_group, normalized=True
            )
            self.assertAlmostEqual(result, expected, places=10)

    def test_betweenness_star_graph(self):
        g_nx = nx.star_graph(4)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{0}, {1}, {0, 1}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_betweenness_centrality(g_nx, [group_nodes])[0]
            result = rustworkx.graph_group_betweenness_centrality(
                g_rx, rx_group, normalized=True
            )
            self.assertAlmostEqual(result, expected, places=10)

    def test_betweenness_barbell_graph(self):
        g_nx = nx.barbell_graph(4, 1)
        g_rx, nmap = self._build_graphs(g_nx)
        for group_nodes in [{4}, {3, 4, 5}, {0, 8}]:
            rx_group = [nmap[n] for n in group_nodes]
            expected = nx.group_betweenness_centrality(g_nx, [group_nodes])[0]
            result = rustworkx.graph_group_betweenness_centrality(
                g_rx, rx_group, normalized=True
            )
            self.assertAlmostEqual(result, expected, places=10)
