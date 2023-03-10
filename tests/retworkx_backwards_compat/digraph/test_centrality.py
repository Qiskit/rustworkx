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

import retworkx


class TestCentralityDiGraph(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyDiGraph()
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
        betweenness = retworkx.digraph_betweenness_centrality(self.graph)
        expected = {
            0: 0.0,
            1: 0.3333333333333333,
            2: 0.3333333333333333,
            3: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints(self):
        betweenness = retworkx.digraph_betweenness_centrality(self.graph, endpoints=True)
        expected = {
            0: 0.25,
            1: 0.41666666666666663,
            2: 0.41666666666666663,
            3: 0.25,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_unnormalized(self):
        betweenness = retworkx.digraph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 3: 0.0}
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_parallel(self):
        betweenness = retworkx.digraph_betweenness_centrality(self.graph, parallel_threshold=1)
        expected = {
            0: 0.0,
            1: 0.3333333333333333,
            2: 0.3333333333333333,
            3: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints_parallel(self):
        betweenness = retworkx.digraph_betweenness_centrality(
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
        betweenness = retworkx.digraph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False, parallel_threshold=1
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 3: 0.0}
        self.assertEqual(expected, betweenness)

    def test_closeness_centrality(self):
        closeness = retworkx.digraph_closeness_centrality(self.graph)
        expected = {0: 0.0, 1: 1.0 / 3.0, 2: 4.0 / 9.0, 3: 0.5}
        self.assertEqual(expected, closeness)

    def test_closeness_centrality_wf_improved(self):
        closeness = retworkx.digraph_closeness_centrality(self.graph, wf_improved=False)
        expected = {0: 0.0, 1: 1.0, 2: 2.0 / 3.0, 3: 0.5}
        self.assertEqual(expected, closeness)


class TestCentralityDiGraphDeletedNode(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyDiGraph()
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
        betweenness = retworkx.digraph_betweenness_centrality(self.graph)
        expected = {
            0: 0.0,
            1: 0.3333333333333333,
            2: 0.3333333333333333,
            4: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints(self):
        betweenness = retworkx.digraph_betweenness_centrality(self.graph, endpoints=True)
        expected = {
            0: 0.25,
            1: 0.41666666666666663,
            2: 0.41666666666666663,
            4: 0.25,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_unnormalized(self):
        betweenness = retworkx.digraph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 4: 0.0}
        self.assertEqual(expected, betweenness)


class TestEigenvectorCentrality(unittest.TestCase):
    def test_complete_graph(self):
        graph = retworkx.generators.directed_mesh_graph(5)
        centrality = retworkx.eigenvector_centrality(graph)
        expected_value = math.sqrt(1.0 / 5.0)
        for value in centrality.values():
            self.assertAlmostEqual(value, expected_value)

    def test_path_graph(self):
        graph = retworkx.generators.directed_path_graph(3, bidirectional=True)
        centrality = retworkx.eigenvector_centrality(graph)
        expected = [0.5, 0.7071, 0.5]
        for k, v in centrality.items():
            self.assertAlmostEqual(v, expected[k], 4)

    def test_no_convergence(self):
        graph = retworkx.PyDiGraph()
        with self.assertRaises(retworkx.FailedToConverge):
            retworkx.eigenvector_centrality(graph, max_iter=0)
