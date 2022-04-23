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

import retworkx


class TestCentralityGraph(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyGraph()
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
        betweenness = retworkx.graph_betweenness_centrality(self.graph)
        expected = {
            0: 0.0,
            1: 0.6666666666666666,
            2: 0.6666666666666666,
            3: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints(self):
        betweenness = retworkx.graph_betweenness_centrality(self.graph, endpoints=True)
        expected = {
            0: 0.5,
            1: 0.8333333333333333,
            2: 0.8333333333333333,
            3: 0.5,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_unnormalized(self):
        betweenness = retworkx.graph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 3: 0.0}
        self.assertEqual(expected, betweenness)

    def test_closeness_centrality(self):
        betweenness = retworkx.graph_closeness_centrality(self.graph)
        expected = {0: 0.5, 1: 0.75, 2: 0.75, 3: 0.5}
        self.assertEqual(expected, betweenness)

    def test_closeness_centrality_wf_improved(self):
        betweenness = retworkx.graph_closeness_centrality(self.graph, wf_improved=False)
        expected = {0: 0.5, 1: 0.75, 2: 0.75, 3: 0.5}
        self.assertEqual(expected, betweenness)


class TestCentralityGraphDeletedNode(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyGraph()
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
        betweenness = retworkx.graph_betweenness_centrality(self.graph)
        expected = {
            0: 0.0,
            1: 0.6666666666666666,
            2: 0.6666666666666666,
            4: 0.0,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_endpoints(self):
        betweenness = retworkx.graph_betweenness_centrality(self.graph, endpoints=True)
        expected = {
            0: 0.5,
            1: 0.8333333333333333,
            2: 0.8333333333333333,
            4: 0.5,
        }
        self.assertEqual(expected, betweenness)

    def test_betweenness_centrality_unnormalized(self):
        betweenness = retworkx.graph_betweenness_centrality(
            self.graph, endpoints=False, normalized=False
        )
        expected = {0: 0.0, 1: 2.0, 2: 2.0, 4: 0.0}
        self.assertEqual(expected, betweenness)

    def test_closeness_centrality(self):
        betweenness = retworkx.graph_closeness_centrality(self.graph)
        expected = {0: 0.5, 1: 0.75, 2: 0.75, 4: 0.5}
        self.assertEqual(expected, betweenness)

    def test_closeness_centrality_wf_improved(self):
        betweenness = retworkx.graph_closeness_centrality(self.graph, wf_improved=False)
        expected = {0: 0.5, 1: 0.75, 2: 0.75, 4: 0.5}
        self.assertEqual(expected, betweenness)
