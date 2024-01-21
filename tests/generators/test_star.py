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


class TestStarGraph(unittest.TestCase):
    def test_directed_star_graph(self):
        graph = rustworkx.generators.directed_star_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 19)
        expected_edges = [(0, i, None) for i in range(1, 20)]
        self.assertEqual(sorted(graph.out_edges(0)), sorted(expected_edges))

    def test_star_directed_graph_inward(self):
        graph = rustworkx.generators.directed_star_graph(20, inward=True)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 19)
        expected_edges = [(i, 0, None) for i in range(1, 20)]
        self.assertEqual(sorted(graph.in_edges(0)), sorted(expected_edges))

    def test_directed_star_graph_weights(self):
        graph = rustworkx.generators.directed_star_graph(weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 19)
        expected_edges = sorted([(0, i, None) for i in range(1, 20)])
        self.assertEqual(sorted(graph.out_edges(0)), expected_edges)

    def test_directed_star_graph_bidirectional(self):
        graph = rustworkx.generators.directed_star_graph(20, bidirectional=True)
        outw = []
        inw = []
        for i in range(1, 20):
            outw.append((0, i, None))
            inw.append((i, 0, None))
            self.assertEqual(graph.out_edges(i), [(i, 0, None)])
            self.assertEqual(graph.in_edges(i), [(0, i, None)])
        self.assertEqual(graph.out_edges(0), outw[::-1])
        self.assertEqual(graph.in_edges(0), inw[::-1])

    def test_directed_star_graph_bidirectional_inward(self):
        graph = rustworkx.generators.directed_star_graph(20, bidirectional=True, inward=True)
        outw = []
        inw = []
        for i in range(1, 20):
            outw.append((0, i, None))
            inw.append((i, 0, None))
            self.assertEqual(graph.out_edges(i), [(i, 0, None)])
            self.assertEqual(graph.in_edges(i), [(0, i, None)])
        self.assertEqual(graph.out_edges(0), outw[::-1])
        self.assertEqual(graph.in_edges(0), inw[::-1])
        graph = rustworkx.generators.directed_star_graph(20, bidirectional=True, inward=False)
        outw = []
        inw = []
        for i in range(1, 20):
            outw.append((0, i, None))
            inw.append((i, 0, None))
            self.assertEqual(graph.out_edges(i), [(i, 0, None)])
            self.assertEqual(graph.in_edges(i), [(0, i, None)])
        self.assertEqual(graph.out_edges(0), outw[::-1])
        self.assertEqual(graph.in_edges(0), inw[::-1])

    def test_star_directed_graph_weights_inward(self):
        graph = rustworkx.generators.directed_star_graph(weights=list(range(20)), inward=True)
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 19)
        expected_edges = [(i, 0, None) for i in range(1, 20)]
        self.assertEqual(sorted(graph.in_edges(0)), sorted(expected_edges))

    def test_star_directed_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.directed_star_graph()

    def test_star_graph(self):
        graph = rustworkx.generators.star_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 19)

    def test_star_graph_weights(self):
        graph = rustworkx.generators.star_graph(weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 19)

    def test_star_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.star_graph()

    def test_zero_length_star_graph(self):
        graph = rustworkx.generators.star_graph(0)
        self.assertEqual(0, len(graph))

    def test_zero_length_directed_star_graph(self):
        graph = rustworkx.generators.directed_star_graph(0)
        self.assertEqual(0, len(graph))
