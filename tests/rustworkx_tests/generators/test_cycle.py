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


class TestCycleGraph(unittest.TestCase):
    def test_directed_cycle_graph(self):
        graph = rustworkx.generators.directed_cycle_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 20)
        for i in range(19):
            self.assertEqual(graph.out_edges(i), [(i, i + 1, None)])
        self.assertEqual(graph.out_edges(19), [(19, 0, None)])

    def test_directed_cycle_graph_weights(self):
        graph = rustworkx.generators.directed_cycle_graph(weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 20)
        for i in range(19):
            self.assertEqual(graph.out_edges(i), [(i, i + 1, None)])
        self.assertEqual(graph.out_edges(19), [(19, 0, None)])

    def test_directed_cycle_graph_bidirectional(self):
        graph = rustworkx.generators.directed_cycle_graph(20, bidirectional=True)
        self.assertEqual(graph.out_edges(0), [(0, 19, None), (0, 1, None)])
        self.assertEqual(graph.in_edges(0), [(19, 0, None), (1, 0, None)])
        for i in range(1, 19):
            self.assertEqual(graph.out_edges(i), [(i, i + 1, None), (i, i - 1, None)])
            self.assertEqual(graph.in_edges(i), [(i + 1, i, None), (i - 1, i, None)])
        self.assertEqual(graph.out_edges(19), [(19, 0, None), (19, 18, None)])
        self.assertEqual(graph.in_edges(19), [(0, 19, None), (18, 19, None)])

    def test_cycle_directed_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.directed_cycle_graph()

    def test_cycle_graph(self):
        graph = rustworkx.generators.cycle_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 20)

    def test_cycle_graph_weights(self):
        graph = rustworkx.generators.cycle_graph(weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 20)

    def test_cycle_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.cycle_graph()

    def test_zero_length_cycle_graph(self):
        graph = rustworkx.generators.cycle_graph(0)
        self.assertEqual(0, len(graph))

    def test_zero_length_directed_cycle_graph(self):
        graph = rustworkx.generators.directed_cycle_graph(0)
        self.assertEqual(0, len(graph))
