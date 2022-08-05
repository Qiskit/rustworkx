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


class TestGridGraph(unittest.TestCase):
    def test_directed_grid_graph_dimensions(self):
        graph = rustworkx.generators.directed_grid_graph(4, 5)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 31)
        self.assertEqual(graph.out_edges(0), [(0, 1, None), (0, 5, None)])
        self.assertEqual(graph.out_edges(7), [(7, 8, None), (7, 12, None)])
        self.assertEqual(graph.out_edges(9), [(9, 14, None)])
        self.assertEqual(graph.out_edges(17), [(17, 18, None)])
        self.assertEqual(graph.out_edges(19), [])
        self.assertEqual(graph.in_edges(0), [])
        self.assertEqual(graph.in_edges(2), [(1, 2, None)])
        self.assertEqual(graph.in_edges(5), [(0, 5, None)])
        self.assertEqual(graph.in_edges(7), [(6, 7, None), (2, 7, None)])
        self.assertEqual(graph.in_edges(19), [(18, 19, None), (14, 19, None)])

    def test_directed_grid_graph_weights(self):
        graph = rustworkx.generators.directed_grid_graph(weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 19)
        for i in range(19):
            self.assertEqual(graph.out_edges(i), [(i, i + 1, None)])
        self.assertEqual(graph.out_edges(19), [])
        for i in range(1, 20):
            self.assertEqual(graph.in_edges(i), [(i - 1, i, None)])
        self.assertEqual(graph.in_edges(0), [])

    def test_directed_grid_graph_dimensions_weights(self):
        graph = rustworkx.generators.directed_grid_graph(4, 5, weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 31)
        self.assertEqual(graph.out_edges(0), [(0, 1, None), (0, 5, None)])
        self.assertEqual(graph.out_edges(7), [(7, 8, None), (7, 12, None)])
        self.assertEqual(graph.out_edges(9), [(9, 14, None)])
        self.assertEqual(graph.out_edges(17), [(17, 18, None)])
        self.assertEqual(graph.out_edges(19), [])
        self.assertEqual(graph.in_edges(0), [])
        self.assertEqual(graph.in_edges(2), [(1, 2, None)])
        self.assertEqual(graph.in_edges(5), [(0, 5, None)])
        self.assertEqual(graph.in_edges(7), [(6, 7, None), (2, 7, None)])
        self.assertEqual(graph.in_edges(19), [(18, 19, None), (14, 19, None)])

    def test_directed_grid_graph_more_dimensions_weights(self):
        graph = rustworkx.generators.directed_grid_graph(4, 5, weights=list(range(16)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(16)] + [None] * 4, graph.nodes())
        self.assertEqual(len(graph.edges()), 31)
        self.assertEqual(graph.out_edges(0), [(0, 1, None), (0, 5, None)])
        self.assertEqual(graph.out_edges(7), [(7, 8, None), (7, 12, None)])
        self.assertEqual(graph.out_edges(9), [(9, 14, None)])
        self.assertEqual(graph.out_edges(17), [(17, 18, None)])
        self.assertEqual(graph.out_edges(19), [])
        self.assertEqual(graph.in_edges(0), [])
        self.assertEqual(graph.in_edges(2), [(1, 2, None)])
        self.assertEqual(graph.in_edges(5), [(0, 5, None)])
        self.assertEqual(graph.in_edges(7), [(6, 7, None), (2, 7, None)])
        self.assertEqual(graph.in_edges(19), [(18, 19, None), (14, 19, None)])

    def test_directed_grid_graph_less_dimensions_weights(self):
        graph = rustworkx.generators.directed_grid_graph(4, 5, weights=list(range(24)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 31)
        self.assertEqual(graph.out_edges(0), [(0, 1, None), (0, 5, None)])
        self.assertEqual(graph.out_edges(7), [(7, 8, None), (7, 12, None)])
        self.assertEqual(graph.out_edges(9), [(9, 14, None)])
        self.assertEqual(graph.out_edges(17), [(17, 18, None)])
        self.assertEqual(graph.out_edges(19), [])
        self.assertEqual(graph.in_edges(0), [])
        self.assertEqual(graph.in_edges(2), [(1, 2, None)])
        self.assertEqual(graph.in_edges(5), [(0, 5, None)])
        self.assertEqual(graph.in_edges(7), [(6, 7, None), (2, 7, None)])
        self.assertEqual(graph.in_edges(19), [(18, 19, None), (14, 19, None)])

    def test_grid_directed_no_weights_or_dim(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.directed_grid_graph()
            rustworkx.generators.directed_grid_graph(rows=5, weights=[1] * 5)
            rustworkx.generators.directed_grid_graph(cols=5, weights=[1] * 5)

    def test_grid_graph_dimensions(self):
        graph = rustworkx.generators.grid_graph(4, 5)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 31)

    def test_grid_graph_weights(self):
        graph = rustworkx.generators.grid_graph(weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 19)

    def test_grid_graph_dimensions_weights(self):
        graph = rustworkx.generators.grid_graph(4, 5, weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 31)

        graph = rustworkx.generators.grid_graph(4, 5, weights=list(range(16)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(16)] + [None] * 4, graph.nodes())
        self.assertEqual(len(graph.edges()), 31)

        graph = rustworkx.generators.grid_graph(4, 5, weights=list(range(24)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 31)

    def test_grid_no_weights_or_dim(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.grid_graph()
            rustworkx.generators.grid_graph(rows=5, weights=[1] * 5)
            rustworkx.generators.grid_graph(cols=5, weights=[1] * 5)
