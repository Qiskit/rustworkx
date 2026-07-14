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


class TestNdGridGraph(unittest.TestCase):
    def test_1d_grid(self):
        graph = rustworkx.generators.nd_grid_graph([5])
        self.assertEqual(len(graph), 5)
        self.assertEqual(len(graph.edges()), 4)

    def test_2d_grid(self):
        graph = rustworkx.generators.nd_grid_graph([3, 4])
        self.assertEqual(len(graph), 12)
        self.assertEqual(len(graph.edges()), 17)

    def test_3d_cube(self):
        graph = rustworkx.generators.nd_grid_graph([2, 2, 2])
        self.assertEqual(len(graph), 8)
        self.assertEqual(len(graph.edges()), 12)

    def test_3d_larger(self):
        graph = rustworkx.generators.nd_grid_graph([2, 3, 4])
        self.assertEqual(len(graph), 24)
        self.assertEqual(len(graph.edges()), 46)

    def test_4d(self):
        graph = rustworkx.generators.nd_grid_graph([2, 2, 2, 2])
        self.assertEqual(len(graph), 16)
        self.assertEqual(len(graph.edges()), 32)

    def test_with_positions(self):
        graph = rustworkx.generators.nd_grid_graph([2, 3], with_positions=True)
        nodes = graph.nodes()
        self.assertEqual(nodes[0], [0, 0])
        self.assertEqual(nodes[1], [1, 0])
        self.assertEqual(nodes[3], [1, 1])

    def test_periodic(self):
        # 1D periodic is a cycle
        graph = rustworkx.generators.nd_grid_graph([5], periodic=True)
        self.assertEqual(len(graph.edges()), 5)

    def test_torus(self):
        graph = rustworkx.generators.nd_grid_graph([3, 3], periodic=True)
        self.assertEqual(len(graph.edges()), 18)

    def test_empty_dim(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.nd_grid_graph([])

    def test_zero_dim(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.nd_grid_graph([2, 0, 3])


class TestDirectedNdGridGraph(unittest.TestCase):
    def test_basic(self):
        graph = rustworkx.generators.directed_nd_grid_graph([3, 3])
        self.assertEqual(len(graph), 9)
        self.assertEqual(len(graph.edges()), 12)

    def test_cube(self):
        graph = rustworkx.generators.directed_nd_grid_graph([2, 2, 2])
        self.assertEqual(len(graph), 8)
        self.assertEqual(len(graph.edges()), 12)

    def test_bidirectional(self):
        graph = rustworkx.generators.directed_nd_grid_graph([2, 2, 2], bidirectional=True)
        self.assertEqual(len(graph.edges()), 24)

    def test_with_positions(self):
        graph = rustworkx.generators.directed_nd_grid_graph([2, 2], with_positions=True)
        self.assertEqual(graph.nodes()[0], [0, 0])
        self.assertEqual(graph.nodes()[3], [1, 1])

    def test_periodic(self):
        graph = rustworkx.generators.directed_nd_grid_graph([3, 3], periodic=True)
        self.assertEqual(len(graph.edges()), 18)

    def test_edge_direction(self):
        graph = rustworkx.generators.directed_nd_grid_graph([2, 2])
        out_edges = graph.out_edges(0)
        out_targets = [e[1] for e in out_edges]
        self.assertIn(1, out_targets)
        self.assertIn(2, out_targets)
        self.assertEqual(len(graph.in_edges(0)), 0)
