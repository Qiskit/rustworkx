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


class TestUnweightedAvgShortestPath(unittest.TestCase):
    def test_simple_example(self):
        edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)]
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list(edge_list)
        res = rustworkx.graph_unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(2.5714285714285716, res, delta=1e-7)

    def test_cycle_graph(self):
        graph = rustworkx.generators.cycle_graph(7)
        res = rustworkx.unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(2, res, delta=1e-7)

    def test_path_graph(self):
        graph = rustworkx.generators.path_graph(5)
        res = rustworkx.unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(2, res, delta=1e-7)

    def test_parallel_grid(self):
        graph = rustworkx.generators.grid_graph(30, 11)
        res = rustworkx.unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(13.666666666666666, res, delta=1e-7)

    def test_empty(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.unweighted_average_shortest_path_length(graph)
        self.assertTrue(math.isnan(res), "Output is not NaN")

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        res = rustworkx.unweighted_average_shortest_path_length(graph)
        self.assertTrue(math.isnan(res), "Output is not NaN")

    def test_single_node_self_edge(self):
        graph = rustworkx.PyGraph()
        node = graph.add_node(0)
        graph.add_edge(node, node, 0)
        res = rustworkx.unweighted_average_shortest_path_length(graph)
        self.assertTrue(math.isnan(res), "Output is not NaN")

    def test_disconnected_graph(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(32)))
        with self.subTest(disconnected=False):
            res = rustworkx.unweighted_average_shortest_path_length(graph)
            self.assertTrue(math.isinf(res), "Output is not infinity")

        with self.subTest(disconnected=True):
            res = rustworkx.unweighted_average_shortest_path_length(graph, disconnected=True)
            self.assertTrue(math.isnan(res), "Output is not NaN")

    def test_partially_connected_graph(self):
        graph = rustworkx.generators.cycle_graph(32)
        graph.add_nodes_from(list(range(32)))
        with self.subTest(disconnected=False):
            res = rustworkx.unweighted_average_shortest_path_length(graph)
            self.assertTrue(math.isinf(res), "Output is not infinity")

        with self.subTest(disconnected=True):
            s = 8192
            den = 992  # n*(n-1), n=32 (only connected pairs considered)
            res = rustworkx.unweighted_average_shortest_path_length(graph, disconnected=True)
            self.assertAlmostEqual(s / den, res, delta=1e-7)

    def test_connected_cycle_graph(self):
        graph = rustworkx.generators.cycle_graph(32)
        res = rustworkx.unweighted_average_shortest_path_length(graph)
        s = 8192
        den = 992  # n*(n-1)
        self.assertAlmostEqual(s / den, res, delta=1e-7)
