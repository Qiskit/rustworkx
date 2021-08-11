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


class TestAvgShortestPath(unittest.TestCase):
    def test_simple_example(self):
        edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)]
        graph = retworkx.PyDiGraph()
        graph.extend_from_edge_list(edge_list)
        res = retworkx.digraph_unweighted_average_shortest_path_length(graph)
        self.assertEqual(1.0714285714285714, res)

    def test_cycle_graph(self):
        graph = retworkx.generators.directed_cycle_graph(7)
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(3.5, res, delta=1e-7)

    def test_path_graph(self):
        graph = retworkx.generators.directed_path_graph(5)
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(1.0, res, delta=1e-7)

    def test_parallel_grid(self):
        graph = retworkx.generators.directed_grid_graph(30, 11)
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(3.674772036474164, res, delta=1e-7)

    def test_empty(self):
        graph = retworkx.PyDiGraph()
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertTrue(math.isnan(res), "Output is not NaN")

    def test_single_node(self):
        graph = retworkx.PyDiGraph()
        graph.add_node(0)
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertEqual(0.0, res)

    def test_single_node_self_edge(self):
        graph = retworkx.PyDiGraph()
        node = graph.add_node(0)
        graph.add_edge(node, node, 0)
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertEqual(0.0, res)

    def test_disconnected_graph(self):
        graph = retworkx.PyDiGraph()
        node = graph.add_nodes_from(list(range(32)))
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertEqual(math.inf, res)

    def test_partially_connected_graph(self):
        graph = retworkx.generators.directed_cycle_graph(32)
        graph.add_nodes_from(list(range(32)))
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertEqual(math.inf, res)
