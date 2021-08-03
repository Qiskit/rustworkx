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


class TestUnweightedAvgShortestPath(unittest.TestCase):
    def test_simple_example(self):
        edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)]
        graph = retworkx.PyGraph()
        graph.extend_from_edge_list(edge_list)
        res = retworkx.graph_unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(2.5714285714285716, res, delta=1e-7)

    def test_cycle_graph(self):
        graph = retworkx.generators.cycle_graph(7)
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(2, res, delta=1e-7)

    def test_path_graph(self):
        graph = retworkx.generators.path_graph(5)
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(2, res, delta=1e-7)

    def test_parallel_grid(self):
        graph = retworkx.generators.grid_graph(30, 11)
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertAlmostEqual(13.666666666666666, res, delta=1e-7)

    def test_empty(self):
        graph = retworkx.PyGraph()
        res = retworkx.unweighted_average_shortest_path_length(graph)
        self.assertTrue(math.isnan(res), "Output is not NaN")
