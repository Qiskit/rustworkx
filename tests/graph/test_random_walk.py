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

import collections
import unittest

import rustworkx as rx


class TestRandomWalk(unittest.TestCase):
    def test_invalid_node_error(self):
        graph = rx.PyGraph()
        with self.assertRaises(IndexError):
            rx.generate_random_path_graph(graph, 0, 10, None)

    def test_zero_degree_early_stop(self):
        graph = rx.PyGraph()
        graph.add_node(0)
        res = rx.generate_random_path_graph(graph, 0, 10, None)
        self.assertEqual(res, [0])

    def test_node_frequency(self):
        # a -- b -- c -- d
        #         / |
        #      e -- f -- g
        graph = rx.PyGraph()

        graph.add_nodes_from(range(7))
        graph.add_edges_from_no_data([(0, 1), (1, 2), (2, 3), (4, 5), (2, 4), (2, 5), (5, 6)])

        path_length = 5_000
        path = rx.generate_random_path_graph(graph, 0, path_length, 5)
        counts = collections.Counter(path)

        # Expected frequency is degree/2 number of edges.
        tol = 1e-2
        self.assertAlmostEqual(counts[0] / (path_length + 1), 1 / 14, delta=tol)
        self.assertAlmostEqual(counts[1] / (path_length + 1), 2 / 14, delta=tol)
        self.assertAlmostEqual(counts[2] / (path_length + 1), 4 / 14, delta=tol)
        self.assertAlmostEqual(counts[3] / (path_length + 1), 1 / 14, delta=tol)
        self.assertAlmostEqual(counts[4] / (path_length + 1), 2 / 14, delta=tol)
        self.assertAlmostEqual(counts[5] / (path_length + 1), 3 / 14, delta=tol)
        self.assertAlmostEqual(counts[6] / (path_length + 1), 1 / 14, delta=tol)
