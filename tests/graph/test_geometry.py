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

import numpy as np

import rustworkx as rx


class TestHyperbolicGreedyRouting(unittest.TestCase):

    def test_invalid_node_error(self):
        graph = rx.PyGraph()
        positions = []
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_routing(graph, positions, 0, 0)

        graph.add_node(0)
        positions.append([0.0, 0.0])
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_routing(graph, positions, 0, 1)
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_routing(graph, positions, 1, 0)

        graph.add_node(1)
        graph.remove_node(0)
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_routing(graph, positions, 0, 1)
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_routing(graph, positions, 1, 0)

    def test_invalid_positions_error(self):
        graph = rx.PyGraph()
        graph.add_nodes_from([0, 1])
        positions = [[0.0, 0.0]]
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_routing(graph, positions, 0, 1)
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_success_rate(graph, positions)

        positions = [[0.0, 0.0], [0.0, 0.0, 0.0]]
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_routing(graph, positions, 0, 1)
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_success_rate(graph, positions)

        positions = [[0.0, 0.0], [0.0]]
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_routing(graph, positions, 0, 1)
        with self.assertRaises(ValueError):
            rx.hyperbolic_greedy_success_rate(graph, positions)

    def test_disconnected_graph_is_none(self):
        # Disconnected graph
        graph = rx.PyGraph()
        graph.add_node(0)
        graph.add_node(1)

        positions = [[0.0, 0.0], [1.0, 0.0]]
        self.assertIsNone(rx.hyperbolic_greedy_routing(graph, positions, 0, 1))

    def test_greedy_loop_is_none(self):
        # 0 -- 1 -- 2 -- 3
        graph = rx.PyGraph()
        graph.add_nodes_from(range(4))
        graph.add_edges_from_no_data([(0, 1), (1, 2), (2, 3)])

        positions = [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [4.0, 0.0]]
        self.assertIsNone(rx.hyperbolic_greedy_routing(graph, positions, 0, 3))

    def test_correct_successful_path(self):
        # 0 -- 1 -- 2 -- 3
        #      |    |
        #      4 -- 5 -- 6
        graph = rx.PyGraph()
        graph.add_nodes_from(range(7))
        graph.add_edges_from_no_data([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (2, 5), (5, 6)])

        positions = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.5, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ]

        path, dist = rx.hyperbolic_greedy_routing(graph, positions, 0, 3)

        def hyperbolic_dist(x, y):
            x_array = np.asarray(x)
            y_array = np.asarray(y)
            dot = np.sum(x_array * y_array)
            arg = (
                np.sqrt(1 + np.sum(x_array * x_array)) * np.sqrt(1 + np.sum(y_array * y_array))
                - dot
            )
            return 0 if arg < 0 else np.arccosh(arg)

        def total_length(path):
            return sum(
                (
                    hyperbolic_dist(positions[i], positions[j])
                    for i, j in zip(path[:-1], np.roll(path, -1)[:-1])
                )
            )

        self.assertEqual(path, [0, 1, 2, 3])
        self.assertAlmostEqual(dist, total_length(path))

        path, dist = rx.hyperbolic_greedy_routing(graph, positions, 0, 6)
        self.assertEqual(path, [0, 1, 2, 5, 6])
        self.assertAlmostEqual(dist, total_length(path))

    def test_correct_greedy_success_rate(self):
        # 0 -- 1 -- 2
        # |    |
        # 3 -- 4 -- 5
        graph = rx.PyGraph()
        graph.add_nodes_from(range(6))
        graph.add_edges_from_no_data([(0, 1), (1, 2), (0, 3), (3, 4), (1, 4), (4, 5)])

        positions = [
            [1.0, 0.0],
            [2.0, 3.0],
            [3.0, 0.0],
            [1.0, -1.0],
            [2.0, -1.0],
            [3.0, -1.0],
        ]
        success_rate = rx.hyperbolic_greedy_success_rate(graph, positions)
        # Greedy paths 0 -> 2, 3->2, 4->2 and 5->2 fail.
        self.assertAlmostEqual(success_rate, 26.0 / 30.0)

    def test_correct_greedy_routing_after_removed_node(self):
        graph = rx.PyGraph()
        graph.add_nodes_from(range(7))
        graph.add_edges_from_no_data([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (2, 5), (5, 6)])

        positions = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 3.0],
            [3.0, 0.0],
            [1.0, -1.0],
            [2.0, -1.0],
            [3.0, -1.0],
        ]
        graph.remove_node(0)

        success_rate = rx.hyperbolic_greedy_success_rate(graph, positions)
        # Greedy paths 0 -> 2, 3->2, 4->2 and 5->2 fail.
        self.assertAlmostEqual(success_rate, 26.0 / 30.0)
