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


class TestDijkstraCutoff(unittest.TestCase):
    graph_types = (rustworkx.PyGraph, rustworkx.PyDiGraph)

    def test_inclusive_boundary_and_shorter_alternative(self):
        for graph_type in self.graph_types:
            graph = graph_type()
            graph.add_nodes_from(range(4))
            graph.add_edges_from([(0, 1, 20), (0, 1, 2), (0, 2, 10), (1, 2, 3), (2, 3, 1)])
            with self.subTest(graph_type=graph_type):
                self.assertEqual(
                    {1: 2.0, 2: 5.0},
                    rustworkx.dijkstra_shortest_path_lengths(graph, 0, float, cutoff=5.0),
                )

    def test_zero_weight_paths_and_cycles(self):
        for graph_type in self.graph_types:
            graph = graph_type()
            graph.add_nodes_from(range(4))
            graph.add_edges_from([(0, 1, 0), (1, 0, 0), (1, 2, 0), (2, 3, 1)])
            with self.subTest(graph_type=graph_type):
                self.assertEqual(
                    {1: 0.0, 2: 0.0},
                    rustworkx.dijkstra_shortest_path_lengths(graph, 0, float, cutoff=0.0),
                )

    def test_goal_inside_and_outside_cutoff(self):
        for graph_type in self.graph_types:
            graph = graph_type()
            graph.add_nodes_from(range(4))
            graph.add_edges_from([(0, 1, 2), (1, 2, 3)])
            for goal, expected in [(0, {0: 0.0}), (1, {1: 2.0}), (2, {}), (3, {})]:
                with self.subTest(graph_type=graph_type, goal=goal):
                    self.assertEqual(
                        expected,
                        rustworkx.dijkstra_shortest_path_lengths(graph, 0, float, goal, cutoff=2.0),
                    )

    def test_none_and_infinity_match_unbounded_search(self):
        for graph_type in self.graph_types:
            graph = graph_type()
            graph.add_nodes_from(range(3))
            graph.add_edges_from([(0, 1, 2), (1, 2, float("inf"))])
            expected = rustworkx.dijkstra_shortest_path_lengths(graph, 0, float)
            for cutoff in [None, float("inf")]:
                with self.subTest(graph_type=graph_type, cutoff=cutoff):
                    self.assertEqual(
                        expected,
                        rustworkx.dijkstra_shortest_path_lengths(graph, 0, float, cutoff=cutoff),
                    )

    def test_invalid_cutoff(self):
        for graph_type in self.graph_types:
            graph = graph_type()
            graph.add_node(None)
            for cutoff in [-1.0, float("-inf"), float("nan")]:
                with self.subTest(graph_type=graph_type, cutoff=cutoff):
                    with self.assertRaisesRegex(ValueError, "cutoff"):
                        rustworkx.dijkstra_shortest_path_lengths(graph, 0, float, cutoff=cutoff)

    def test_direct_graph_functions(self):
        functions = (
            (rustworkx.PyGraph, rustworkx.graph_dijkstra_shortest_path_lengths),
            (rustworkx.PyDiGraph, rustworkx.digraph_dijkstra_shortest_path_lengths),
        )
        for graph_type, function in functions:
            graph = graph_type()
            graph.add_nodes_from(range(3))
            graph.add_edges_from([(0, 1, 1), (1, 2, 1)])
            with self.subTest(function=function):
                self.assertEqual({1: 1.0}, function(graph, 0, float, cutoff=1.0))

    def test_stops_before_visiting_distant_edges(self):
        for graph_type in self.graph_types:
            graph = graph_type()
            graph.add_nodes_from(range(100))
            graph.add_edges_from((i, i + 1, i) for i in range(99))
            seen = []

            def cost(edge):
                seen.append(edge)
                return 1.0

            with self.subTest(graph_type=graph_type):
                result = rustworkx.dijkstra_shortest_path_lengths(graph, 0, cost, cutoff=2.0)
                self.assertEqual({1: 1.0, 2: 2.0}, result)
                self.assertEqual([0, 1, 2], seen)

    def test_callback_errors_and_invalid_weights(self):
        for graph_type in self.graph_types:
            graph = graph_type()
            graph.add_nodes_from(range(2))
            graph.add_edge(0, 1, None)
            for weight in [-1.0, float("nan")]:
                with self.subTest(graph_type=graph_type, weight=weight):
                    with self.assertRaises(ValueError):
                        rustworkx.dijkstra_shortest_path_lengths(
                            graph, 0, lambda _: weight, cutoff=0.0
                        )

            def cost(_):
                raise RuntimeError("edge callback failed")

            with self.subTest(graph_type=graph_type):
                with self.assertRaisesRegex(RuntimeError, "edge callback failed"):
                    rustworkx.dijkstra_shortest_path_lengths(graph, 0, cost, cutoff=0.0)

    def test_removed_nodes_and_self_loops(self):
        for graph_type in self.graph_types:
            graph = graph_type()
            graph.add_nodes_from(range(5))
            graph.remove_node(1)
            graph.add_edges_from([(0, 0, 0), (0, 2, 1), (2, 3, 1), (3, 4, 1)])
            with self.subTest(graph_type=graph_type):
                self.assertEqual(
                    {2: 1.0, 3: 2.0},
                    rustworkx.dijkstra_shortest_path_lengths(graph, 0, float, cutoff=2.0),
                )

    def test_invalid_source(self):
        for graph_type in self.graph_types:
            graph = graph_type()
            with self.subTest(graph_type=graph_type):
                with self.assertRaises(IndexError):
                    rustworkx.dijkstra_shortest_path_lengths(graph, 0, float, cutoff=0.0)
