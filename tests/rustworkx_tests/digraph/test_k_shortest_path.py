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


class TestKShortestpath(unittest.TestCase):
    def test_digraph_k_shortest_path_lengths(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(8)))
        graph.add_edges_from_no_data(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (1, 4),
                (5, 6),
                (6, 7),
                (7, 5),
            ]
        )
        res = rustworkx.digraph_k_shortest_path_lengths(graph, 1, 2, lambda _: 1)
        expected = {
            0: 7.0,
            1: 4.0,
            2: 5.0,
            3: 6.0,
            4: 5.0,
            5: 5.0,
            6: 6.0,
            7: 7.0,
        }
        self.assertEqual(res, expected)

    def test_digraph_k_shortest_path_lengths_with_goal(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(8)))
        graph.add_edges_from_no_data(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (1, 4),
                (5, 6),
                (6, 7),
                (7, 5),
            ]
        )
        res = rustworkx.digraph_k_shortest_path_lengths(graph, 1, 2, lambda _: 1, 3)
        self.assertEqual(res, {3: 6})

    def test_digraph_k_shortest_path_with_goal_node_hole(self):
        graph = rustworkx.generators.directed_path_graph(4)
        graph.remove_node(0)
        res = rustworkx.digraph_k_shortest_path_lengths(
            graph, start=1, k=1, edge_cost=lambda _: 1, goal=3
        )
        self.assertEqual({3: 2}, res)

    def test_digraph_k_shortest_path_with_invalid_weight(self):
        graph = rustworkx.generators.directed_path_graph(4)
        for invalid_weight in [float("nan"), -1]:
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises(ValueError):
                    rustworkx.digraph_k_shortest_path_lengths(
                        graph,
                        start=1,
                        k=1,
                        edge_cost=lambda _: invalid_weight,
                        goal=3,
                    )

    def test_k_shortest_path_with_no_path(self):
        g = rustworkx.PyDiGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        path_lenghts = rustworkx.digraph_k_shortest_path_lengths(
            g, start=a, k=1, edge_cost=float, goal=b
        )
        expected = {}
        self.assertEqual(expected, path_lenghts)
