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


class TestNumShortestpath(unittest.TestCase):
    def test_num_shortest_path_unweighted(self):
        graph = rustworkx.PyDiGraph()
        node_a = graph.add_node(0)
        node_b = graph.add_node("end")
        for i in range(3):
            node = graph.add_child(node_a, i, None)
            graph.add_edge(node, node_b, None)
        res = rustworkx.digraph_num_shortest_paths_unweighted(graph, node_a)
        expected = {2: 1, 4: 1, 3: 1, 1: 3}
        self.assertEqual(expected, res)

    def test_parallel_paths(self):
        graph = rustworkx.PyDiGraph()
        graph.extend_from_edge_list(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (0, 4),
                (4, 5),
                (5, 3),
            ]
        )
        res = rustworkx.num_shortest_paths_unweighted(graph, 0)
        expected = {
            1: 1,
            2: 1,
            3: 2,
            4: 1,
            5: 1,
        }
        self.assertEqual(expected, res)

    def test_grid_graph(self):
        """Test num shortest paths for a 5x5 grid graph
        0 -> 1 -> 2 -> 3 -> 4
        |    |    |    |    |
        v    v    v    v    v
        5 -> 6 -> 7 -> 8 -> 9
        |    |    |    |    |
        v    v    v    v    v
        10-> 11-> 12-> 13-> 14
        |    |    |    |    |
        v    v    v    v    v
        15-> 16-> 17-> 18-> 19
        |    |    |    |    |
        v    v    v    v    v
        20-> 21-> 22-> 23-> 24
        """
        graph = rustworkx.generators.directed_grid_graph(5, 5)
        res = rustworkx.num_shortest_paths_unweighted(graph, 0)
        expected = {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 3,
            8: 4,
            9: 5,
            10: 1,
            11: 3,
            12: 6,
            13: 10,
            14: 15,
            15: 1,
            16: 4,
            17: 10,
            18: 20,
            19: 35,
            20: 1,
            21: 5,
            22: 15,
            23: 35,
            24: 70,
        }
        self.assertEqual(expected, res)

    def test_node_with_no_path(self):
        graph = rustworkx.generators.directed_path_graph(5)
        graph.extend_from_edge_list([(6, 7), (7, 8), (8, 9), (9, 10), (10, 11)])
        expected = {1: 1, 2: 1, 3: 1, 4: 1}
        res = rustworkx.num_shortest_paths_unweighted(graph, 0)
        self.assertEqual(expected, res)
        res = rustworkx.num_shortest_paths_unweighted(graph, 6)
        expected = {7: 1, 8: 1, 9: 1, 10: 1, 11: 1}
        self.assertEqual(expected, res)

    def test_node_indices_with_holes(self):
        graph = rustworkx.generators.directed_path_graph(5)
        graph.extend_from_edge_list([(6, 7), (7, 8), (8, 9), (9, 10), (10, 11)])
        graph.add_edge(4, 6, None)
        graph.remove_node(5)
        expected = {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 1,
            11: 1,
        }
        res = rustworkx.num_shortest_paths_unweighted(graph, 0)
        self.assertEqual(expected, res)

    def test_no_edges(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        graph.add_node(1)
        res = rustworkx.num_shortest_paths_unweighted(graph, 0)
        self.assertEqual({}, res)

    def test_invalid_source_index(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        graph.add_child(0, 1, None)
        with self.assertRaises(IndexError):
            rustworkx.num_shortest_paths_unweighted(graph, 4)
