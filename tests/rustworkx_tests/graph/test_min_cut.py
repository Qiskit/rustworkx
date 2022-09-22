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
import numpy


class TestMinCut(unittest.TestCase):
    def test_min_cut_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.stoer_wagner_min_cut(graph)
        self.assertEqual(res, None)

    def test_min_cut_graph_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node(None)
        res = rustworkx.stoer_wagner_min_cut(graph)
        self.assertEqual(res, None)

    def test_min_cut_graph_single_edge(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list([(0, 1, 10)])
        value, partition = rustworkx.stoer_wagner_min_cut(graph, lambda x: x)
        self.assertEqual(value, 10.0)
        self.assertEqual(partition, [1])

    def test_min_cut_graph_parallel_edge(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list([(0, 1, 4), (0, 1, 6)])
        value, partition = rustworkx.stoer_wagner_min_cut(graph, lambda x: x)
        self.assertEqual(value, 10.0)
        self.assertEqual(partition, [1])

    def test_min_cut_path_graph(self):
        graph = rustworkx.generators.path_graph(4)
        value, _ = rustworkx.stoer_wagner_min_cut(graph)
        self.assertEqual(value, 1.0)

    def test_min_cut_grid_graph(self):
        graph = rustworkx.generators.grid_graph(4, 4)
        value, _ = rustworkx.stoer_wagner_min_cut(graph)
        self.assertEqual(value, 2.0)

    def test_min_cut_example_graph(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (0, 1, 2),
                (1, 2, 3),
                (2, 3, 4),
                (4, 5, 3),
                (5, 6, 1),
                (6, 7, 3),
                (0, 4, 3),
                (1, 5, 2),
                (2, 6, 2),
                (3, 7, 2),
                (1, 4, 2),
                (3, 6, 2),
            ]
        )
        value, _ = rustworkx.stoer_wagner_min_cut(graph, lambda x: x)
        self.assertEqual(value, 4.0)

    def test_min_cut_example_graph_node_hole(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (0, 1, 2),
                (1, 2, 3),
                (2, 3, 4),
                (4, 5, 3),
                (5, 6, 1),
                (6, 7, 3),
                (0, 4, 3),
                (1, 5, 2),
                (2, 6, 2),
                (3, 7, 2),
                (1, 4, 2),
                (3, 6, 2),
            ]
        )
        graph.remove_node(5)
        value, _ = rustworkx.stoer_wagner_min_cut(graph, lambda x: x)
        self.assertEqual(value, 3.0)

    def test_min_cut_disconnected_graph(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list([(0, 1, 1), (2, 3, 1)])
        value, _ = rustworkx.stoer_wagner_min_cut(graph, lambda x: x)
        self.assertEqual(value, 0.0)

    def test_min_cut_graph_nan_edge_weight(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list([(0, 1, 4), (0, 1, numpy.nan)])
        value, partition = rustworkx.stoer_wagner_min_cut(graph, lambda x: x)
        self.assertEqual(value, 4.0)
        self.assertEqual(partition, [1])

    def test_min_cut_invalid_edge_weight(self):
        graph = rustworkx.generators.path_graph(3)
        with self.assertRaises(TypeError):
            rustworkx.stoer_wagner_min_cut(graph, lambda x: x)
