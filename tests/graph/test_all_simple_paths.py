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


class TestGraphAllSimplePaths(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 2),
            (3, 4),
            (4, 2),
            (4, 5),
            (5, 2),
            (5, 3),
        ]

    def test_all_simple_paths(self):
        graph = rustworkx.PyGraph()
        for i in range(6):
            graph.add_node(i)
        graph.add_edges_from_no_data(self.edges)
        paths = rustworkx.graph_all_simple_paths(graph, 0, 5)
        expected = [
            [0, 3, 4, 5],
            [0, 3, 4, 2, 5],
            [0, 3, 4, 2, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 2, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 2, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 1, 2, 4, 5],
            [0, 3, 1, 2, 5],
            [0, 3, 1, 2, 4, 5],
            [0, 2, 4, 5],
            [0, 2, 4, 3, 5],
            [0, 2, 3, 4, 5],
            [0, 2, 3, 5],
            [0, 2, 5],
            [0, 2, 4, 5],
            [0, 2, 4, 3, 5],
            [0, 2, 3, 4, 5],
            [0, 2, 3, 5],
            [0, 2, 1, 3, 4, 5],
            [0, 2, 1, 3, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 2, 5],
            [0, 1, 3, 4, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 5],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 5],
        ]
        self.assertEqual(len(expected), len(paths))
        for i in expected:
            self.assertIn(i, paths)

    def test_all_simple_paths_default_min_depth(self):
        graph = rustworkx.PyGraph()
        for i in range(6):
            graph.add_node(i)
        graph.add_edges_from_no_data(self.edges)
        paths = rustworkx.graph_all_simple_paths(graph, 0, 5, min_depth=0)
        expected = [
            [0, 3, 4, 5],
            [0, 3, 4, 2, 5],
            [0, 3, 4, 2, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 2, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 2, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 1, 2, 4, 5],
            [0, 3, 1, 2, 5],
            [0, 3, 1, 2, 4, 5],
            [0, 2, 4, 5],
            [0, 2, 4, 3, 5],
            [0, 2, 3, 4, 5],
            [0, 2, 3, 5],
            [0, 2, 5],
            [0, 2, 4, 5],
            [0, 2, 4, 3, 5],
            [0, 2, 3, 4, 5],
            [0, 2, 3, 5],
            [0, 2, 1, 3, 4, 5],
            [0, 2, 1, 3, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 2, 5],
            [0, 1, 3, 4, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 5],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 5],
        ]
        self.assertEqual(len(expected), len(paths))
        for i in expected:
            self.assertIn(i, paths)

    def test_all_simple_paths_with_min_depth(self):
        graph = rustworkx.PyGraph()
        for i in range(6):
            graph.add_node(i)
        graph.add_edges_from_no_data(self.edges)
        paths = rustworkx.graph_all_simple_paths(graph, 0, 5, min_depth=6)
        expected = [
            [0, 3, 1, 2, 4, 5],
            [0, 3, 1, 2, 4, 5],
            [0, 2, 1, 3, 4, 5],
            [0, 1, 3, 4, 2, 5],
            [0, 1, 3, 4, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 3, 4, 5],
        ]
        self.assertEqual(len(expected), len(paths))
        for i in expected:
            self.assertIn(i, paths)

    def test_all_simple_paths_with_cutoff(self):
        graph = rustworkx.PyGraph()
        for i in range(6):
            graph.add_node(i)
        graph.add_edges_from_no_data(self.edges)
        paths = rustworkx.graph_all_simple_paths(graph, 0, 5, cutoff=4)
        expected = [
            [0, 3, 4, 5],
            [0, 3, 2, 5],
            [0, 3, 5],
            [0, 3, 2, 5],
            [0, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 2, 5],
            [0, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 1, 3, 5],
            [0, 1, 2, 5],
        ]
        self.assertEqual(len(expected), len(paths))
        for i in expected:
            self.assertIn(i, paths)

    def test_all_simple_paths_with_min_depth_and_cutoff(self):
        graph = rustworkx.PyGraph()
        for i in range(6):
            graph.add_node(i)
        graph.add_edges_from_no_data(self.edges)
        paths = rustworkx.graph_all_simple_paths(graph, 0, 5, min_depth=4, cutoff=4)
        expected = [
            [0, 3, 4, 5],
            [0, 3, 2, 5],
            [0, 3, 2, 5],
            [0, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 1, 3, 5],
            [0, 1, 2, 5],
        ]
        self.assertEqual(len(expected), len(paths))
        for i in expected:
            self.assertIn(i, paths)

    def test_all_simple_path_no_path(self):
        dag = rustworkx.PyGraph()
        dag.add_node(0)
        dag.add_node(1)
        self.assertEqual([], rustworkx.graph_all_simple_paths(dag, 0, 1))

    def test_all_simple_path_invalid_node_index(self):
        dag = rustworkx.PyGraph()
        dag.add_node(0)
        dag.add_node(1)
        with self.assertRaises(rustworkx.InvalidNode):
            rustworkx.graph_all_simple_paths(dag, 0, 5)

    def test_digraph_graph_all_simple_paths(self):
        dag = rustworkx.PyDAG()
        dag.add_node(0)
        dag.add_node(1)
        self.assertRaises(TypeError, rustworkx.graph_all_simple_paths, (dag, 0, 1))


class TestGraphAllSimplePathsAllPairs(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.graph = rustworkx.generators.cycle_graph(4)

    def test_all_simple_paths(self):
        paths = rustworkx.all_pairs_all_simple_paths(self.graph)
        expected = {
            0: {1: [[0, 1], [0, 3, 2, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 1, 2, 3], [0, 3]]},
            1: {2: [[1, 2], [1, 0, 3, 2]], 3: [[1, 2, 3], [1, 0, 3]], 0: [[1, 2, 3, 0], [1, 0]]},
            2: {
                3: [[2, 3], [2, 1, 0, 3]],
                0: [[2, 3, 0], [2, 1, 0]],
                1: [[2, 3, 0, 1], [2, 1]],
            },
            3: {0: [[3, 0], [3, 2, 1, 0]], 1: [[3, 0, 1], [3, 2, 1]], 2: [[3, 0, 1, 2], [3, 2]]},
        }
        self.assertEqual(paths, expected)

    def test_all_simple_paths_min_depth(self):
        paths = rustworkx.all_pairs_all_simple_paths(self.graph, min_depth=3)
        expected = {
            0: {1: [[0, 3, 2, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 1, 2, 3]]},
            1: {2: [[1, 0, 3, 2]], 3: [[1, 2, 3], [1, 0, 3]], 0: [[1, 2, 3, 0]]},
            2: {
                3: [[2, 1, 0, 3]],
                0: [[2, 3, 0], [2, 1, 0]],
                1: [[2, 3, 0, 1]],
            },
            3: {0: [[3, 2, 1, 0]], 1: [[3, 0, 1], [3, 2, 1]], 2: [[3, 0, 1, 2]]},
        }
        self.assertEqual(paths, expected)

    def test_all_simple_paths_with_cutoff(self):
        paths = rustworkx.all_pairs_all_simple_paths(self.graph, cutoff=3)
        expected = {
            0: {1: [[0, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 3]]},
            1: {2: [[1, 2]], 3: [[1, 2, 3], [1, 0, 3]], 0: [[1, 0]]},
            2: {
                3: [[2, 3]],
                0: [[2, 3, 0], [2, 1, 0]],
                1: [[2, 1]],
            },
            3: {0: [[3, 0]], 1: [[3, 0, 1], [3, 2, 1]], 2: [[3, 2]]},
        }
        self.assertEqual(paths, expected)

    def test_all_simple_paths_with_min_depth_and_cutoff(self):
        paths = rustworkx.all_pairs_all_simple_paths(self.graph, min_depth=3, cutoff=3)
        expected = {
            0: {2: [[0, 1, 2], [0, 3, 2]]},
            1: {3: [[1, 2, 3], [1, 0, 3]]},
            2: {0: [[2, 3, 0], [2, 1, 0]]},
            3: {1: [[3, 0, 1], [3, 2, 1]]},
        }
        self.assertEqual(paths, expected)

    def test_all_simple_path_no_path(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        graph.add_node(1)
        self.assertEqual({0: {}, 1: {}}, rustworkx.all_pairs_all_simple_paths(graph))

    def test_all_simple_paths_empty(self):
        self.assertEqual({}, rustworkx.all_pairs_all_simple_paths(rustworkx.PyGraph()))


class TestGraphLongestSimplePath(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.graph = rustworkx.generators.cycle_graph(4)

    def test_all_simple_paths(self):
        res = rustworkx.longest_simple_path(self.graph)
        expected = {
            (0, 3, 2, 1),
            (0, 1, 2, 3),
            (1, 0, 3, 2),
            (1, 2, 3, 0),
            (2, 1, 0, 3),
            (2, 3, 0, 1),
            (3, 0, 1, 2),
            (3, 2, 1, 0),
        }
        self.assertIn(tuple(res), expected)

    def test_all_simple_path_no_path(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        graph.add_node(1)
        self.assertEqual([0], rustworkx.longest_simple_path(graph))

    def test_all_simple_paths_empty(self):
        self.assertIsNone(rustworkx.longest_simple_path(rustworkx.PyGraph()))
