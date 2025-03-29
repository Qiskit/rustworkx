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


class TestGraphAllShortestPaths(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
        self.a = self.graph.add_node("A")
        self.b = self.graph.add_node("B")
        self.c = self.graph.add_node("C")
        self.d = self.graph.add_node("D")
        self.e = self.graph.add_node("E")
        self.f = self.graph.add_node("F")
        self.graph.add_edge(self.a, self.b, 7)
        self.graph.add_edge(self.c, self.a, 9)
        self.graph.add_edge(self.a, self.d, 14)
        self.graph.add_edge(self.b, self.c, 10)
        self.graph.add_edge(self.d, self.c, 2)
        self.graph.add_edge(self.d, self.e, 9)
        self.graph.add_edge(self.b, self.f, 15)
        self.graph.add_edge(self.c, self.f, 11)
        self.graph.add_edge(self.e, self.f, 6)

    def test_all_shortest_paths_single(self):
        paths = rustworkx.graph_all_shortest_paths(self.graph, self.a, self.e, float)
        # a -> d -> e = 23
        # a -> c -> d -> e = 20
        expected = [[self.a, self.c, self.d, self.e]]
        self.assertEqual(expected, paths)

    def test_all_shortest_paths(self):
        self.graph.update_edge(self.a, self.d, 11)

        paths = rustworkx.graph_all_shortest_paths(self.graph, self.a, self.e, float)
        # a -> d -> e = 20
        # a -> c -> d -> e = 20
        expected = [[self.a, self.d, self.e], [self.a, self.c, self.d, self.e]]
        self.assertEqual(len(paths), 2)
        self.assertIn(expected[0], paths)
        self.assertIn(expected[1], paths)

    def test_all_shortest_paths_with_no_path(self):
        g = rustworkx.PyGraph()
        a = g.add_node("A")
        b = g.add_node("B")
        paths = rustworkx.graph_all_shortest_paths(g, a, b, float)
        expected = []
        self.assertEqual(expected, paths)

    def test_all_shortest_paths_with_invalid_weights(self):
        graph = rustworkx.generators.path_graph(2)
        for invalid_weight in [float("nan"), -1]:
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises(ValueError):
                    rustworkx.graph_all_shortest_paths(
                        graph,
                        source=0,
                        target=1,
                        weight_fn=lambda _: invalid_weight,
                    )

    def test_all_shortest_paths_graph_with_digraph_input(self):
        g = rustworkx.PyDAG()
        g.add_node(0)
        g.add_node(1)
        with self.assertRaises(TypeError):
            rustworkx.graph_all_shortest_paths(g, 0, 1, lambda x: x)

    def test_all_shortest_paths_digraph(self):
        g = rustworkx.PyDAG()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1, 1)
        paths_directed = rustworkx.digraph_all_shortest_paths(g, 1, 0, lambda x: x)
        self.assertEqual([], paths_directed)

        paths_undirected = rustworkx.digraph_all_shortest_paths(
            g, 1, 0, lambda x: x, as_undirected=True
        )
        self.assertEqual([[1, 0]], paths_undirected)
