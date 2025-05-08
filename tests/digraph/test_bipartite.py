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


class TestBipartite(unittest.TestCase):
    def test_is_bipartite(self):
        graph = rustworkx.generators.directed_heavy_square_graph(5)
        self.assertTrue(rustworkx.is_bipartite(graph))

    def test_two_colors(self):
        graph = rustworkx.generators.directed_star_graph(5)
        self.assertEqual(rustworkx.two_color(graph), {0: 1, 1: 0, 2: 0, 3: 0, 4: 0})

    def test_two_colors_reverse_direction(self):
        graph = rustworkx.generators.directed_star_graph(5, inward=True)
        self.assertEqual(rustworkx.two_color(graph), {0: 1, 1: 0, 2: 0, 3: 0, 4: 0})

    def test_two_colors_with_isolates(self):
        graph = rustworkx.generators.directed_star_graph(5)
        graph.add_nodes_from(range(3))
        self.assertEqual(
            rustworkx.two_color(graph), {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1}
        )

    def test_is_bipartite_with_isolates(self):
        graph = rustworkx.generators.directed_star_graph(5)
        graph.add_nodes_from(range(3))
        self.assertTrue(rustworkx.is_bipartite(graph))

    def test_two_colors_not_bipartite_with_isolates(self):
        graph = rustworkx.generators.directed_complete_graph(5)
        graph.add_nodes_from(range(3))
        self.assertIsNone(rustworkx.two_color(graph))

    def test_not_bipartite_with_isolates(self):
        graph = rustworkx.generators.directed_complete_graph(5)
        graph.add_nodes_from(range(3))
        self.assertFalse(rustworkx.is_bipartite(graph))

    def test_not_bipartite(self):
        graph = rustworkx.generators.directed_complete_graph(5)
        self.assertFalse(rustworkx.is_bipartite(graph))

    def test_two_color_not_bipartite(self):
        graph = rustworkx.generators.directed_complete_graph(5)
        self.assertIsNone(rustworkx.two_color(graph))

    def test_grid_graph(self):
        for i in range(10):
            for j in range(10):
                with self.subTest((i, j)):
                    graph = rustworkx.generators.directed_grid_graph(i, j)
                    self.assertTrue(rustworkx.is_bipartite(graph))

    def test_cycle_graph(self):
        for i in range(20):
            with self.subTest(i):
                graph = rustworkx.generators.directed_cycle_graph(i)
                res = rustworkx.is_bipartite(graph)
                if i % 2:
                    self.assertFalse(res)
                else:
                    self.assertTrue(res)
