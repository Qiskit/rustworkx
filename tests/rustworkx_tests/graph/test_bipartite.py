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
        graph = rustworkx.generators.heavy_square_graph(5)
        self.assertTrue(rustworkx.is_bipartite(graph))

    def test_two_colors(self):
        graph = rustworkx.generators.star_graph(5)
        self.assertEqual(rustworkx.two_color(graph), {0: 1, 1: 0, 2: 0, 3: 0, 4: 0})

    def test_two_colors_with_isolates(self):
        graph = rustworkx.generators.star_graph(5)
        graph.add_nodes_from(range(3))
        self.assertEqual(
            rustworkx.two_color(graph), {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1}
        )

    def test_is_bipartite_with_isolates(self):
        graph = rustworkx.generators.star_graph(5)
        graph.add_nodes_from(range(3))
        self.assertTrue(rustworkx.is_bipartite(graph))

    def test_two_colors_not_biparite_with_isolates(self):
        graph = rustworkx.generators.complete_graph(5)
        graph.add_nodes_from(range(3))
        self.assertIsNone(rustworkx.two_color(graph))

    def test_not_biparite_with_isolates(self):
        graph = rustworkx.generators.complete_graph(5)
        graph.add_nodes_from(range(3))
        self.assertFalse(rustworkx.is_bipartite(graph))

    def test_not_biparite(self):
        graph = rustworkx.generators.complete_graph(5)
        self.assertFalse(rustworkx.is_bipartite(graph))

    def test_two_color_not_biparite(self):
        graph = rustworkx.generators.complete_graph(5)
        self.assertIsNone(rustworkx.two_color(graph))
