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
import rustworkx.generators


class TestSemiConnected(unittest.TestCase):
    def test_is_semi_connected_true(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(3)))
        graph.add_edge(0, 1, None)
        graph.add_edge(1, 2, None)

        self.assertTrue(rustworkx.is_semi_connected(graph))

    def test_is_semi_connected_reverse(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(3)))
        graph.add_edge(0, 1, None)
        graph.add_edge(2, 1, None)

        self.assertTrue(rustworkx.is_semi_connected(graph))

    def test_is_semi_connected_single_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        self.assertTrue(rustworkx.is_semi_connected(graph))

    def test_is_semi_connected_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        with self.assertRaises(rustworkx.NullGraph):
            rustworkx.is_semi_connected(graph)

    def test_is_semi_connected_disconnected_graph(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        graph.add_node(1)

        self.assertFalse(rustworkx.is_semi_connected(graph))

    def test_is_semi_connected_directed_cycle_graph(self):
        graph = rustworkx.generators.directed_cycle_graph(100)
        self.assertTrue(rustworkx.is_semi_connected(graph))

    def test_is_semi_connected_directed_path_graph(self):
        graph = rustworkx.generators.directed_path_graph(100)
        self.assertTrue(rustworkx.is_semi_connected(graph))
  
    def test_is_semi_connected_directed_star_graph(self):
        graph = rustworkx.generators.directed_star_graph(100)
        self.assertFalse(rustworkx.is_semi_connected(graph))
  
    def test_is_semi_connected_directed_grid_graph(self):
        graph = rustworkx.generators.directed_grid_graph(50,50)
        self.assertFalse(rustworkx.is_semi_connected(graph))
