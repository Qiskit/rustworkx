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


class TestConnectedComponents(unittest.TestCase):
    def test_number_connected(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(range(3))
        graph.add_edge(0, 1, None)
        self.assertEqual(rustworkx.number_connected_components(graph), 2)

    def test_number_connected_direct(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(4))
        graph.add_edges_from_no_data([(3, 2), (2, 1), (1, 0)])
        self.assertEqual(len(rustworkx.weakly_connected_components(graph)), 1)

    def test_number_connected_node_holes(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(range(3))
        graph.remove_node(1)
        self.assertEqual(rustworkx.number_connected_components(graph), 2)

    def test_connected_components(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list(
            [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
        )
        components = rustworkx.connected_components(graph)
        self.assertEqual([{0, 1, 2, 3}, {4, 5, 6, 7}], components)

    def test_node_connected_component(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list(
            [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
        )
        component = rustworkx.node_connected_component(graph, 0)
        self.assertEqual({0, 1, 2, 3}, component)

    def test_node_connected_component_invalid_node(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list(
            [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
        )
        with self.assertRaises(rustworkx.InvalidNode):
            rustworkx.node_connected_component(graph, 10)

    def test_is_connected_false(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list(
            [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
        )
        self.assertFalse(rustworkx.is_connected(graph))

    def test_is_connected_true(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (2, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
            ]
        )
        self.assertTrue(rustworkx.is_connected(graph))

    def test_is_connected_null_graph(self):
        graph = rustworkx.PyGraph()
        with self.assertRaises(rustworkx.NullGraph):
            rustworkx.is_connected(graph)
