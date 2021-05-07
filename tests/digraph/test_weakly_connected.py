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

import retworkx


class TestWeaklyConnected(unittest.TestCase):
    def test_number_weakly_connected_all_strong(self):
        G = retworkx.PyDAG()
        node_a = G.add_node(1)
        node_b = G.add_child(node_a, 2, {})
        G.add_child(node_b, 3, {})
        self.assertEqual(retworkx.number_weakly_connected_components(G), 1)

    def test_number_weakly_connected(self):
        G = retworkx.PyDAG()
        node_a = G.add_node(1)
        G.add_child(node_a, 2, {})
        G.add_node(3)
        self.assertEqual(retworkx.number_weakly_connected_components(G), 2)

    def test_number_weakly_connected_big(self):
        G = retworkx.PyDAG()
        for i in range(100000):
            node = G.add_node(i)
            G.add_child(node, str(i), {})
        self.assertEqual(retworkx.number_weakly_connected_components(G), 100000)

    def test_weakly_connected_components(self):
        graph = retworkx.PyDiGraph()
        graph.extend_from_edge_list(
            [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
        )
        components = retworkx.weakly_connected_components(graph)
        self.assertEqual([{0, 1, 2, 3}, {4, 5, 6, 7}], components)

    def test_is_weakly_connected_false(self):
        graph = retworkx.PyDiGraph()
        graph.extend_from_edge_list(
            [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
        )
        self.assertFalse(retworkx.is_weakly_connected(graph))

    def test_is_weakly_connected_true(self):
        graph = retworkx.PyDiGraph()
        graph.extend_from_edge_list(
            [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5), (5, 6), (6, 7), (7, 4)]
        )
        self.assertTrue(retworkx.is_weakly_connected(graph))

    def test_is_weakly_connected_null_graph(self):
        graph = retworkx.PyDiGraph()
        with self.assertRaises(retworkx.NullGraph):
            retworkx.is_weakly_connected(graph)
