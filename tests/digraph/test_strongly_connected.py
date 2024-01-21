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


class TestStronglyConnected(unittest.TestCase):
    def test_number_strongly_connected_all_strong(self):
        G = rustworkx.PyDiGraph()
        node_a = G.add_node(1)
        node_b = G.add_child(node_a, 2, {})
        node_c = G.add_child(node_b, 3, {})
        self.assertEqual(
            rustworkx.strongly_connected_components(G),
            [[node_c], [node_b], [node_a]],
        )

    def test_number_strongly_connected(self):
        G = rustworkx.PyDiGraph()
        node_a = G.add_node(1)
        node_b = G.add_child(node_a, 2, {})
        node_c = G.add_node(3)
        self.assertEqual(
            rustworkx.strongly_connected_components(G),
            [[node_c], [node_b], [node_a]],
        )

    def test_stongly_connected_no_linear(self):
        G = rustworkx.PyDiGraph()
        G.add_nodes_from(list(range(8)))
        G.add_edges_from_no_data(
            [
                (0, 1),
                (1, 2),
                (1, 7),
                (2, 3),
                (2, 6),
                (3, 4),
                (4, 2),
                (4, 5),
                (6, 3),
                (6, 5),
                (7, 0),
                (7, 6),
            ]
        )
        expected = [[5], [2, 3, 4, 6], [0, 1, 7]]
        components = rustworkx.strongly_connected_components(G)
        self.assertEqual(components, expected)

    def test_number_strongly_connected_big(self):
        G = rustworkx.PyDiGraph()
        for i in range(100000):
            node = G.add_node(i)
            G.add_child(node, str(i), {})
        self.assertEqual(len(rustworkx.strongly_connected_components(G)), 200000)
