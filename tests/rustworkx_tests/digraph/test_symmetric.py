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


class TestSymmetric(unittest.TestCase):
    def test_single_neighbor(self):
        digraph = rustworkx.PyDiGraph()
        node_a = digraph.add_node("a")
        digraph.add_child(node_a, "b", {"a": 1})
        digraph.add_child(node_a, "c", {"a": 2})
        self.assertFalse(digraph.is_symmetric())

    def test_bidirectional_ring(self):
        digraph = rustworkx.PyDiGraph()
        edge_list = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 0),
            (0, 3),
        ]
        digraph.extend_from_edge_list(edge_list)
        self.assertTrue(digraph.is_symmetric())

    def test_empty_graph_make_symmetric(self):
        digraph = rustworkx.PyDiGraph()
        digraph.make_symmetric()
        self.assertEqual(0, digraph.num_edges())
        self.assertEqual(0, digraph.num_nodes())

    def test_path_graph_make_symmetric(self):
        digraph = rustworkx.generators.directed_path_graph(4)
        digraph.make_symmetric()
        expected_edge_list = [
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 0),
            (2, 1),
            (3, 2),
        ]
        self.assertEqual(digraph.edge_list(), expected_edge_list)

    def test_path_graph_make_symmetric_existing_reverse_edges(self):
        digraph = rustworkx.generators.directed_path_graph(4)
        digraph.add_edge(3, 2, None)
        digraph.add_edge(1, 0, None)
        digraph.make_symmetric()
        expected_edge_list = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 2),
            (1, 0),
            (2, 1),
        ]
        self.assertEqual(digraph.edge_list(), expected_edge_list)

    def test_empty_graph_make_symmetric_with_function_arg(self):
        digraph = rustworkx.PyDiGraph()
        digraph.make_symmetric(lambda _: "Reversi")
        self.assertEqual(0, digraph.num_edges())
        self.assertEqual(0, digraph.num_nodes())

    def test_path_graph_make_symmetric_with_function_arg(self):
        digraph = rustworkx.generators.directed_path_graph(4)
        digraph.make_symmetric(lambda: _: "Reversi")
        expected_edge_list = [
            (0, 1, "Reversi"),
            (1, 2, "Reversi"),
            (2, 3, "Reversi"),
            (1, 0, "Reversi"),
            (2, 1, "Reversi"),
            (3, 2, "Reversi"),
        ]
        self.assertEqual(digraph.weighted_edge_list(), expected_edge_list)

    def test_path_graph_make_symmetric_existing_reverse_edges_function_arg(self):
        digraph = rustworkx.generators.directed_path_graph(4)
        digraph.add_edge(3, 2, None)
        digraph.add_edge(1, 0, None)
        digraph.make_symmetric(lambda _: "Reversi")
        expected_edge_list = [
            (0, 1, "Reversi"),
            (1, 2, "Reversi"),
            (2, 3, "Reversi"),
            (3, 2, None),
            (1, 0, None),
            (2, 1, "Reversi"),
        ]
        self.assertEqual(digraph.weighted_edge_list(), expected_edge_list)

    def test_path_graph_make_symmetric_function_arg_raises(self):
        digraph = rustworkx.generators.directed_path_graph(4)

        def weight_function(edge):
            if edge is None:
                raise TypeError("I'm expected")

        with self.assertRaises(TypeError):
            digraph.make_symmetric(weight_function)
