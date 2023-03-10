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


class TestCompose(unittest.TestCase):
    def test_simple_graph_composition(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {"a": 1})
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, {"a": 2})
        graph_other = rustworkx.PyGraph()
        node_d = graph_other.add_node("d")
        node_e = graph_other.add_node("e")
        graph_other.add_edge(node_d, node_e, {"a": 3})
        res = graph.compose(graph_other, {node_c: (node_d, {"b": 1})})
        self.assertEqual({0: 3, 1: 4}, res)
        self.assertEqual([0, 1, 2, 3, 4], graph.node_indexes())

    def test_edge_map_and_node_map_funcs_graph_compose(self):
        graph = rustworkx.PyGraph()
        original_input_nodes = graph.add_nodes_from(["qr[0]", "qr[1]"])
        original_op_nodes = graph.add_nodes_from(["h"])
        output_nodes = graph.add_nodes_from(["qr[0]", "qr[1]"])
        graph.add_edge(original_input_nodes[0], original_op_nodes[0], "qr[0]")
        graph.add_edge(original_op_nodes[0], output_nodes[0], "qr[0]")
        # Setup other graph
        other_graph = rustworkx.PyGraph()
        input_nodes = other_graph.add_nodes_from(["qr[2]", "qr[3]"])
        op_nodes = other_graph.add_nodes_from(["cx"])
        other_output_nodes = other_graph.add_nodes_from(["qr[2]", "qr[3]"])
        other_graph.add_edges_from(
            [
                (input_nodes[0], op_nodes[0], "qr[2]"),
                (input_nodes[1], op_nodes[0], "qr[3]"),
            ]
        )
        other_graph.add_edges_from(
            [
                (op_nodes[0], other_output_nodes[0], "qr[2]"),
                (op_nodes[0], other_output_nodes[1], "qr[3]"),
            ]
        )

        def map_fn(weight):
            if weight == "qr[2]":
                return "qr[0]"
            elif weight == "qr[3]":
                return "qr[1]"
            else:
                return weight

        graph.remove_nodes_from(output_nodes)
        other_graph.remove_nodes_from(input_nodes)
        node_map = {
            original_op_nodes[0]: (op_nodes[0], "qr[0]"),
            original_input_nodes[1]: (op_nodes[0], "qr[1]"),
        }
        res = graph.compose(other_graph, node_map, node_map_func=map_fn, edge_map_func=map_fn)
        self.assertEqual({2: 4, 3: 3, 4: 5}, res)
        self.assertEqual(graph[res[other_output_nodes[0]]], "qr[0]")
        self.assertEqual(graph[res[other_output_nodes[1]]], "qr[1]")
        # qr[0] -> h
        self.assertTrue(graph.has_edge(0, 2))
        self.assertTrue(graph.get_all_edge_data(0, 2), ["qr[0]"])
        # qr[1] -> cx
        self.assertTrue(graph.has_edge(1, 4))
        self.assertTrue(graph.get_all_edge_data(0, 2), ["qr[1]"])
        # h -> cx
        self.assertTrue(graph.has_edge(2, 4))
        self.assertTrue(graph.get_all_edge_data(0, 2), ["qr[0]"])
        # cx -> qr[0]
        self.assertTrue(graph.has_edge(4, 3))
        self.assertTrue(graph.get_all_edge_data(0, 2), ["qr[0]"])
        # cx -> qr[1]
        self.assertTrue(graph.has_edge(4, 5))
        self.assertTrue(graph.get_all_edge_data(0, 2), ["qr[1]"])

    def test_compose_digraph_onto_graph_error(self):
        digraph = rustworkx.PyDiGraph()
        graph = rustworkx.PyGraph()
        with self.assertRaises(TypeError):
            graph.compose(digraph, {})
