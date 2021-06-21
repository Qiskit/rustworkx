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


class TestIsomorphic(unittest.TestCase):
    def test_isomorphic_identical(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )

        nodes = g_b.add_nodes_from(["a_1", "a_2", "a_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(g_a, g_b, id_order=id_order)
                )

    def test_isomorphic_mismatch_node_data(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )

        nodes = g_b.add_nodes_from(["b_1", "b_2", "b_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "b_1"), (nodes[1], nodes[2], "b_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(g_a, g_b, id_order=id_order)
                )

    def test_isomorphic_compare_nodes_mismatch_node_data(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )

        nodes = g_b.add_nodes_from(["b_1", "b_2", "b_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "b_1"), (nodes[1], nodes[2], "b_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertFalse(
                    retworkx.is_isomorphic(
                        g_a, g_b, lambda x, y: x == y, id_order=id_order
                    )
                )

    def test_is_isomorphic_nodes_compare_raises(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )

        nodes = g_b.add_nodes_from(["b_1", "b_2", "b_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "b_1"), (nodes[1], nodes[2], "b_2")]
        )

        def compare_nodes(a, b):
            raise TypeError("Failure")

        self.assertRaises(
            TypeError, retworkx.is_isomorphic, (g_a, g_b, compare_nodes)
        )

    def test_isomorphic_compare_nodes_identical(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )

        nodes = g_b.add_nodes_from(["a_1", "a_2", "a_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(
                        g_a, g_b, lambda x, y: x == y, id_order=id_order
                    )
                )

    def test_isomorphic_compare_edges_identical(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )

        nodes = g_b.add_nodes_from(["a_1", "a_2", "a_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(
                        g_a,
                        g_b,
                        edge_matcher=lambda x, y: x == y,
                        id_order=id_order,
                    )
                )

    def test_isomorphic_removed_nodes_in_second_graph(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )

        nodes = g_b.add_nodes_from(["a_0", "a_2", "a_1", "a_3"])
        g_b.add_edges_from(
            [
                (nodes[0], nodes[1], "e_01"),
                (nodes[0], nodes[3], "e_03"),
                (nodes[2], nodes[1], "a_1"),
                (nodes[1], nodes[3], "a_2"),
            ]
        )
        g_b.remove_node(nodes[0])
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(
                        g_a, g_b, lambda x, y: x == y, id_order=id_order
                    )
                )

    def test_isomorphic_node_count_not_equal(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from([(nodes[0], nodes[1], "a_1")])

        nodes = g_b.add_nodes_from(["a_0", "a_1"])
        g_b.add_edges_from([(nodes[0], nodes[1], "a_1")])
        g_b.remove_node(nodes[0])
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertFalse(
                    retworkx.is_isomorphic(g_a, g_b, id_order=id_order)
                )

    def test_same_degrees_non_isomorphic(self):
        g_a = retworkx.PyGraph()
        g_b = retworkx.PyGraph()

        nodes = g_a.add_nodes_from(
            ["a_1", "a_2", "a_3", "a_4", "b_1", "b_2", "b_3", "b_4"]
        )
        g_a.add_edges_from(
            [
                (nodes[0], nodes[1], "a_1"),
                (nodes[1], nodes[2], "a_2"),
                (nodes[2], nodes[3], "a_3"),
                (nodes[3], nodes[0], "a_4"),
                (nodes[4], nodes[5], "b_1"),
                (nodes[5], nodes[6], "b_2"),
                (nodes[6], nodes[7], "b_3"),
                (nodes[7], nodes[4], "b_4"),
                (nodes[0], nodes[4], "e_1"),
                (nodes[1], nodes[5], "e_2"),
            ]
        )

        nodes = g_b.add_nodes_from(
            ["a_1", "a_2", "a_3", "a_4", "b_1", "b_2", "b_3", "b_4"]
        )
        g_b.add_edges_from(
            [
                (nodes[0], nodes[1], "a_1"),
                (nodes[1], nodes[2], "a_2"),
                (nodes[2], nodes[3], "a_3"),
                (nodes[3], nodes[0], "a_4"),
                (nodes[4], nodes[5], "b_1"),
                (nodes[5], nodes[6], "b_2"),
                (nodes[6], nodes[7], "b_3"),
                (nodes[7], nodes[4], "b_4"),
                (nodes[0], nodes[4], "e_1"),
                (nodes[2], nodes[6], "e_2"),
            ]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertFalse(
                    retworkx.is_isomorphic(g_a, g_b, id_order=id_order)
                )

    def test_graph_vf2_mapping_identical(self):
        graph = retworkx.generators.grid_graph(2, 2)
        second_graph = retworkx.generators.grid_graph(2, 2)
        mapping = retworkx.graph_vf2_mapping(graph, second_graph)
        self.assertEqual(mapping, {0: 0, 1: 1, 2: 2, 3: 3})

    def test_graph_vf2_mapping_identical_removals(self):
        graph = retworkx.generators.path_graph(2)
        second_graph = retworkx.generators.path_graph(4)
        second_graph.remove_nodes_from([1, 2])
        second_graph.add_edge(0, 3, None)
        mapping = retworkx.graph_vf2_mapping(graph, second_graph)
        self.assertEqual({0: 0, 1: 3}, mapping)

    def test_graph_vf2_mapping_identical_removals_first(self):
        second_graph = retworkx.generators.path_graph(2)
        graph = retworkx.generators.path_graph(4)
        graph.remove_nodes_from([1, 2])
        graph.add_edge(0, 3, None)
        mapping = retworkx.graph_vf2_mapping(
            graph,
            second_graph,
        )
        self.assertEqual({0: 0, 3: 1}, mapping)

    def test_subgraph_vf2_mapping(self):
        graph = retworkx.generators.grid_graph(10, 10)
        second_graph = retworkx.generators.grid_graph(2, 2)
        mapping = retworkx.graph_vf2_mapping(graph, second_graph, subgraph=True)
        self.assertEqual(mapping, {0: 0, 1: 1, 10: 2, 11: 3})

    def test_graph_vf2_mapping_identical_vf2pp(self):
        graph = retworkx.generators.grid_graph(2, 2)
        second_graph = retworkx.generators.grid_graph(2, 2)
        mapping = retworkx.graph_vf2_mapping(
            graph, second_graph, id_order=False
        )
        valid_mappings = [
            {0: 0, 1: 1, 2: 2, 3: 3},
            {0: 0, 1: 2, 2: 1, 3: 3},
        ]
        self.assertIn(mapping, valid_mappings)

    def test_graph_vf2_mapping_identical_removals_vf2pp(self):
        graph = retworkx.generators.path_graph(2)
        second_graph = retworkx.generators.path_graph(4)
        second_graph.remove_nodes_from([1, 2])
        second_graph.add_edge(0, 3, None)
        mapping = retworkx.graph_vf2_mapping(
            graph, second_graph, id_order=False
        )
        self.assertEqual({0: 0, 1: 3}, mapping)

    def test_graph_vf2_mapping_identical_removals_first_vf2pp(self):
        second_graph = retworkx.generators.path_graph(2)
        graph = retworkx.generators.path_graph(4)
        graph.remove_nodes_from([1, 2])
        graph.add_edge(0, 3, None)
        mapping = retworkx.graph_vf2_mapping(
            graph, second_graph, id_order=False
        )
        self.assertEqual({0: 0, 3: 1}, mapping)

    def test_subgraph_vf2_mapping_vf2pp(self):
        graph = retworkx.generators.grid_graph(3, 3)
        second_graph = retworkx.generators.grid_graph(2, 2)
        mapping = retworkx.graph_vf2_mapping(
            graph, second_graph, subgraph=True, id_order=False
        )
        valid_mappings = [
            {3: 2, 4: 3, 6: 0, 7: 1},
            {3: 1, 4: 3, 6: 0, 7: 2},
            {4: 3, 5: 1, 7: 2, 8: 0},
            {0: 0, 1: 1, 3: 2, 4: 3},
            {7: 1, 8: 0, 4: 3, 5: 2},
            {5: 1, 2: 0, 1: 2, 4: 3},
            {3: 1, 0: 0, 4: 3, 1: 2},
            {1: 1, 2: 0, 4: 3, 5: 2},
        ]
        self.assertIn(mapping, valid_mappings)

    def test_vf2pp_remapping(self):
        temp = retworkx.generators.grid_graph(3, 3)

        graph = retworkx.PyGraph()
        dummy = graph.add_node(0)

        graph.compose(temp, dict())
        graph.remove_node(dummy)

        second_graph = retworkx.generators.grid_graph(2, 2)
        mapping = retworkx.graph_vf2_mapping(
            graph, second_graph, subgraph=True, id_order=False
        )
        expected_mappings = [
            {2: 2, 3: 0, 5: 3, 6: 1},
            {5: 3, 6: 1, 8: 2, 9: 0},
            {2: 1, 5: 3, 6: 2, 3: 0},
            {2: 2, 1: 0, 5: 3, 4: 1},
            {5: 3, 6: 2, 8: 1, 9: 0},
            {4: 2, 1: 0, 5: 3, 2: 1},
            {5: 3, 8: 1, 4: 2, 7: 0},
            {5: 3, 4: 1, 7: 0, 8: 2},
        ]
        self.assertIn(mapping, expected_mappings)
