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


class TestSubgraphIsomorphic(unittest.TestCase):
    def test_subgraph_isomorphic_identical(self):
        g_a = retworkx.PyDiGraph()
        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_subgraph_isomorphic(g_a, g_a, id_order=id_order)
                )

    def test_subgraph_isomorphic_mismatch_node_data(self):
        g_a = retworkx.PyDiGraph()
        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3", "a_4"])
        g_a.add_edges_from(
            [
                (nodes[0], nodes[1], "a_1"),
                (nodes[1], nodes[2], "a_2"),
                (nodes[0], nodes[3], "a_3"),
            ]
        )

        nodes = g_b.add_nodes_from(["b_1", "b_2", "b_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "b_1"), (nodes[1], nodes[2], "b_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_subgraph_isomorphic(g_a, g_b, id_order=id_order)
                )

    def test_subgraph_isomorphic_compare_nodes_mismatch_node_data(self):
        g_a = retworkx.PyDiGraph()
        g_b = retworkx.PyDiGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3", "a_4"])
        g_a.add_edges_from(
            [
                (nodes[0], nodes[1], "a_1"),
                (nodes[1], nodes[2], "a_2"),
                (nodes[0], nodes[3], "a_3"),
            ]
        )

        nodes = g_b.add_nodes_from(["b_1", "b_2", "b_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "b_1"), (nodes[1], nodes[2], "b_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertFalse(
                    retworkx.is_subgraph_isomorphic(
                        g_a, g_b, lambda x, y: x == y, id_order=id_order
                    )
                )

    def test_subgraph_isomorphic_compare_nodes_identical(self):
        g_a = retworkx.PyDiGraph()
        g_b = retworkx.PyDiGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3", "a_4"])
        g_a.add_edges_from(
            [
                (nodes[0], nodes[1], "a_1"),
                (nodes[1], nodes[2], "a_2"),
                (nodes[0], nodes[3], "a_3"),
            ]
        )

        nodes = g_b.add_nodes_from(["a_1", "a_2", "a_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_subgraph_isomorphic(
                        g_a, g_b, lambda x, y: x == y, id_order=id_order
                    )
                )

    def test_is_subgraph_isomorphic_nodes_compare_raises(self):
        g_a = retworkx.PyDiGraph()
        g_b = retworkx.PyDiGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )

        def compare_nodes(a, b):
            raise TypeError("Failure")

        self.assertRaises(
            TypeError,
            retworkx.is_subgraph_isomorphic,
            (g_a, g_a, compare_nodes),
        )

    def test_subgraph_isomorphic_compare_edges_identical(self):
        g_a = retworkx.PyDiGraph()
        g_b = retworkx.PyDiGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3", "a_4"])
        g_a.add_edges_from(
            [
                (nodes[0], nodes[1], "a_1"),
                (nodes[1], nodes[2], "a_2"),
                (nodes[0], nodes[3], "a_3"),
            ]
        )

        nodes = g_b.add_nodes_from(["a_1", "a_2", "a_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_subgraph_isomorphic(
                        g_a,
                        g_b,
                        edge_matcher=lambda x, y: x == y,
                        id_order=id_order,
                    )
                )

    def test_subgraph_isomorphic_node_count_not_ge(self):
        g_a = retworkx.PyDiGraph()
        g_b = retworkx.PyDiGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2"])
        g_a.add_edges_from([(nodes[0], nodes[1], "a_1")])

        nodes = g_b.add_nodes_from(["a_0", "a_1", "a_3"])
        g_b.add_edges_from([(nodes[0], nodes[1], "a_1")])
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertFalse(
                    retworkx.is_subgraph_isomorphic(g_a, g_b, id_order=id_order)
                )

    def test_non_induced_subgraph_isomorphic(self):
        g_a = retworkx.PyDiGraph()
        g_b = retworkx.PyDiGraph()

        nodes = g_a.add_nodes_from(["a_1", "a_2", "a_3"])
        g_a.add_edges_from(
            [
                (nodes[0], nodes[1], "a_1"),
                (nodes[1], nodes[2], "a_2"),
                (nodes[2], nodes[0], "a_3"),
            ]
        )

        nodes = g_b.add_nodes_from(["a_1", "a_2", "a_3"])
        g_b.add_edges_from(
            [(nodes[0], nodes[1], "a_1"), (nodes[1], nodes[2], "a_2")]
        )
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertFalse(
                    retworkx.is_subgraph_isomorphic(g_a, g_b, id_order=id_order)
                )
