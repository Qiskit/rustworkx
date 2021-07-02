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

import copy
import unittest

import retworkx


class TestIsomorphic(unittest.TestCase):
    def test_isomorphic_identical(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("a_1")
        dag_b.add_child(node_b, "a_2", "a_1")
        dag_b.add_child(node_b, "a_3", "a_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(dag_a, dag_b, id_order=id_order)
                )

    def test_isomorphic_mismatch_node_data(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("b_1")
        dag_b.add_child(node_b, "b_2", "b_1")
        dag_b.add_child(node_b, "b_3", "b_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(dag_a, dag_b, id_order=id_order)
                )

    def test_isomorphic_compare_nodes_mismatch_node_data(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("b_1")
        dag_b.add_child(node_b, "b_2", "b_1")
        dag_b.add_child(node_b, "b_3", "b_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertFalse(
                    retworkx.is_isomorphic(
                        dag_a, dag_b, lambda x, y: x == y, id_order=id_order
                    )
                )

    def test_is_isomorphic_nodes_compare_raises(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("b_1")
        dag_b.add_child(node_b, "b_2", "b_1")
        dag_b.add_child(node_b, "b_3", "b_2")

        def compare_nodes(a, b):
            raise TypeError("Failure")

        self.assertRaises(
            TypeError, retworkx.is_isomorphic, (dag_a, dag_b, compare_nodes)
        )

    def test_isomorphic_compare_nodes_identical(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("a_1")
        dag_b.add_child(node_b, "a_2", "a_1")
        dag_b.add_child(node_b, "a_3", "a_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(
                        dag_a, dag_b, lambda x, y: x == y, id_order=id_order
                    )
                )

    def test_isomorphic_compare_edges_identical(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("a_1")
        dag_b.add_child(node_b, "a_2", "a_1")
        dag_b.add_child(node_b, "a_3", "a_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(
                        dag_a,
                        dag_b,
                        edge_matcher=lambda x, y: x == y,
                        id_order=id_order,
                    )
                )

    def test_isomorphic_compare_nodes_with_removals(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        qr_0_in = dag_a.add_node("qr[0]")
        qr_1_in = dag_a.add_node("qr[1]")
        cr_0_in = dag_a.add_node("cr[0]")
        qr_0_out = dag_a.add_node("qr[0]")
        qr_1_out = dag_a.add_node("qr[1]")
        cr_0_out = dag_a.add_node("qr[0]")
        cu1 = dag_a.add_child(qr_0_in, "cu1", "qr[0]")
        dag_a.add_edge(qr_1_in, cu1, "qr[1]")
        measure_0 = dag_a.add_child(cr_0_in, "measure", "cr[0]")
        dag_a.add_edge(cu1, measure_0, "qr[0]")
        measure_1 = dag_a.add_child(cu1, "measure", "qr[1]")
        dag_a.add_edge(measure_0, measure_1, "cr[0]")
        dag_a.add_edge(measure_1, qr_1_out, "qr[1]")
        dag_a.add_edge(measure_1, cr_0_out, "cr[0]")
        dag_a.add_edge(measure_0, qr_0_out, "qr[0]")
        dag_a.remove_node(cu1)
        dag_a.add_edge(qr_0_in, measure_0, "qr[0]")
        dag_a.add_edge(qr_1_in, measure_1, "qr[1]")

        qr_0_in = dag_b.add_node("qr[0]")
        qr_1_in = dag_b.add_node("qr[1]")
        cr_0_in = dag_b.add_node("cr[0]")
        qr_0_out = dag_b.add_node("qr[0]")
        qr_1_out = dag_b.add_node("qr[1]")
        cr_0_out = dag_b.add_node("qr[0]")
        measure_0 = dag_b.add_child(cr_0_in, "measure", "cr[0]")
        dag_b.add_edge(qr_0_in, measure_0, "qr[0]")
        measure_1 = dag_b.add_child(qr_1_in, "measure", "qr[1]")
        dag_b.add_edge(measure_1, qr_1_out, "qr[1]")
        dag_b.add_edge(measure_1, cr_0_out, "cr[0]")
        dag_b.add_edge(measure_0, measure_1, "cr[0]")
        dag_b.add_edge(measure_0, qr_0_out, "qr[0]")

        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(
                        dag_a, dag_b, lambda x, y: x == y, id_order=id_order
                    )
                )

    def test_isomorphic_compare_nodes_with_removals_deepcopy(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        qr_0_in = dag_a.add_node("qr[0]")
        qr_1_in = dag_a.add_node("qr[1]")
        cr_0_in = dag_a.add_node("cr[0]")
        qr_0_out = dag_a.add_node("qr[0]")
        qr_1_out = dag_a.add_node("qr[1]")
        cr_0_out = dag_a.add_node("qr[0]")
        cu1 = dag_a.add_child(qr_0_in, "cu1", "qr[0]")
        dag_a.add_edge(qr_1_in, cu1, "qr[1]")
        measure_0 = dag_a.add_child(cr_0_in, "measure", "cr[0]")
        dag_a.add_edge(cu1, measure_0, "qr[0]")
        measure_1 = dag_a.add_child(cu1, "measure", "qr[1]")
        dag_a.add_edge(measure_0, measure_1, "cr[0]")
        dag_a.add_edge(measure_1, qr_1_out, "qr[1]")
        dag_a.add_edge(measure_1, cr_0_out, "cr[0]")
        dag_a.add_edge(measure_0, qr_0_out, "qr[0]")
        dag_a.remove_node(cu1)
        dag_a.add_edge(qr_0_in, measure_0, "qr[0]")
        dag_a.add_edge(qr_1_in, measure_1, "qr[1]")

        qr_0_in = dag_b.add_node("qr[0]")
        qr_1_in = dag_b.add_node("qr[1]")
        cr_0_in = dag_b.add_node("cr[0]")
        qr_0_out = dag_b.add_node("qr[0]")
        qr_1_out = dag_b.add_node("qr[1]")
        cr_0_out = dag_b.add_node("qr[0]")
        measure_0 = dag_b.add_child(cr_0_in, "measure", "cr[0]")
        dag_b.add_edge(qr_0_in, measure_0, "qr[0]")
        measure_1 = dag_b.add_child(qr_1_in, "measure", "qr[1]")
        dag_b.add_edge(measure_1, qr_1_out, "qr[1]")
        dag_b.add_edge(measure_1, cr_0_out, "cr[0]")
        dag_b.add_edge(measure_0, measure_1, "cr[0]")
        dag_b.add_edge(measure_0, qr_0_out, "qr[0]")

        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    retworkx.is_isomorphic(
                        copy.deepcopy(dag_a),
                        copy.deepcopy(dag_b),
                        lambda x, y: x == y,
                        id_order=id_order,
                    )
                )

    def test_digraph_vf2_mapping_identical(self):
        graph = retworkx.generators.directed_grid_graph(2, 2)
        second_graph = retworkx.generators.directed_grid_graph(2, 2)
        mapping = retworkx.digraph_vf2_mapping(graph, second_graph)
        self.assertEqual(next(mapping), {0: 0, 1: 1, 2: 2, 3: 3})

    def test_digraph_vf2_mapping_identical_removals(self):
        graph = retworkx.generators.directed_path_graph(2)
        second_graph = retworkx.generators.directed_path_graph(4)
        second_graph.remove_nodes_from([1, 2])
        second_graph.add_edge(0, 3, None)
        mapping = retworkx.digraph_vf2_mapping(graph, second_graph)
        self.assertEqual({0: 0, 1: 3}, next(mapping))

    def test_digraph_vf2_mapping_identical_removals_first(self):
        second_graph = retworkx.generators.directed_path_graph(2)
        graph = retworkx.generators.directed_path_graph(4)
        graph.remove_nodes_from([1, 2])
        graph.add_edge(0, 3, None)
        mapping = retworkx.digraph_vf2_mapping(graph, second_graph)
        self.assertEqual({0: 0, 3: 1}, next(mapping))

    def test_subgraph_vf2_mapping(self):
        graph = retworkx.generators.directed_grid_graph(10, 10)
        second_graph = retworkx.generators.directed_grid_graph(2, 2)
        mapping = retworkx.digraph_vf2_mapping(
            graph, second_graph, subgraph=True
        )
        self.assertEqual(next(mapping), {0: 0, 1: 1, 10: 2, 11: 3})

    def test_digraph_vf2_mapping_identical_vf2pp(self):
        graph = retworkx.generators.directed_grid_graph(2, 2)
        second_graph = retworkx.generators.directed_grid_graph(2, 2)
        mapping = retworkx.digraph_vf2_mapping(
            graph, second_graph, id_order=False
        )
        valid_mappings = [
            {0: 0, 1: 1, 2: 2, 3: 3},
            {0: 0, 1: 2, 2: 1, 3: 3},
        ]
        self.assertIn(next(mapping), valid_mappings)

    def test_graph_vf2_mapping_identical_removals_vf2pp(self):
        graph = retworkx.generators.directed_path_graph(2)
        second_graph = retworkx.generators.directed_path_graph(4)
        second_graph.remove_nodes_from([1, 2])
        second_graph.add_edge(0, 3, None)
        mapping = retworkx.digraph_vf2_mapping(
            graph, second_graph, id_order=False
        )
        self.assertEqual({0: 0, 1: 3}, next(mapping))

    def test_graph_vf2_mapping_identical_removals_first_vf2pp(self):
        second_graph = retworkx.generators.directed_path_graph(2)
        graph = retworkx.generators.directed_path_graph(4)
        graph.remove_nodes_from([1, 2])
        graph.add_edge(0, 3, None)
        mapping = retworkx.digraph_vf2_mapping(
            graph, second_graph, id_order=False
        )
        self.assertEqual({0: 0, 3: 1}, next(mapping))

    def test_subgraph_vf2_mapping_vf2pp(self):
        graph = retworkx.generators.directed_grid_graph(3, 3)
        second_graph = retworkx.generators.directed_grid_graph(2, 2)
        mapping = retworkx.digraph_vf2_mapping(
            graph, second_graph, subgraph=True, id_order=False
        )
        valid_mappings = [
            {8: 3, 5: 2, 7: 1, 4: 0},
            {7: 2, 5: 1, 4: 0, 8: 3},
        ]
        self.assertIn(next(mapping), valid_mappings)

    def test_vf2pp_remapping(self):
        temp = retworkx.generators.directed_grid_graph(3, 3)

        graph = retworkx.PyDiGraph()
        dummy = graph.add_node(0)

        graph.compose(temp, dict())
        graph.remove_node(dummy)

        second_graph = retworkx.generators.directed_grid_graph(2, 2)
        mapping = retworkx.digraph_vf2_mapping(
            graph, second_graph, subgraph=True, id_order=False
        )
        expected_mappings = [
            {6: 1, 5: 0, 8: 2, 9: 3},
            {6: 2, 5: 0, 9: 3, 8: 1},
        ]
        self.assertIn(next(mapping), expected_mappings)
