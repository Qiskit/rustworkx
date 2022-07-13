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

import rustworkx


class TestIsomorphic(unittest.TestCase):
    def test_empty_isomorphic(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(rustworkx.is_isomorphic(dag_a, dag_b, id_order=id_order))

    def test_empty_isomorphic_compare_nodes(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    rustworkx.is_isomorphic(dag_a, dag_b, lambda x, y: x == y, id_order=id_order)
                )

    def test_isomorphic_identical(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("a_1")
        dag_b.add_child(node_b, "a_2", "a_1")
        dag_b.add_child(node_b, "a_3", "a_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(rustworkx.is_isomorphic(dag_a, dag_b, id_order=id_order))

    def test_isomorphic_mismatch_node_data(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("b_1")
        dag_b.add_child(node_b, "b_2", "b_1")
        dag_b.add_child(node_b, "b_3", "b_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(rustworkx.is_isomorphic(dag_a, dag_b, id_order=id_order))

    def test_isomorphic_compare_nodes_mismatch_node_data(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("b_1")
        dag_b.add_child(node_b, "b_2", "b_1")
        dag_b.add_child(node_b, "b_3", "b_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertFalse(
                    rustworkx.is_isomorphic(dag_a, dag_b, lambda x, y: x == y, id_order=id_order)
                )

    def test_is_isomorphic_nodes_compare_raises(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("b_1")
        dag_b.add_child(node_b, "b_2", "b_1")
        dag_b.add_child(node_b, "b_3", "b_2")

        def compare_nodes(a, b):
            raise TypeError("Failure")

        self.assertRaises(TypeError, rustworkx.is_isomorphic, (dag_a, dag_b, compare_nodes))

    def test_isomorphic_compare_nodes_identical(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("a_1")
        dag_b.add_child(node_b, "a_2", "a_1")
        dag_b.add_child(node_b, "a_3", "a_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    rustworkx.is_isomorphic(dag_a, dag_b, lambda x, y: x == y, id_order=id_order)
                )

    def test_isomorphic_compare_edges_identical(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

        node_a = dag_a.add_node("a_1")
        dag_a.add_child(node_a, "a_2", "a_1")
        dag_a.add_child(node_a, "a_3", "a_2")

        node_b = dag_b.add_node("a_1")
        dag_b.add_child(node_b, "a_2", "a_1")
        dag_b.add_child(node_b, "a_3", "a_2")
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                self.assertTrue(
                    rustworkx.is_isomorphic(
                        dag_a,
                        dag_b,
                        edge_matcher=lambda x, y: x == y,
                        id_order=id_order,
                    )
                )

    def test_isomorphic_compare_nodes_with_removals(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

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
                    rustworkx.is_isomorphic(dag_a, dag_b, lambda x, y: x == y, id_order=id_order)
                )

    def test_isomorphic_compare_nodes_with_removals_deepcopy(self):
        dag_a = rustworkx.PyDAG()
        dag_b = rustworkx.PyDAG()

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
                    rustworkx.is_isomorphic(
                        copy.deepcopy(dag_a),
                        copy.deepcopy(dag_b),
                        lambda x, y: x == y,
                        id_order=id_order,
                    )
                )

    def test_digraph_isomorphic_parallel_edges_with_edge_matcher(self):
        graph = rustworkx.PyDiGraph()
        graph.extend_from_weighted_edge_list([(0, 1, "a"), (0, 1, "b"), (1, 2, "c")])
        self.assertTrue(rustworkx.is_isomorphic(graph, graph, edge_matcher=lambda x, y: x == y))

    def test_digraph_isomorphic_self_loop(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from([0])
        graph.add_edges_from([(0, 0, "a")])
        self.assertTrue(rustworkx.is_isomorphic(graph, graph))

    def test_digraph_non_isomorphic_edge_mismatch_self_loop(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from([0])
        graph.add_edges_from([(0, 0, "a")])
        second_graph = rustworkx.PyDiGraph()
        second_graph.add_nodes_from([0])
        second_graph.add_edges_from([(0, 0, "b")])
        self.assertFalse(
            rustworkx.is_isomorphic(graph, second_graph, edge_matcher=lambda x, y: x == y)
        )

    def test_digraph_non_isomorphic_rule_out_incoming(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from_no_data([(0, 1), (0, 2), (2, 1)])
        second_graph = rustworkx.PyDiGraph()
        second_graph.add_nodes_from([0, 1, 2, 3])
        second_graph.add_edges_from_no_data([(0, 1), (0, 2), (3, 1)])
        self.assertFalse(rustworkx.is_isomorphic(graph, second_graph, id_order=True))

    def test_digraph_non_isomorphic_rule_ins_outgoing(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from_no_data([(1, 0), (2, 0), (1, 2)])
        second_graph = rustworkx.PyDiGraph()
        second_graph.add_nodes_from([0, 1, 2, 3])
        second_graph.add_edges_from_no_data([(1, 0), (2, 0), (1, 3)])
        self.assertFalse(rustworkx.is_isomorphic(graph, second_graph, id_order=True))

    def test_digraph_non_isomorphic_rule_ins_incoming(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from_no_data([(1, 0), (2, 0), (2, 1)])
        second_graph = rustworkx.PyDiGraph()
        second_graph.add_nodes_from([0, 1, 2, 3])
        second_graph.add_edges_from_no_data([(1, 0), (2, 0), (3, 1)])
        self.assertFalse(rustworkx.is_isomorphic(graph, second_graph, id_order=True))

    def test_isomorphic_parallel_edges(self):
        first = rustworkx.PyDiGraph()
        first.extend_from_edge_list([(0, 1), (0, 1), (1, 2), (2, 3)])
        second = rustworkx.PyDiGraph()
        second.extend_from_edge_list([(0, 1), (1, 2), (1, 2), (2, 3)])
        self.assertFalse(rustworkx.is_isomorphic(first, second))

    def test_digraph_isomorphic_insufficient_call_limit(self):
        graph = rustworkx.generators.directed_path_graph(5)
        self.assertFalse(rustworkx.is_isomorphic(graph, graph, call_limit=2))

    def test_digraph_vf2_mapping_identical(self):
        graph = rustworkx.generators.directed_grid_graph(2, 2)
        second_graph = rustworkx.generators.directed_grid_graph(2, 2)
        mapping = rustworkx.digraph_vf2_mapping(graph, second_graph)
        self.assertEqual(next(mapping), {0: 0, 1: 1, 2: 2, 3: 3})

    def test_digraph_vf2_mapping_identical_removals(self):
        graph = rustworkx.generators.directed_path_graph(2)
        second_graph = rustworkx.generators.directed_path_graph(4)
        second_graph.remove_nodes_from([1, 2])
        second_graph.add_edge(0, 3, None)
        mapping = rustworkx.digraph_vf2_mapping(graph, second_graph)
        self.assertEqual({0: 0, 1: 3}, next(mapping))

    def test_digraph_vf2_mapping_identical_removals_first(self):
        second_graph = rustworkx.generators.directed_path_graph(2)
        graph = rustworkx.generators.directed_path_graph(4)
        graph.remove_nodes_from([1, 2])
        graph.add_edge(0, 3, None)
        mapping = rustworkx.digraph_vf2_mapping(graph, second_graph)
        self.assertEqual({0: 0, 3: 1}, next(mapping))

    def test_digraph_vf2_mapping_identical_vf2pp(self):
        graph = rustworkx.generators.directed_grid_graph(2, 2)
        second_graph = rustworkx.generators.directed_grid_graph(2, 2)
        mapping = rustworkx.digraph_vf2_mapping(graph, second_graph, id_order=False)
        self.assertEqual(next(mapping), {0: 0, 1: 1, 2: 2, 3: 3})

    def test_digraph_vf2_mapping_identical_removals_vf2pp(self):
        graph = rustworkx.generators.directed_path_graph(2)
        second_graph = rustworkx.generators.directed_path_graph(4)
        second_graph.remove_nodes_from([1, 2])
        second_graph.add_edge(0, 3, None)
        mapping = rustworkx.digraph_vf2_mapping(graph, second_graph, id_order=False)
        self.assertEqual({0: 0, 1: 3}, next(mapping))

    def test_digraph_vf2_mapping_identical_removals_first_vf2pp(self):
        second_graph = rustworkx.generators.directed_path_graph(2)
        graph = rustworkx.generators.directed_path_graph(4)
        graph.remove_nodes_from([1, 2])
        graph.add_edge(0, 3, None)
        mapping = rustworkx.digraph_vf2_mapping(graph, second_graph, id_order=False)
        self.assertEqual({0: 0, 3: 1}, next(mapping))

    def test_digraph_vf2_number_of_valid_mappings(self):
        graph = rustworkx.generators.directed_mesh_graph(3)
        mapping = rustworkx.digraph_vf2_mapping(graph, graph, id_order=True)
        total = 0
        for _ in mapping:
            total += 1
        self.assertEqual(total, 6)

    def test_empty_digraph_vf2_mapping(self):
        g_a = rustworkx.PyDiGraph()
        g_b = rustworkx.PyDiGraph()
        for id_order in [False, True]:
            with self.subTest(id_order=id_order):
                mapping = rustworkx.digraph_vf2_mapping(g_a, g_b, id_order=id_order, subgraph=False)
                self.assertEqual({}, next(mapping))
