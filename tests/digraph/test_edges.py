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


class TestEdges(unittest.TestCase):
    def test_get_edge_data(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        res = dag.get_edge_data(node_a, node_b)
        self.assertEqual("Edgy", res)

    def test_get_all_edge_data(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_edge(node_a, node_b, "b")
        res = dag.get_all_edge_data(node_a, node_b)
        self.assertIn("b", res)
        self.assertIn("Edgy", res)

    def test_no_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_node("b")
        self.assertRaises(
            retworkx.NoEdgeBetweenNodes, dag.get_edge_data, node_a, node_b
        )

    def test_num_edges(self):
        graph = retworkx.PyDiGraph()
        graph.add_node(1)
        graph.add_node(42)
        graph.add_node(146)
        graph.add_edges_from_no_data([(0, 1), (1, 2)])
        self.assertEqual(2, graph.num_edges())

    def test_num_edges_no_edges(self):
        graph = retworkx.PyDiGraph()
        graph.add_node(1)
        graph.add_node(42)
        self.assertEqual(0, graph.num_edges())

    def test_update_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "not edgy")
        dag.update_edge(node_a, node_b, "Edgy")
        self.assertEqual([(0, 1, "Edgy")], dag.weighted_edge_list())

    def test_update_edge_no_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_node("b")
        self.assertRaises(
            retworkx.NoEdgeBetweenNodes, dag.update_edge, node_a, node_b, None
        )

    def test_update_edge_by_index(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", "not edgy")
        dag.update_edge_by_index(0, "Edgy")
        self.assertEqual([(0, 1, "Edgy")], dag.weighted_edge_list())

    def test_update_edge_invalid_index(self):
        dag = retworkx.PyDAG()
        dag.add_node("a")
        dag.add_node("b")
        self.assertRaises(IndexError, dag.update_edge_by_index, 0, None)

    def test_update_edge_parallel_edges(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "not edgy")
        edge_index = graph.add_edge(node_a, node_b, "not edgy")
        graph.update_edge_by_index(edge_index, "Edgy")
        self.assertEqual(
            [(0, 1, "not edgy"), (0, 1, "Edgy")],
            list(graph.weighted_edge_list()),
        )

    def test_has_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        self.assertTrue(dag.has_edge(node_a, node_b))

    def test_has_edge_no_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_node("b")
        self.assertFalse(dag.has_edge(node_a, node_b))

    def test_edges(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_child(node_b, "c", "Super edgy")
        self.assertEqual(["Edgy", "Super edgy"], dag.edges())

    def test_edges_empty(self):
        dag = retworkx.PyDAG()
        dag.add_node("a")
        self.assertEqual([], dag.edges())

    def test_edge_indices(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_child(node_b, "c", "Super edgy")
        self.assertEqual([0, 1], dag.edge_indices())

    def test_edge_indices_empty(self):
        dag = retworkx.PyDAG()
        dag.add_node("a")
        self.assertEqual([], dag.edge_indices())

    def test_add_duplicates(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "a", "a")
        dag.add_edge(node_a, node_b, "b")
        self.assertEqual(["a", "b"], dag.edges())

    def test_remove_no_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_node("b")
        self.assertRaises(
            retworkx.NoEdgeBetweenNodes, dag.remove_edge, node_a, node_b
        )

    def test_remove_edge_single(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "edgy")
        dag.remove_edge(node_a, node_b)
        self.assertEqual([], dag.edges())

    def test_remove_multiple(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "edgy")
        dag.add_edge(node_a, node_b, "super_edgy")
        dag.remove_edge_from_index(0)
        self.assertEqual(["super_edgy"], dag.edges())

    def test_remove_edges_from(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_b, "edgy")
        graph.add_edge(node_a, node_c, "super_edgy")
        graph.remove_edges_from([(node_a, node_b), (node_a, node_c)])
        self.assertEqual([], graph.edges())

    def test_remove_edges_from_invalid(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_b, "edgy")
        graph.add_edge(node_a, node_c, "super_edgy")
        with self.assertRaises(retworkx.NoEdgeBetweenNodes):
            graph.remove_edges_from([(node_b, node_c), (node_a, node_c)])

    def test_remove_edge_from_index(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", "edgy")
        dag.remove_edge_from_index(0)
        self.assertEqual([], dag.edges())

    def test_remove_edge_no_edge(self):
        dag = retworkx.PyDAG()
        dag.add_node("a")
        dag.remove_edge_from_index(0)
        self.assertEqual([], dag.edges())

    def test_add_cycle(self):
        dag = retworkx.PyDAG()
        dag.check_cycle = True
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        self.assertRaises(
            retworkx.DAGWouldCycle, dag.add_edge, node_b, node_a, {}
        )

    def test_add_edge_with_cycle_check_enabled(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node("a")
        node_c = dag.add_node("c")
        node_b = dag.add_child(node_a, "b", {})
        dag.add_edge(node_c, node_b, {})
        self.assertTrue(dag.has_edge(node_c, node_b))

    def test_enable_cycle_checking_after_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        dag.add_edge(node_b, node_a, {})
        with self.assertRaises(retworkx.DAGHasCycle):
            dag.check_cycle = True

    def test_cycle_checking_at_init(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        with self.assertRaises(retworkx.DAGWouldCycle):
            dag.add_edge(node_b, node_a, {})

    def test_find_adjacent_node_by_edge(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"weights": [1, 2]})
        dag.add_child(node_a, "c", {"weights": [3, 4]})

        def compare_edges(edge):
            return 4 in edge["weights"]

        res = dag.find_adjacent_node_by_edge(node_a, compare_edges)
        self.assertEqual("c", res)

    def test_find_adjacent_node_by_edge_no_match(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"weights": [1, 2]})
        dag.add_child(node_a, "c", {"weights": [3, 4]})

        def compare_edges(edge):
            return 5 in edge["weights"]

        with self.assertRaises(retworkx.NoSuitableNeighbors):
            dag.find_adjacent_node_by_edge(node_a, compare_edges)

    def test_add_edge_from(self):
        dag = retworkx.PyDAG()
        nodes = list(range(4))
        dag.add_nodes_from(nodes)
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        res = dag.add_edges_from(edge_list)
        self.assertEqual(len(res), 5)
        self.assertEqual(["a", "b", "c", "d", "e"], dag.edges())
        self.assertEqual(3, dag.out_degree(0))
        self.assertEqual(0, dag.in_degree(0))
        self.assertEqual(1, dag.out_degree(1))
        self.assertEqual(1, dag.out_degree(2))
        self.assertEqual(2, dag.in_degree(3))

    def test_add_edge_from_empty(self):
        dag = retworkx.PyDAG()
        res = dag.add_edges_from([])
        self.assertEqual([], res)

    def test_cycle_checking_at_init_nodes_from(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        node_c = dag.add_child(node_b, "c", {})
        with self.assertRaises(retworkx.DAGWouldCycle):
            dag.add_edges_from([(node_a, node_c, {}), (node_c, node_b, {})])

    def test_is_directed_acyclic_graph(self):
        dag = retworkx.generators.directed_path_graph(1000)
        res = retworkx.is_directed_acyclic_graph(dag)
        self.assertTrue(res)

    def test_is_directed_acyclic_graph_false(self):
        digraph = retworkx.generators.directed_cycle_graph(1000)
        self.assertFalse(retworkx.is_directed_acyclic_graph(digraph))

    def test_add_edge_from_no_data(self):
        dag = retworkx.PyDAG()
        nodes = list(range(4))
        dag.add_nodes_from(nodes)
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        res = dag.add_edges_from_no_data(edge_list)
        self.assertEqual(len(res), 5)
        self.assertEqual([None, None, None, None, None], dag.edges())
        self.assertEqual(3, dag.out_degree(0))
        self.assertEqual(0, dag.in_degree(0))
        self.assertEqual(1, dag.out_degree(1))
        self.assertEqual(1, dag.out_degree(2))
        self.assertEqual(2, dag.in_degree(3))

    def test_add_edge_from_empty_no_data(self):
        dag = retworkx.PyDAG()
        res = dag.add_edges_from_no_data([])
        self.assertEqual([], res)

    def test_cycle_checking_at_init_nodes_from_no_data(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        node_c = dag.add_child(node_b, "c", {})
        with self.assertRaises(retworkx.DAGWouldCycle):
            dag.add_edges_from_no_data([(node_a, node_c), (node_c, node_b)])

    def test_edge_list(self):
        dag = retworkx.PyDiGraph()
        dag.add_nodes_from(list(range(4)))
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        dag.add_edges_from(edge_list)
        self.assertEqual([(x[0], x[1]) for x in edge_list], dag.edge_list())

    def test_edge_list_empty(self):
        dag = retworkx.PyDiGraph()
        self.assertEqual([], dag.edge_list())

    def test_weighted_edge_list(self):
        dag = retworkx.PyDiGraph()
        dag.add_nodes_from(list(range(4)))
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        dag.add_edges_from(edge_list)
        self.assertEqual(edge_list, dag.weighted_edge_list())

    def test_weighted_edge_list_empty(self):
        dag = retworkx.PyDiGraph()
        self.assertEqual([], dag.weighted_edge_list())

    def test_extend_from_edge_list(self):
        dag = retworkx.PyDAG()
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        dag.extend_from_edge_list(edge_list)
        self.assertEqual(len(dag), 4)
        self.assertEqual([None] * 5, dag.edges())
        self.assertEqual(3, dag.out_degree(0))
        self.assertEqual(0, dag.in_degree(0))
        self.assertEqual(1, dag.out_degree(1))
        self.assertEqual(1, dag.out_degree(2))
        self.assertEqual(2, dag.in_degree(3))

    def test_extend_from_edge_list_empty(self):
        dag = retworkx.PyDAG()
        dag.extend_from_edge_list([])
        self.assertEqual(0, len(dag))

    def test_cycle_checking_at_init_extend_from_weighted_edge_list(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        node_c = dag.add_child(node_b, "c", {})
        with self.assertRaises(retworkx.DAGWouldCycle):
            dag.extend_from_weighted_edge_list(
                [(node_a, node_c, {}), (node_c, node_b, {})]
            )

    def test_extend_from_edge_list_nodes_exist(self):
        dag = retworkx.PyDiGraph()
        dag.add_nodes_from(list(range(4)))
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        dag.extend_from_edge_list(edge_list)
        self.assertEqual(len(dag), 4)
        self.assertEqual([None] * 5, dag.edges())
        self.assertEqual(3, dag.out_degree(0))
        self.assertEqual(0, dag.in_degree(0))
        self.assertEqual(1, dag.out_degree(1))
        self.assertEqual(1, dag.out_degree(2))
        self.assertEqual(2, dag.in_degree(3))

    def test_extend_from_weighted_edge_list(self):
        dag = retworkx.PyDAG()
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        dag.extend_from_weighted_edge_list(edge_list)
        self.assertEqual(len(dag), 4)
        self.assertEqual(["a", "b", "c", "d", "e"], dag.edges())
        self.assertEqual(3, dag.out_degree(0))
        self.assertEqual(0, dag.in_degree(0))
        self.assertEqual(1, dag.out_degree(1))
        self.assertEqual(1, dag.out_degree(2))
        self.assertEqual(2, dag.in_degree(3))

    def test_extend_from_weighted_edge_list_empty(self):
        dag = retworkx.PyDAG()
        dag.extend_from_weighted_edge_list([])
        self.assertEqual(0, len(dag))

    def test_cycle_checking_at_init_nodes_extend_from_edge_list(self):
        dag = retworkx.PyDAG(True)
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        node_c = dag.add_child(node_b, "c", {})
        with self.assertRaises(retworkx.DAGWouldCycle):
            dag.extend_from_edge_list([(node_a, node_c), (node_c, node_b)])

    def test_extend_from_weighted_edge_list_nodes_exist(self):
        dag = retworkx.PyDiGraph()
        dag.add_nodes_from(list(range(4)))
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        dag.extend_from_weighted_edge_list(edge_list)
        self.assertEqual(len(dag), 4)
        self.assertEqual(["a", "b", "c", "d", "e"], dag.edges())
        self.assertEqual(3, dag.out_degree(0))
        self.assertEqual(0, dag.in_degree(0))
        self.assertEqual(1, dag.out_degree(1))
        self.assertEqual(1, dag.out_degree(2))
        self.assertEqual(2, dag.in_degree(3))

    def test_insert_node_on_in_edges(self):
        graph = retworkx.PyDiGraph()
        in_node = graph.add_node("qr[0]")
        out_node = graph.add_child(in_node, "qr[0]", "qr[0]")
        h_gate = graph.add_node("h")
        graph.insert_node_on_in_edges(h_gate, out_node)
        self.assertEqual(
            [(in_node, h_gate, "qr[0]"), (h_gate, out_node, "qr[0]")],
            graph.weighted_edge_list(),
        )

    def test_insert_node_on_in_edges_multiple(self):
        graph = retworkx.PyDiGraph()
        in_node_0 = graph.add_node("qr[0]")
        out_node_0 = graph.add_child(in_node_0, "qr[0]", "qr[0]")
        in_node_1 = graph.add_node("qr[1]")
        out_node_1 = graph.add_child(in_node_1, "qr[1]", "qr[1]")
        cx_gate = graph.add_node("cx")
        graph.insert_node_on_in_edges_multiple(
            cx_gate, [out_node_0, out_node_1]
        )
        self.assertEqual(
            {
                (in_node_0, cx_gate, "qr[0]"),
                (cx_gate, out_node_0, "qr[0]"),
                (in_node_1, cx_gate, "qr[1]"),
                (cx_gate, out_node_1, "qr[1]"),
            },
            set(graph.weighted_edge_list()),
        )

    def test_insert_node_on_in_edges_double(self):
        graph = retworkx.PyDiGraph()
        in_node = graph.add_node("qr[0]")
        out_node = graph.add_child(in_node, "qr[0]", "qr[0]")
        h_gate = graph.add_node("h")
        z_gate = graph.add_node("z")
        graph.insert_node_on_in_edges(h_gate, out_node)
        graph.insert_node_on_in_edges(z_gate, out_node)
        self.assertEqual(
            {
                (in_node, h_gate, "qr[0]"),
                (h_gate, z_gate, "qr[0]"),
                (z_gate, out_node, "qr[0]"),
            },
            set(graph.weighted_edge_list()),
        )

    def test_insert_node_on_in_edges_multiple_double(self):
        graph = retworkx.PyDiGraph()
        in_node_0 = graph.add_node("qr[0]")
        out_node_0 = graph.add_child(in_node_0, "qr[0]", "qr[0]")
        in_node_1 = graph.add_node("qr[1]")
        out_node_1 = graph.add_child(in_node_1, "qr[1]", "qr[1]")
        cx_gate = graph.add_node("cx")
        cz_gate = graph.add_node("cz")
        graph.insert_node_on_in_edges_multiple(
            cx_gate, [out_node_0, out_node_1]
        )
        graph.insert_node_on_in_edges_multiple(
            cz_gate, [out_node_0, out_node_1]
        )
        self.assertEqual(
            {
                (in_node_0, cx_gate, "qr[0]"),
                (cx_gate, cz_gate, "qr[0]"),
                (in_node_1, cx_gate, "qr[1]"),
                (cx_gate, cz_gate, "qr[1]"),
                (cz_gate, out_node_0, "qr[0]"),
                (cz_gate, out_node_1, "qr[1]"),
            },
            set(graph.weighted_edge_list()),
        )

    def test_insert_node_on_out_edges(self):
        graph = retworkx.PyDiGraph()
        in_node = graph.add_node("qr[0]")
        out_node = graph.add_child(in_node, "qr[0]", "qr[0]")
        h_gate = graph.add_node("h")
        graph.insert_node_on_out_edges(h_gate, in_node)
        self.assertEqual(
            {(in_node, h_gate, "qr[0]"), (h_gate, out_node, "qr[0]")},
            set(graph.weighted_edge_list()),
        )

    def test_insert_node_on_out_edges_multiple(self):
        graph = retworkx.PyDiGraph()
        in_node_0 = graph.add_node("qr[0]")
        out_node_0 = graph.add_child(in_node_0, "qr[0]", "qr[0]")
        in_node_1 = graph.add_node("qr[1]")
        out_node_1 = graph.add_child(in_node_1, "qr[1]", "qr[1]")
        cx_gate = graph.add_node("cx")
        graph.insert_node_on_out_edges_multiple(cx_gate, [in_node_0, in_node_1])
        self.assertEqual(
            {
                (in_node_0, cx_gate, "qr[0]"),
                (cx_gate, out_node_0, "qr[0]"),
                (in_node_1, cx_gate, "qr[1]"),
                (cx_gate, out_node_1, "qr[1]"),
            },
            set(graph.weighted_edge_list()),
        )

    def test_insert_node_on_out_edges_double(self):
        graph = retworkx.PyDiGraph()
        in_node = graph.add_node("qr[0]")
        out_node = graph.add_child(in_node, "qr[0]", "qr[0]")
        h_gate = graph.add_node("h")
        z_gate = graph.add_node("z")
        graph.insert_node_on_out_edges(h_gate, in_node)
        graph.insert_node_on_out_edges(z_gate, in_node)
        self.assertEqual(
            {
                (in_node, z_gate, "qr[0]"),
                (z_gate, h_gate, "qr[0]"),
                (h_gate, out_node, "qr[0]"),
            },
            set(graph.weighted_edge_list()),
        )

    def test_insert_node_on_out_edges_multiple_double(self):
        graph = retworkx.PyDiGraph()
        in_node_0 = graph.add_node("qr[0]")
        out_node_0 = graph.add_child(in_node_0, "qr[0]", "qr[0]")
        in_node_1 = graph.add_node("qr[1]")
        out_node_1 = graph.add_child(in_node_1, "qr[1]", "qr[1]")
        cx_gate = graph.add_node("cx")
        cz_gate = graph.add_node("cz")
        graph.insert_node_on_out_edges_multiple(cx_gate, [in_node_0, in_node_1])
        graph.insert_node_on_out_edges_multiple(cz_gate, [in_node_0, in_node_1])
        self.assertEqual(
            {
                (in_node_0, cz_gate, "qr[0]"),
                (cz_gate, cx_gate, "qr[0]"),
                (in_node_1, cz_gate, "qr[1]"),
                (cz_gate, cx_gate, "qr[1]"),
                (cx_gate, out_node_0, "qr[0]"),
                (cx_gate, out_node_1, "qr[1]"),
            },
            set(graph.weighted_edge_list()),
        )

    def test_insert_node_on_in_edges_no_edges(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node(None)
        node_b = graph.add_node(None)
        graph.insert_node_on_in_edges(node_b, node_a)
        self.assertEqual([], graph.edge_list())

    def test_insert_node_on_in_edges_multiple_no_edges(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node(None)
        node_b = graph.add_node(None)
        graph.insert_node_on_in_edges_multiple(node_b, [node_a])
        self.assertEqual([], graph.edge_list())

    def test_insert_node_on_out_edges_no_edges(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node(None)
        node_b = graph.add_node(None)
        graph.insert_node_on_out_edges(node_b, node_a)
        self.assertEqual([], graph.edge_list())

    def test_insert_node_on_out_edges_multiple_no_edges(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node(None)
        node_b = graph.add_node(None)
        graph.insert_node_on_out_edges_multiple(node_b, [node_a])
        self.assertEqual([], graph.edge_list())

    def test_edge_index_map(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node(0)
        node_b = graph.add_node(1)
        node_c = graph.add_child(node_a, "c", "edge a")
        node_d = graph.add_parent(node_b, "d", "edge_b")
        graph.add_edge(node_c, node_d, "edge c")
        self.assertEqual(
            {
                0: (node_a, node_c, "edge a"),
                1: (node_d, node_b, "edge_b"),
                2: (node_c, node_d, "edge c"),
            },
            graph.edge_index_map(),
        )

    def test_edge_index_map_empty(self):
        graph = retworkx.PyDiGraph()
        self.assertEqual({}, graph.edge_index_map())


class TestEdgesMultigraphFalse(unittest.TestCase):
    def test_multigraph_attr(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        self.assertFalse(graph.multigraph)

    def test_get_edge_data(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        res = graph.get_edge_data(node_a, node_b)
        self.assertEqual("Edgy", res)

    def test_get_all_edge_data(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        graph.add_edge(node_a, node_b, "b")
        res = graph.get_all_edge_data(node_a, node_b)
        self.assertIn("b", res)
        self.assertNotIn("Edgy", res)

    def test_no_edge(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(
            retworkx.NoEdgeBetweenNodes, graph.get_edge_data, node_a, node_b
        )

    def test_no_edge_get_all_edge_data(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(
            retworkx.NoEdgeBetweenNodes, graph.get_all_edge_data, node_a, node_b
        )

    def test_has_edge(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {})
        self.assertTrue(graph.has_edge(node_a, node_b))

    def test_has_edge_no_edge(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertFalse(graph.has_edge(node_a, node_b))

    def test_edges(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Super edgy")
        self.assertEqual(["Edgy", "Super edgy"], graph.edges())

    def test_edges_empty(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        graph.add_node("a")
        self.assertEqual([], graph.edges())

    def test_add_duplicates(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "a")
        graph.add_edge(node_a, node_b, "b")
        self.assertEqual(["b"], graph.edges())

    def test_remove_no_edge(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(
            retworkx.NoEdgeBetweenNodes, graph.remove_edge, node_a, node_b
        )

    def test_remove_edge_single(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edgy")
        graph.remove_edge(node_a, node_b)
        self.assertEqual([], graph.edges())

    def test_remove_multiple(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edgy")
        graph.add_edge(node_a, node_b, "super_edgy")
        graph.remove_edge_from_index(0)
        self.assertEqual([], graph.edges())

    def test_remove_edge_from_index(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edgy")
        graph.remove_edge_from_index(0)
        self.assertEqual([], graph.edges())

    def test_remove_edge_no_edge(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        graph.add_node("a")
        graph.remove_edge_from_index(0)
        self.assertEqual([], graph.edges())

    def test_add_edge_from_empty(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        res = graph.add_edges_from([])
        self.assertEqual([], res)

    def test_add_edge_from_empty_no_data(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        res = graph.add_edges_from_no_data([])
        self.assertEqual([], res)

    def test_add_edges_from_parallel_edges(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        graph.add_nodes_from([0, 1])
        res = graph.add_edges_from([(0, 1, False), (0, 1, True)])
        self.assertEqual([0, 0], res)
        self.assertEqual([True], graph.edges())

    def test_add_edges_from_no_data_parallel_edges(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        graph.add_nodes_from([0, 1])
        res = graph.add_edges_from_no_data([(0, 1), (0, 1)])
        self.assertEqual([0, 0], res)
        self.assertEqual([None], graph.edges())

    def test_extend_from_weighted_edge_list_empty(self):
        graph = retworkx.PyDiGraph()
        graph.extend_from_weighted_edge_list([])
        self.assertEqual(0, len(graph))

    def test_extend_from_weighted_edge_list_nodes_exist(self):
        graph = retworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        graph.extend_from_weighted_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual(["a", "b", "c", "d", "e"], graph.edges())

    def test_extend_from_weighted_edge_list_edges_exist(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        graph.add_nodes_from(list(range(4)))
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
            (0, 1, "not_a"),
        ]
        graph.extend_from_weighted_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual(["not_a", "b", "c", "d", "e"], graph.edges())

    def test_extend_from_edge_list(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        graph.extend_from_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual([None] * 5, graph.edges())

    def test_extend_from_edge_list_empty(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        graph.extend_from_edge_list([])
        self.assertEqual(0, len(graph))

    def test_extend_from_edge_list_existing_edge(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        graph.add_nodes_from(list(range(4)))
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3), (0, 1)]
        graph.extend_from_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual([None] * 5, graph.edges())

    def test_extend_from_weighted_edge_list(self):
        graph = retworkx.PyDiGraph(multigraph=False)
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        graph.extend_from_weighted_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual(["a", "b", "c", "d", "e"], graph.edges())
