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


class TestNodes(unittest.TestCase):
    def test_nodes(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", "Edgy")
        res = dag.nodes()
        self.assertEqual(["a", "b"], res)
        self.assertEqual([0, 1], dag.node_indexes())

    def test_node_indices(self):
        graph = rustworkx.PyDiGraph()
        node_a = graph.add_node("a")
        graph.add_child(node_a, "b", "Edgy")
        self.assertEqual([0, 1], graph.node_indices())

    def test_no_nodes(self):
        dag = rustworkx.PyDAG()
        self.assertEqual([], dag.nodes())
        self.assertEqual([], dag.node_indexes())
        self.assertEqual([], dag.node_indices())

    def test_remove_node(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_node(node_b)
        res = dag.nodes()
        self.assertEqual(["a", "c"], res)
        self.assertEqual([0, 2], dag.node_indexes())

    def test_remove_node_invalid_index(self):
        graph = rustworkx.PyDAG()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.remove_node(76)
        res = graph.nodes()
        self.assertEqual(["a", "b", "c"], res)
        self.assertEqual([0, 1, 2], graph.node_indexes())

    def test_remove_nodes_from(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        node_c = dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_nodes_from([node_b, node_c])
        res = dag.nodes()
        self.assertEqual(["a"], res)
        self.assertEqual([0], dag.node_indexes())

    def test_remove_nodes_from_with_invalid_index(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        node_c = dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_nodes_from([node_b, node_c, 76])
        res = dag.nodes()
        self.assertEqual(["a"], res)
        self.assertEqual([0], dag.node_indexes())

    def test_remove_nodes_retain_edges_single_edge(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        node_c = dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_node_retain_edges(node_b)
        res = dag.nodes()
        self.assertEqual(["a", "c"], res)
        self.assertEqual([0, 2], dag.node_indexes())
        self.assertTrue(dag.has_edge(node_a, node_c))
        self.assertEqual(dag.get_all_edge_data(node_a, node_c), ["Edgy"])

    def test_remove_nodes_retain_edges_single_edge_outgoing_weight(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        node_c = dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_node_retain_edges(node_b, use_outgoing=True)
        res = dag.nodes()
        self.assertEqual(["a", "c"], res)
        self.assertEqual([0, 2], dag.node_indexes())
        self.assertTrue(dag.has_edge(node_a, node_c))
        self.assertEqual(dag.get_all_edge_data(node_a, node_c), ["Edgy_mk2"])

    def test_remove_nodes_retain_edges_multiple_in_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_d = dag.add_node("d")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_edge(node_d, node_b, "Multiple in edgy")
        node_c = dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_node_retain_edges(node_b)
        res = dag.nodes()
        self.assertEqual(["a", "d", "c"], res)
        self.assertEqual([0, 1, 3], dag.node_indexes())
        self.assertTrue(dag.has_edge(node_a, node_c))
        self.assertTrue(dag.has_edge(node_d, node_c))

    def test_remove_nodes_retain_edges_multiple_out_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_d = dag.add_node("d")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_edge(node_b, node_d, "Multiple out edgy")
        node_c = dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_node_retain_edges(node_b)
        res = dag.nodes()
        self.assertEqual(["a", "d", "c"], res)
        self.assertEqual([0, 1, 3], dag.node_indexes())
        self.assertTrue(dag.has_edge(node_a, node_c))
        self.assertTrue(dag.has_edge(node_a, node_d))

    def test_remove_nodes_retain_edges_multiple_in_and_out_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_d = dag.add_node("d")
        node_e = dag.add_node("e")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_edge(node_b, node_d, "Multiple out edgy")
        dag.add_edge(node_e, node_b, "multiple in edgy")
        node_c = dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_node_retain_edges(node_b)
        res = dag.nodes()
        self.assertEqual(["a", "d", "e", "c"], res)
        self.assertEqual([0, 1, 2, 4], dag.node_indexes())
        self.assertTrue(dag.has_edge(node_a, node_c))
        self.assertTrue(dag.has_edge(node_a, node_d))
        self.assertTrue(dag.has_edge(node_e, node_c))
        self.assertTrue(dag.has_edge(node_e, node_d))

    def test_remove_node_retain_edges_with_condition(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_d = dag.add_node("d")
        node_e = dag.add_node("e")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_edge(node_b, node_d, "Multiple out edgy")
        dag.add_edge(node_e, node_b, "multiple in edgy")
        node_c = dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_node_retain_edges(node_b, condition=lambda a, b: a == "multiple in edgy")
        res = dag.nodes()
        self.assertEqual(["a", "d", "e", "c"], res)
        self.assertEqual([0, 1, 2, 4], dag.node_indexes())
        self.assertFalse(dag.has_edge(node_a, node_c))
        self.assertFalse(dag.has_edge(node_a, node_d))
        self.assertTrue(dag.has_edge(node_e, node_c))
        self.assertTrue(dag.has_edge(node_e, node_d))

    def test_remove_nodes_retain_edges_with_invalid_index(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_child(node_b, "c", "Edgy_mk2")
        dag.remove_node_retain_edges(76)
        res = dag.nodes()
        self.assertEqual(["a", "b", "c"], res)
        self.assertEqual([0, 1, 2], dag.node_indexes())

    def test_remove_nodes_retain_edges_by_id_singles(self):
        dag = rustworkx.PyDAG()
        weights = [object(), object()]
        before_nodes = [dag.add_node(i) for i, _ in enumerate(weights)]
        middle = dag.add_node(10)
        after_nodes = [dag.add_node(20 + i) for i, _ in enumerate(weights)]
        for before, after, weight in zip(before_nodes, after_nodes, weights):
            dag.add_edge(before, middle, weight)
            dag.add_edge(middle, after, weight)
        dag.remove_node_retain_edges_by_id(middle)
        self.assertEqual(set(dag.node_indices()), set(before_nodes) | set(after_nodes))
        expected_edges = set(zip(before_nodes, after_nodes, weights))
        self.assertEqual(set(dag.weighted_edge_list()), expected_edges)

    def test_remove_nodes_retain_edges_by_id_parallel(self):
        dag = rustworkx.PyDAG()
        nodes = [dag.add_node(i) for i in range(3)]
        weights = [object(), object(), object()]
        for weight in weights:
            dag.add_edge(nodes[0], nodes[1], weight)
            dag.add_edge(nodes[1], nodes[2], weight)
        # The middle node has three precessor edges and three successor edges, where each set has
        # one edge each of three weights. Edges should be paired up in bijection during the removal.
        dag.remove_node_retain_edges_by_id(nodes[1])
        self.assertEqual(set(dag.node_indices()), {nodes[0], nodes[2]})
        expected_edges = {(nodes[0], nodes[2], weight) for weight in weights}
        self.assertEqual(set(dag.weighted_edge_list()), expected_edges)

    def test_remove_nodes_retain_edges_by_id_broadcast(self):
        dag = rustworkx.PyDAG()
        nodes = {a: dag.add_node(a) for a in "abcdefghijklmn"}
        mid = dag.add_node("middle")
        weights = [object(), object(), object(), object(), object()]
        expected_edges = set()

        # 2:1 broadcast.
        dag.add_edge(nodes["a"], mid, weights[0])
        dag.add_edge(nodes["b"], mid, weights[0])
        dag.add_edge(mid, nodes["c"], weights[0])
        expected_edges |= {
            (nodes["a"], nodes["c"], weights[0]),
            (nodes["b"], nodes["c"], weights[0]),
        }

        # 1:2 broadcast
        dag.add_edge(nodes["d"], mid, weights[1])
        dag.add_edge(mid, nodes["e"], weights[1])
        dag.add_edge(mid, nodes["f"], weights[1])
        expected_edges |= {
            (nodes["d"], nodes["e"], weights[1]),
            (nodes["d"], nodes["f"], weights[1]),
        }

        # 2:2 broadacst
        dag.add_edge(nodes["g"], mid, weights[2])
        dag.add_edge(nodes["h"], mid, weights[2])
        dag.add_edge(mid, nodes["i"], weights[2])
        dag.add_edge(mid, nodes["j"], weights[2])
        expected_edges |= {
            (nodes["g"], nodes["i"], weights[2]),
            (nodes["g"], nodes["j"], weights[2]),
            (nodes["h"], nodes["i"], weights[2]),
            (nodes["h"], nodes["j"], weights[2]),
        }

        # 0:1 broadcast
        dag.add_edge(mid, nodes["k"], weights[3])

        # 1:0 broadcast
        dag.add_edge(nodes["l"], mid, weights[4])

        # Edge that doesn't go via the middle at all, but shares an id with another edge.  This
        # shouldn't be touched.
        dag.add_edge(nodes["m"], nodes["n"], weights[0])
        expected_edges |= {
            (nodes["m"], nodes["n"], weights[0]),
        }

        dag.remove_node_retain_edges_by_id(mid)

        self.assertEqual(set(dag.nodes()), set(nodes))
        self.assertEqual(set(dag.weighted_edge_list()), expected_edges)

    def test_remove_nodes_retain_edges_by_key_singles_id_map(self):
        dag = rustworkx.PyDAG()
        before = [dag.add_node(0), dag.add_node(1)]
        middle = dag.add_node(2)
        after = [dag.add_node(3), dag.add_node(4)]

        expected_edges = set()
        dag.add_edge(before[0], middle, (0,))
        dag.add_edge(middle, after[0], (0,))
        expected_edges.add((before[0], after[0], (0,)))
        dag.add_edge(before[1], middle, (1,))
        dag.add_edge(middle, after[1], (1,))
        expected_edges.add((before[1], after[1], (1,)))

        dag.remove_node_retain_edges_by_id(middle)

        self.assertEqual(set(dag.node_indices()), set(before) | set(after))
        self.assertEqual(set(dag.weighted_edge_list()), expected_edges)

    def test_remove_nodes_retain_edges_by_key_broadcast_mod_map(self):
        dag = rustworkx.PyDAG()
        nodes = {a: dag.add_node(a) for a in "abcdefghijklmn"}
        mid = dag.add_node("middle")
        expected_edges = {}

        # 2:1 broadcast.
        dag.add_edge(nodes["a"], mid, 10)
        dag.add_edge(nodes["b"], mid, 20)
        dag.add_edge(mid, nodes["c"], 30)
        # The edge data here is a list of allowed weight - the function doesn't prescribe which
        # exact weight will be used, just that it's from the incoming edges.
        allowed_weights = {10, 20}
        expected_edges[nodes["a"], nodes["c"]] = allowed_weights
        expected_edges[nodes["b"], nodes["c"]] = allowed_weights

        # 1:2 broadcast
        dag.add_edge(nodes["d"], mid, 11)
        dag.add_edge(mid, nodes["e"], 21)
        dag.add_edge(mid, nodes["f"], 31)
        allowed_weights = {11}
        expected_edges[nodes["d"], nodes["e"]] = allowed_weights
        expected_edges[nodes["d"], nodes["f"]] = allowed_weights

        # 2:2 broadacst
        dag.add_edge(nodes["g"], mid, 12)
        dag.add_edge(nodes["h"], mid, 22)
        dag.add_edge(mid, nodes["i"], 32)
        dag.add_edge(mid, nodes["j"], 42)
        allowed_weights = {12, 22}
        expected_edges[nodes["g"], nodes["i"]] = allowed_weights
        expected_edges[nodes["g"], nodes["j"]] = allowed_weights
        expected_edges[nodes["h"], nodes["i"]] = allowed_weights
        expected_edges[nodes["h"], nodes["j"]] = allowed_weights

        # 0:1 broadcast
        dag.add_edge(mid, nodes["k"], 13)

        # 1:0 broadcast
        dag.add_edge(nodes["l"], mid, 14)

        # Edge that doesn't go via the middle at all, but shares a key with another edge.  This
        # shouldn't be touched.
        dag.add_edge(nodes["m"], nodes["n"], 10)
        expected_edges[nodes["m"], nodes["n"]] = {10}

        dag.remove_node_retain_edges_by_key(mid, key=lambda weight: weight % 10)

        self.assertEqual(set(dag.nodes()), set(nodes))
        self.assertEqual(set(dag.edge_list()), set(expected_edges))
        for source, target, weight in dag.weighted_edge_list():
            self.assertIn(weight, expected_edges[source, target])

    def test_remove_nodes_retain_edges_by_key_use_outgoing(self):
        dag = rustworkx.PyDAG()
        before = dag.add_node(0)
        middle = dag.add_node(1)
        after = dag.add_node(2)
        dag.add_edge(before, middle, 0)
        dag.add_edge(middle, after, 2)
        dag.remove_node_retain_edges_by_key(
            middle, key=lambda weight: weight % 2, use_outgoing=True
        )
        self.assertEqual(set(dag.node_indices()), {before, after})
        self.assertEqual(set(dag.weighted_edge_list()), {(before, after, 2)})

    def test_topo_sort_empty(self):
        dag = rustworkx.PyDAG()
        self.assertEqual([], rustworkx.topological_sort(dag))

    def test_topo_sort(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(5):
            dag.add_child(node_a, i, None)
        dag.add_parent(3, "A parent", None)
        res = rustworkx.topological_sort(dag)
        self.assertEqual([6, 0, 5, 4, 3, 2, 1], res)

    def test_topo_sort_with_cycle(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        dag.add_edge(node_b, node_a, {})
        self.assertRaises(rustworkx.DAGHasCycle, rustworkx.topological_sort, dag)

    def test_topo_generations_empty(self):
        dag = rustworkx.PyDAG()
        self.assertEqual([], rustworkx.topological_generations(dag))

    def test_topo_generations(self):
        dag = rustworkx.PyDAG()
        dag.extend_from_edge_list(
            [
                (4, 2),
                (6, 5),
                (7, 3),
                (3, 1),
                (5, 2),
                (3, 0),
                (2, 1),
            ]
        )
        generations = [sorted(gen) for gen in rustworkx.topological_generations(dag)]
        expected = [[4, 6, 7], [3, 5], [0, 2], [1]]
        self.assertEqual(expected, generations)

    def test_topo_generations_with_cycle(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        dag.add_edge(node_b, node_a, {})
        self.assertRaises(rustworkx.DAGHasCycle, rustworkx.topological_generations, dag)

    def test_lexicographical_topo_sort(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(5):
            dag.add_child(node_a, i, None)
        dag.add_parent(3, "A parent", None)
        res = rustworkx.lexicographical_topological_sort(dag, lambda x: str(x))
        # Node values for nodes [6, 0, 5, 4, 3, 2, 1]
        expected = ["A parent", "a", 0, 1, 2, 3, 4]
        self.assertEqual(expected, res)

    def test_lexicographical_topo_sort_qiskit(self):
        dag = rustworkx.PyDAG()
        # inputs
        qr_0 = dag.add_node("qr[0]")
        qr_1 = dag.add_node("qr[1]")
        qr_2 = dag.add_node("qr[2]")
        cr_0 = dag.add_node("cr[0]")
        cr_1 = dag.add_node("cr[1]")

        # wires
        cx_1 = dag.add_node("cx_1")
        dag.add_edge(qr_0, cx_1, "qr[0]")
        dag.add_edge(qr_1, cx_1, "qr[1]")
        h_1 = dag.add_node("h_1")
        dag.add_edge(cx_1, h_1, "qr[0]")
        cx_2 = dag.add_node("cx_2")
        dag.add_edge(cx_1, cx_2, "qr[1]")
        dag.add_edge(qr_2, cx_2, "qr[2]")
        cx_3 = dag.add_node("cx_3")
        dag.add_edge(h_1, cx_3, "qr[0]")
        dag.add_edge(cx_2, cx_3, "qr[2]")
        h_2 = dag.add_node("h_2")
        dag.add_edge(cx_3, h_2, "qr[2]")

        # outputs
        qr_0_out = dag.add_node("qr[0]_out")
        dag.add_edge(cx_3, qr_0_out, "qr[0]")
        qr_1_out = dag.add_node("qr[1]_out")
        dag.add_edge(cx_2, qr_1_out, "qr[1]")
        qr_2_out = dag.add_node("qr[2]_out")
        dag.add_edge(h_2, qr_2_out, "qr[2]")
        cr_0_out = dag.add_node("cr[0]_out")
        dag.add_edge(cr_0, cr_0_out, "qr[2]")
        cr_1_out = dag.add_node("cr[1]_out")
        dag.add_edge(cr_1, cr_1_out, "cr[1]")

        res = list(rustworkx.lexicographical_topological_sort(dag, lambda x: str(x)))
        expected = [
            "cr[0]",
            "cr[0]_out",
            "cr[1]",
            "cr[1]_out",
            "qr[0]",
            "qr[1]",
            "cx_1",
            "h_1",
            "qr[2]",
            "cx_2",
            "cx_3",
            "h_2",
            "qr[0]_out",
            "qr[1]_out",
            "qr[2]_out",
        ]
        self.assertEqual(expected, res)

    def test_get_node_data(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        self.assertEqual("b", dag.get_node_data(node_b))

    def test_get_node_data_bad_index(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", "Edgy")
        self.assertRaises(IndexError, dag.get_node_data, 42)

    def test_pydag_length(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", "Edgy")
        self.assertEqual(2, len(dag))

    def test_pydag_length_empty(self):
        dag = rustworkx.PyDAG()
        self.assertEqual(0, len(dag))

    def test_pydigraph_num_nodes(self):
        graph = rustworkx.PyDiGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "An_edge")
        self.assertEqual(2, graph.num_nodes())

    def test_pydigraph_num_nodes_empty(self):
        graph = rustworkx.PyDiGraph()
        self.assertEqual(0, graph.num_nodes())

    def test_add_nodes_from(self):
        dag = rustworkx.PyDAG()
        nodes = list(range(100))
        res = dag.add_nodes_from(nodes)
        self.assertEqual(len(res), 100)
        self.assertEqual(res, nodes)

    def test_add_node_from_empty(self):
        dag = rustworkx.PyDAG()
        res = dag.add_nodes_from([])
        self.assertEqual(len(res), 0)

    def test_get_node_data_getitem(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        self.assertEqual("b", dag[node_b])

    def test_get_node_data_getitem_bad_index(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", "Edgy")
        with self.assertRaises(IndexError):
            dag[42]

    def test_set_node_data_setitem(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag[node_b] = "Oh so cool"
        self.assertEqual("Oh so cool", dag[node_b])

    def test_set_node_data_setitem_bad_index(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", "Edgy")
        with self.assertRaises(IndexError):
            dag[42] = "Oh so cool"

    def test_remove_node_delitem(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", "Edgy")
        dag.add_child(node_b, "c", "Edgy_mk2")
        del dag[node_b]
        res = dag.nodes()
        self.assertEqual(["a", "c"], res)
        self.assertEqual([0, 2], dag.node_indexes())

    def test_remove_node_delitem_invalid_index(self):
        graph = rustworkx.PyDAG()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        with self.assertRaises(IndexError):
            del graph[76]
        res = graph.nodes()
        self.assertEqual(["a", "b", "c"], res)
        self.assertEqual([0, 1, 2], graph.node_indexes())

    def test_find_node_by_weight(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(10)))
        res = graph.find_node_by_weight(9)
        self.assertEqual(res, 9)

    def test_find_node_by_weight_no_match(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(10)))
        res = graph.find_node_by_weight(42)
        self.assertEqual(res, None)

    def test_find_node_by_weight_multiple_matches(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(["a", "a"])
        res = graph.find_node_by_weight("a")
        self.assertEqual(res, 0)

    def test_merge_nodes(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(["a", "a", "b", "c"])
        graph.add_edge(0, 2, "edge0")
        graph.add_edge(3, 0, "edge1")
        graph.merge_nodes(0, 1)
        self.assertEqual(graph.node_indexes(), [1, 2, 3])
        self.assertEqual([(3, 1, "edge1"), (1, 2, "edge0")], graph.weighted_edge_list())

    def test_merge_nodes_no_match(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(["a", "a", "b", "c"])
        graph.add_edge(0, 2, "edge0")
        graph.add_edge(3, 0, "edge1")
        graph.merge_nodes(0, 2)
        self.assertEqual(graph.node_indexes(), [0, 1, 2, 3])
        self.assertEqual([(0, 2, "edge0"), (3, 0, "edge1")], graph.weighted_edge_list())

    def test_merge_nodes_invalid_node_first_index(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(["a", "b"])
        with self.assertRaises(IndexError):
            graph.merge_nodes(2, 0)

    def test_merge_nodes_invalid_node_second_index(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(["a", "b"])
        with self.assertRaises(IndexError):
            graph.merge_nodes(0, 3)
