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


class TestEdges(unittest.TestCase):
    def test_get_edge_data(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        res = graph.get_edge_data(node_a, node_b)
        self.assertEqual("Edgy", res)

    def test_get_all_edge_data(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        graph.add_edge(node_a, node_b, "b")
        res = graph.get_all_edge_data(node_a, node_b)
        self.assertIn("b", res)
        self.assertIn("Edgy", res)

    def test_no_edge(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(rustworkx.NoEdgeBetweenNodes, graph.get_edge_data, node_a, node_b)

    def test_num_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_node(1)
        graph.add_node(42)
        graph.add_node(146)
        graph.add_edges_from_no_data([(0, 1), (1, 2)])
        self.assertEqual(2, graph.num_edges())

    def test_num_edges_no_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_node(1)
        graph.add_node(42)
        self.assertEqual(0, graph.num_edges())

    def test_update_edge(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "not edgy")
        graph.update_edge(node_a, node_b, "Edgy")
        self.assertEqual([(0, 1, "Edgy")], graph.weighted_edge_list())

    def test_update_edge_no_edge(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(rustworkx.NoEdgeBetweenNodes, graph.update_edge, node_a, node_b, None)

    def test_update_edge_by_index(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        edge_index = graph.add_edge(node_a, node_b, "not edgy")
        graph.update_edge_by_index(edge_index, "Edgy")
        self.assertEqual([(0, 1, "Edgy")], graph.weighted_edge_list())

    def test_update_edge_invalid_index(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        self.assertRaises(IndexError, graph.update_edge_by_index, 0, None)

    def test_update_edge_parallel_edges(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "not edgy")
        edge_index = graph.add_edge(node_a, node_b, "not edgy")
        graph.update_edge_by_index(edge_index, "Edgy")
        self.assertEqual(
            [(0, 1, "not edgy"), (0, 1, "Edgy")],
            list(graph.weighted_edge_list()),
        )

    def test_no_edge_get_all_edge_data(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(rustworkx.NoEdgeBetweenNodes, graph.get_all_edge_data, node_a, node_b)

    def test_has_edge(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {})
        self.assertTrue(graph.has_edge(node_a, node_b))
        self.assertTrue(graph.has_edge(node_b, node_a))

    def test_has_edge_no_edge(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertFalse(graph.has_edge(node_a, node_b))

    def test_edges(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Super edgy")
        self.assertEqual(["Edgy", "Super edgy"], graph.edges())

    def test_edges_empty(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        self.assertEqual([], graph.edges())

    def test_edge_indices(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Super edgy")
        self.assertEqual([0, 1], graph.edge_indices())

    def test_get_edge_indices_empty(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        self.assertEqual([], graph.edge_indices())

    def test_add_duplicates(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("a")
        graph.add_edge(node_a, node_b, "a")
        graph.add_edge(node_a, node_b, "b")
        self.assertEqual(["a", "b"], graph.edges())

    def test_remove_no_edge(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(rustworkx.NoEdgeBetweenNodes, graph.remove_edge, node_a, node_b)

    def test_remove_edge_single(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edgy")
        graph.remove_edge(node_a, node_b)
        self.assertEqual([], graph.edges())

    def test_remove_multiple(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edgy")
        graph.add_edge(node_a, node_b, "super_edgy")
        graph.remove_edge_from_index(0)
        self.assertEqual(["super_edgy"], graph.edges())

    def test_remove_edge_from_index(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edgy")
        graph.remove_edge_from_index(0)
        self.assertEqual([], graph.edges())

    def test_remove_edge_no_edge(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.remove_edge_from_index(0)
        self.assertEqual([], graph.edges())

    def test_remove_edges_from(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_b, "edgy")
        graph.add_edge(node_a, node_c, "super_edgy")
        graph.remove_edges_from([(node_a, node_b), (node_a, node_c)])
        self.assertEqual([], graph.edges())

    def test_remove_edges_from_invalid(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_b, "edgy")
        graph.add_edge(node_a, node_c, "super_edgy")
        with self.assertRaises(rustworkx.NoEdgeBetweenNodes):
            graph.remove_edges_from([(node_b, node_c), (node_a, node_c)])

    def test_degree(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Super edgy")
        self.assertEqual(2, graph.degree(node_b))

    def test_degree_with_self_loops(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list([(0, 0), (0, 1), (0, 0)])
        self.assertEqual(5, graph.degree(0))

    def test_add_edge_from(self):
        graph = rustworkx.PyGraph()
        nodes = list(range(4))
        graph.add_nodes_from(nodes)
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        res = graph.add_edges_from(edge_list)
        self.assertEqual(len(res), 5)
        self.assertEqual(["a", "b", "c", "d", "e"], graph.edges())
        self.assertEqual(3, graph.degree(0))
        self.assertEqual(2, graph.degree(1))
        self.assertEqual(3, graph.degree(2))
        self.assertEqual(2, graph.degree(3))

    def test_add_edge_from_empty(self):
        graph = rustworkx.PyGraph()
        res = graph.add_edges_from([])
        self.assertEqual([], res)

    def test_add_edge_from_no_data(self):
        graph = rustworkx.PyGraph()
        nodes = list(range(4))
        graph.add_nodes_from(nodes)
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        res = graph.add_edges_from_no_data(edge_list)
        self.assertEqual(len(res), 5)
        self.assertEqual([None, None, None, None, None], graph.edges())
        self.assertEqual(3, graph.degree(0))
        self.assertEqual(2, graph.degree(1))
        self.assertEqual(3, graph.degree(2))
        self.assertEqual(2, graph.degree(3))

    def test_add_edge_from_empty_no_data(self):
        graph = rustworkx.PyGraph()
        res = graph.add_edges_from_no_data([])
        self.assertEqual([], res)

    def test_extend_from_weighted_edge_list_empty(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list([])
        self.assertEqual(0, len(graph))

    def test_extend_from_weighted_edge_list_nodes_exist(self):
        graph = rustworkx.PyGraph()
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
        graph = rustworkx.PyGraph()
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
        self.assertEqual(["a", "b", "c", "d", "e", "not_a"], graph.edges())

    def test_edge_list(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(4)))
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        graph.add_edges_from(edge_list)
        self.assertEqual([(x[0], x[1]) for x in edge_list], graph.edge_list())

    def test_edge_list_empty(self):
        graph = rustworkx.PyGraph()
        self.assertEqual([], graph.edge_list())

    def test_weighted_edge_list(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(4)))
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        graph.add_edges_from(edge_list)
        self.assertEqual(edge_list, graph.weighted_edge_list())

    def test_weighted_edge_list_empty(self):
        graph = rustworkx.PyGraph()
        self.assertEqual([], graph.weighted_edge_list())

    def test_edge_indices_from_endpoints(self):
        dag = rustworkx.PyGraph()
        dag.add_nodes_from(list(range(4)))
        edge_list = [
            (0, 1, None),
            (1, 2, None),
            (0, 2, None),
            (2, 3, None),
            (0, 3, None),
            (0, 2, None),
            (2, 0, None),
        ]
        dag.add_edges_from(edge_list)
        indices = dag.edge_indices_from_endpoints(0, 0)
        self.assertEqual(indices, [])
        indices = dag.edge_indices_from_endpoints(0, 1)
        self.assertEqual(indices, [0])
        indices = dag.edge_indices_from_endpoints(0, 2)
        self.assertEqual(indices, [2, 5, 6])

    def test_extend_from_edge_list(self):
        graph = rustworkx.PyGraph()
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        graph.extend_from_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual([None] * 5, graph.edges())
        self.assertEqual(3, graph.degree(0))
        self.assertEqual(2, graph.degree(1))
        self.assertEqual(3, graph.degree(2))
        self.assertEqual(2, graph.degree(3))

    def test_extend_from_edge_list_empty(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list([])
        self.assertEqual(0, len(graph))

    def test_extend_from_edge_list_nodes_exist(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(4)))
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        graph.extend_from_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual([None] * 5, graph.edges())
        self.assertEqual(3, graph.degree(0))
        self.assertEqual(2, graph.degree(1))
        self.assertEqual(3, graph.degree(2))
        self.assertEqual(2, graph.degree(3))

    def test_extend_from_edge_list_existing_edge(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(4)))
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3), (0, 1)]
        graph.extend_from_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual([None] * 6, graph.edges())

    def test_extend_from_weighted_edge_list(self):
        graph = rustworkx.PyGraph()
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        graph.extend_from_weighted_edge_list(edge_list)
        self.assertEqual(len(graph), 4)

    def test_add_edges_from_parallel_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from([0, 1])
        res = graph.add_edges_from([(0, 1, False), (1, 0, True)])
        self.assertEqual([0, 1], res)
        self.assertEqual([False, True], graph.edges())

    def test_add_edges_from_no_data_parallel_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from([0, 1])
        res = graph.add_edges_from_no_data([(0, 1), (1, 0)])
        self.assertEqual([0, 1], res)
        self.assertEqual([None, None], graph.edges())

    def test_multigraph_attr(self):
        graph = rustworkx.PyGraph()
        self.assertTrue(graph.multigraph)

    def test_has_parallel_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from([0, 1])
        graph.add_edge(0, 1, None)
        graph.add_edge(1, 0, 0)
        self.assertTrue(graph.has_parallel_edges())

    def test_has_parallel_edges_no_parallel_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from([0, 1])
        graph.add_edge(0, 1, None)
        self.assertFalse(graph.has_parallel_edges())

    def test_has_parallel_edges_empty(self):
        graph = rustworkx.PyGraph()
        self.assertFalse(graph.has_parallel_edges())

    def test_edge_index_map(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node(0)
        node_b = graph.add_node(1)
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        graph.add_edge(node_a, node_c, "edge a")
        graph.add_edge(node_b, node_d, "edge_b")
        graph.add_edge(node_c, node_d, "edge c")
        self.assertEqual(
            {
                0: (node_a, node_c, "edge a"),
                1: (node_b, node_d, "edge_b"),
                2: (node_c, node_d, "edge c"),
            },
            graph.edge_index_map(),
        )

    def test_incident_edges(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node(0)
        node_b = graph.add_node(1)
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        graph.add_edge(node_a, node_c, "edge a")
        graph.add_edge(node_b, node_d, "edge_b")
        graph.add_edge(node_c, node_d, "edge c")
        res = graph.incident_edges(node_d)
        self.assertEqual({1, 2}, set(res))

    def test_incident_edges_invalid_node(self):
        graph = rustworkx.PyGraph()
        res = graph.incident_edges(42)
        self.assertEqual([], res)

    def test_incident_edge_index_map(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node(0)
        node_b = graph.add_node(1)
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        graph.add_edge(node_a, node_c, "edge a")
        graph.add_edge(node_b, node_d, "edge_b")
        graph.add_edge(node_c, node_d, "edge c")
        res = graph.incident_edge_index_map(node_d)
        self.assertEqual({2: (3, 2, "edge c"), 1: (3, 1, "edge_b")}, res)

    def test_incident_edge_index_map_invalid_node(self):
        graph = rustworkx.PyGraph()
        res = graph.incident_edge_index_map(42)
        self.assertEqual({}, res)

    def test_single_neighbor_out_edges(self):
        g = rustworkx.PyGraph()
        node_a = g.add_node("a")
        node_b = g.add_node("b")
        g.add_edge(node_a, node_b, {"a": 1})
        node_c = g.add_node("c")
        g.add_edge(node_a, node_c, {"a": 2})
        res = g.out_edges(node_a)
        self.assertEqual([(node_a, node_c, {"a": 2}), (node_a, node_b, {"a": 1})], res)

    def test_neighbor_surrounded_in_out_edges(self):
        g = rustworkx.PyGraph()
        node_a = g.add_node("a")
        node_b = g.add_node("b")
        node_c = g.add_node("c")
        g.add_edge(node_a, node_b, {"a": 1})
        g.add_edge(node_b, node_c, {"a": 2})
        res = g.out_edges(node_b)
        self.assertEqual([(node_b, node_c, {"a": 2}), (node_b, node_a, {"a": 1})], res)
        res = g.in_edges(node_b)
        self.assertEqual([(node_c, node_b, {"a": 2}), (node_a, node_b, {"a": 1})], res)

    def test_edge_index_map_empty(self):
        graph = rustworkx.PyGraph()
        self.assertEqual({}, graph.edge_index_map())

    def test_get_edge_data_by_index(self):
        graph = rustworkx.PyGraph()
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        graph.extend_from_weighted_edge_list(edge_list)
        res = graph.get_edge_data_by_index(2)
        self.assertEqual("c", res)

    def test_get_edge_data_by_index_invalid_index(self):
        graph = rustworkx.PyGraph()
        with self.assertRaisesRegex(
            IndexError, "Provided edge index 2 is not present in the graph"
        ):
            graph.get_edge_data_by_index(2)

    def test_get_edge_endpoints_by_index(self):
        graph = rustworkx.PyGraph()
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        graph.extend_from_weighted_edge_list(edge_list)
        res = graph.get_edge_endpoints_by_index(2)
        self.assertEqual((0, 2), res)

    def test_get_edge_endpoints_by_index_invalid_index(self):
        graph = rustworkx.PyGraph()
        with self.assertRaisesRegex(
            IndexError, "Provided edge index 2 is not present in the graph"
        ):
            graph.get_edge_endpoints_by_index(2)


class TestEdgesMultigraphFalse(unittest.TestCase):
    def test_multigraph_attr(self):
        graph = rustworkx.PyGraph(multigraph=False)
        self.assertFalse(graph.multigraph)

    def test_has_parallel_edges(self):
        graph = rustworkx.PyGraph(multigraph=False)
        graph.add_nodes_from([0, 1])
        graph.add_edge(0, 1, None)
        graph.add_edge(1, 0, 0)
        self.assertFalse(graph.has_parallel_edges())

    def test_parallel_edges_not_in_edge_list(self):
        graph = rustworkx.PyGraph(multigraph=False)
        edge_list = [
            (8, 6),
            (6, 5),
            (6, 5),
            (4, 5),
            (5, 4),
            (4, 5),
            (3, 4),
            (4, 3),
            (3, 4),
            (2, 3),
            (0, 2),
            (2, 0),
            (0, 2),
            (2, 3),
        ]
        graph.extend_from_edge_list(edge_list)
        graph_edge_list = graph.edge_list()
        expected_edges = [(6, 8), (5, 6), (4, 5), (3, 4), (2, 3), (0, 2)]
        self.assertEqual(len(graph_edge_list), len(expected_edges))
        for edge in expected_edges:
            if edge not in graph_edge_list and (edge[1], edge[0]) not in graph_edge_list:
                self.fail(f"{edge} not found in graph edge list {graph_edge_list}")

    def test_get_edge_data(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        res = graph.get_edge_data(node_a, node_b)
        self.assertEqual("Edgy", res)

    def test_get_all_edge_data(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        graph.add_edge(node_a, node_b, "b")
        res = graph.get_all_edge_data(node_a, node_b)
        self.assertIn("b", res)
        self.assertNotIn("Edgy", res)

    def test_no_edge(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(rustworkx.NoEdgeBetweenNodes, graph.get_edge_data, node_a, node_b)

    def test_no_edge_get_all_edge_data(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(rustworkx.NoEdgeBetweenNodes, graph.get_all_edge_data, node_a, node_b)

    def test_has_edge(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {})
        self.assertTrue(graph.has_edge(node_a, node_b))
        self.assertTrue(graph.has_edge(node_b, node_a))

    def test_has_edge_no_edge(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertFalse(graph.has_edge(node_a, node_b))

    def test_edges(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Super edgy")
        self.assertEqual(["Edgy", "Super edgy"], graph.edges())

    def test_edges_empty(self):
        graph = rustworkx.PyGraph(False)
        graph.add_node("a")
        self.assertEqual([], graph.edges())

    def test_add_duplicates(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("a")
        graph.add_edge(node_a, node_b, "a")
        graph.add_edge(node_a, node_b, "b")
        self.assertEqual(["b"], graph.edges())

    def test_remove_no_edge(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        self.assertRaises(rustworkx.NoEdgeBetweenNodes, graph.remove_edge, node_a, node_b)

    def test_remove_edge_single(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edgy")
        graph.remove_edge(node_a, node_b)
        self.assertEqual([], graph.edges())

    def test_remove_multiple(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edgy")
        graph.add_edge(node_a, node_b, "super_edgy")
        graph.remove_edge_from_index(0)
        self.assertEqual([], graph.edges())

    def test_remove_edge_from_index(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "edgy")
        graph.remove_edge_from_index(0)
        self.assertEqual([], graph.edges())

    def test_remove_edge_no_edge(self):
        graph = rustworkx.PyGraph(False)
        graph.add_node("a")
        graph.remove_edge_from_index(0)
        self.assertEqual([], graph.edges())

    def test_degree(self):
        graph = rustworkx.PyGraph(False)
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, "Edgy")
        node_c = graph.add_node("c")
        graph.add_edge(node_b, node_c, "Super edgy")
        self.assertEqual(2, graph.degree(node_b))

    def test_add_edge_from(self):
        graph = rustworkx.PyGraph(False)
        nodes = list(range(4))
        graph.add_nodes_from(nodes)
        edge_list = [
            (0, 1, "a"),
            (1, 2, "b"),
            (0, 2, "c"),
            (2, 3, "d"),
            (0, 3, "e"),
        ]
        res = graph.add_edges_from(edge_list)
        self.assertEqual(len(res), 5)
        self.assertEqual(["a", "b", "c", "d", "e"], graph.edges())
        self.assertEqual(3, graph.degree(0))
        self.assertEqual(2, graph.degree(1))
        self.assertEqual(3, graph.degree(2))
        self.assertEqual(2, graph.degree(3))

    def test_add_edge_from_empty(self):
        graph = rustworkx.PyGraph(False)
        res = graph.add_edges_from([])
        self.assertEqual([], res)

    def test_add_edge_from_no_data(self):
        graph = rustworkx.PyGraph(False)
        nodes = list(range(4))
        graph.add_nodes_from(nodes)
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        res = graph.add_edges_from_no_data(edge_list)
        self.assertEqual(len(res), 5)
        self.assertEqual([None, None, None, None, None], graph.edges())
        self.assertEqual(3, graph.degree(0))
        self.assertEqual(2, graph.degree(1))
        self.assertEqual(3, graph.degree(2))
        self.assertEqual(2, graph.degree(3))

    def test_add_edge_from_empty_no_data(self):
        graph = rustworkx.PyGraph(False)
        res = graph.add_edges_from_no_data([])
        self.assertEqual([], res)

    def test_add_edges_from_parallel_edges(self):
        graph = rustworkx.PyGraph(False)
        graph.add_nodes_from([0, 1])
        res = graph.add_edges_from([(0, 1, False), (1, 0, True)])
        self.assertEqual([0, 0], res)
        self.assertEqual([True], graph.edges())

    def test_add_edges_from_no_data_parallel_edges(self):
        graph = rustworkx.PyGraph(False)
        graph.add_nodes_from([0, 1])
        res = graph.add_edges_from_no_data([(0, 1), (1, 0)])
        self.assertEqual([0, 0], res)
        self.assertEqual([None], graph.edges())

    def test_extend_from_weighted_edge_list_empty(self):
        graph = rustworkx.PyGraph()
        graph.extend_from_weighted_edge_list([])
        self.assertEqual(0, len(graph))

    def test_extend_from_weighted_edge_list_nodes_exist(self):
        graph = rustworkx.PyGraph()
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
        graph = rustworkx.PyGraph(False)
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
        graph = rustworkx.PyGraph(False)
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        graph.extend_from_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual([None] * 5, graph.edges())

    def test_extend_from_edge_list_empty(self):
        graph = rustworkx.PyGraph(False)
        graph.extend_from_edge_list([])
        self.assertEqual(0, len(graph))

    def test_extend_from_edge_list_nodes_exist(self):
        graph = rustworkx.PyGraph(False)
        graph.add_nodes_from(list(range(4)))
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        graph.extend_from_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual([None] * 5, graph.edges())
        self.assertEqual(3, graph.degree(0))
        self.assertEqual(2, graph.degree(1))
        self.assertEqual(3, graph.degree(2))
        self.assertEqual(2, graph.degree(3))

    def test_extend_from_edge_list_existing_edge(self):
        graph = rustworkx.PyGraph(False)
        graph.add_nodes_from(list(range(4)))
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3), (0, 1)]
        graph.extend_from_edge_list(edge_list)
        self.assertEqual(len(graph), 4)
        self.assertEqual([None] * 5, graph.edges())

    def test_extend_from_weighted_edge_list(self):
        graph = rustworkx.PyGraph(False)
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

    def test_add_edge_non_existent(self):
        g = rustworkx.PyGraph()
        with self.assertRaises(IndexError):
            g.add_edge(2, 3, None)

    def test_add_edges_from_non_existent(self):
        g = rustworkx.PyGraph()
        with self.assertRaises(IndexError):
            g.add_edges_from([(2, 3, 5)])

    def test_add_edges_from_no_data_non_existent(self):
        g = rustworkx.PyGraph()
        with self.assertRaises(IndexError):
            g.add_edges_from_no_data([(2, 3)])
