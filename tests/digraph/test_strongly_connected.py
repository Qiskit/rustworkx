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
        G.add_child(node_b, 3, {})
        self.assertEqual(rustworkx.number_strongly_connected_components(G), 3)

    def test_number_strongly_connected(self):
        G = rustworkx.PyDiGraph()
        node_a = G.add_node(1)
        node_b = G.add_child(node_a, 2, {})
        G.add_edge(node_b, node_a, {})
        G.add_node(3)
        self.assertEqual(rustworkx.number_strongly_connected_components(G), 2)

    def test_strongly_connected_no_linear(self):
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

    def test_is_strongly_connected_false(self):
        graph = rustworkx.PyDiGraph()
        graph.extend_from_edge_list(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (2, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
            ]
        )
        self.assertFalse(rustworkx.is_strongly_connected(graph))

    def test_is_strongly_connected_true(self):
        graph = rustworkx.PyDiGraph()
        graph.extend_from_edge_list(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (2, 4),
                (4, 2),  # <- missing in the test_is_strongly_connected_false
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
            ]
        )
        self.assertTrue(rustworkx.is_strongly_connected(graph))

    def test_is_strongly_connected_null_graph(self):
        graph = rustworkx.PyDiGraph()
        with self.assertRaises(rustworkx.NullGraph):
            rustworkx.is_strongly_connected(graph)


class TestCondensation(unittest.TestCase):
    def setUp(self):
        # Set up the graph
        self.graph = rustworkx.PyDiGraph()
        self.node_a = self.graph.add_node("a")
        self.node_b = self.graph.add_node("b")
        self.node_c = self.graph.add_node("c")
        self.node_d = self.graph.add_node("d")
        self.node_e = self.graph.add_node("e")
        self.node_f = self.graph.add_node("f")
        self.node_g = self.graph.add_node("g")
        self.node_h = self.graph.add_node("h")

        # Add edges
        self.graph.add_edge(self.node_a, self.node_b, "a->b")
        self.graph.add_edge(self.node_b, self.node_c, "b->c")
        self.graph.add_edge(self.node_c, self.node_d, "c->d")
        self.graph.add_edge(self.node_d, self.node_a, "d->a")  # Cycle: a -> b -> c -> d -> a

        self.graph.add_edge(self.node_b, self.node_e, "b->e")

        self.graph.add_edge(self.node_e, self.node_f, "e->f")
        self.graph.add_edge(self.node_f, self.node_g, "f->g")
        self.graph.add_edge(self.node_g, self.node_h, "g->h")
        self.graph.add_edge(self.node_h, self.node_e, "h->e")  # Cycle: e -> f -> g -> h -> e

    def test_condensation(self):
        # Call the condensation function
        condensed_graph, node_map = rustworkx.condensation(self.graph)

        # Check the number of nodes (two cycles should be condensed into one node each)
        self.assertEqual(
            len(condensed_graph.node_indices()), 2
        )  # [SCC(a, b, c, d), SCC(e, f, g, h)]

        # Check the number of edges
        self.assertEqual(
            len(condensed_graph.edge_indices()), 1
        )  # Edge: [SCC(a, b, c, d)] -> [SCC(e, f, g, h)]

        # Check the contents of the condensed nodes
        nodes = list(condensed_graph.nodes())
        scc1 = nodes[0]
        scc2 = nodes[1]
        self.assertTrue(set(scc1) == {"a", "b", "c", "d"} or set(scc2) == {"a", "b", "c", "d"})
        self.assertTrue(set(scc1) == {"e", "f", "g", "h"} or set(scc2) == {"e", "f", "g", "h"})

        # Check the contents of the edge
        weight = condensed_graph.edges()[0]
        self.assertIn("b->e", weight)  # Ensure the correct edge remains in the condensed graph

    def test_condensation_with_sccs_argument(self):
        # Compute SCCs manually
        sccs = rustworkx.strongly_connected_components(self.graph)
        # Call condensation with explicit sccs argument
        condensed_graph = rustworkx.condensation(self.graph, sccs=sccs)
        node_map = condensed_graph.attrs["node_map"]

        # Check the number of nodes (should match SCC count)
        self.assertEqual(len(condensed_graph.node_indices()), len(sccs))

        # Check the number of edges
        self.assertEqual(len(condensed_graph.edge_indices()), 1)

        # Check the contents of the condensed nodes
        nodes = list(condensed_graph.nodes())
        scc_sets = [set(n) for n in nodes]
        self.assertIn(set(["a", "b", "c", "d"]), scc_sets)
        self.assertIn(set(["e", "f", "g", "h"]), scc_sets)

        # Check the contents of the edge
        weight = condensed_graph.edges()[0]
        self.assertIn("b->e", weight)
