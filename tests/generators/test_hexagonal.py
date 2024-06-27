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
import networkx


class TestHexagonalLatticeGraph(unittest.TestCase):
    def test_directed_hexagonal_graph_2_2(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(2, 2)
        expected_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (0, 5),
            (2, 7),
            (4, 9),
            (6, 11),
            (8, 13),
            (10, 15),
        ]
        self.assertEqual(len(graph), 16)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_3_2(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(3, 2)
        expected_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (0, 7),
            (2, 9),
            (4, 11),
            (6, 13),
            (8, 15),
            (10, 17),
            (12, 19),
            (14, 21),
        ]
        self.assertEqual(len(graph), 22)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_2_4(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(2, 4)
        expected_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (23, 24),
            (24, 25),
            (25, 26),
            (26, 27),
            (0, 5),
            (2, 7),
            (4, 9),
            (6, 12),
            (8, 14),
            (10, 16),
            (11, 17),
            (13, 19),
            (15, 21),
            (18, 23),
            (20, 25),
            (22, 27),
        ]
        self.assertEqual(len(graph), 28)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_2_2_bidirectional(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(2, 2, bidirectional=True)
        expected_edges = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (5, 6),
            (6, 5),
            (6, 7),
            (7, 6),
            (7, 8),
            (8, 7),
            (8, 9),
            (9, 8),
            (9, 10),
            (10, 9),
            (11, 12),
            (12, 11),
            (12, 13),
            (13, 12),
            (13, 14),
            (14, 13),
            (14, 15),
            (15, 14),
            (0, 5),
            (5, 0),
            (2, 7),
            (7, 2),
            (4, 9),
            (9, 4),
            (6, 11),
            (11, 6),
            (8, 13),
            (13, 8),
            (10, 15),
            (15, 10),
        ]
        self.assertEqual(len(graph), 16)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_3_2_bidirectional(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(3, 2, bidirectional=True)
        expected_edges = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
            (5, 6),
            (6, 5),
            (7, 8),
            (8, 7),
            (8, 9),
            (9, 8),
            (9, 10),
            (10, 9),
            (10, 11),
            (11, 10),
            (11, 12),
            (12, 11),
            (12, 13),
            (13, 12),
            (13, 14),
            (14, 13),
            (15, 16),
            (16, 15),
            (16, 17),
            (17, 16),
            (17, 18),
            (18, 17),
            (18, 19),
            (19, 18),
            (19, 20),
            (20, 19),
            (20, 21),
            (21, 20),
            (0, 7),
            (7, 0),
            (2, 9),
            (9, 2),
            (4, 11),
            (11, 4),
            (6, 13),
            (13, 6),
            (8, 15),
            (15, 8),
            (10, 17),
            (17, 10),
            (12, 19),
            (19, 12),
            (14, 21),
            (21, 14),
        ]
        self.assertEqual(len(graph), 22)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_directed_hexagonal_graph_2_4_bidirectional(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(2, 4, bidirectional=True)
        expected_edges = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (5, 6),
            (6, 5),
            (6, 7),
            (7, 6),
            (7, 8),
            (8, 7),
            (8, 9),
            (9, 8),
            (9, 10),
            (10, 9),
            (11, 12),
            (12, 11),
            (12, 13),
            (13, 12),
            (13, 14),
            (14, 13),
            (14, 15),
            (15, 14),
            (15, 16),
            (16, 15),
            (17, 18),
            (18, 17),
            (18, 19),
            (19, 18),
            (19, 20),
            (20, 19),
            (20, 21),
            (21, 20),
            (21, 22),
            (22, 21),
            (23, 24),
            (24, 23),
            (24, 25),
            (25, 24),
            (25, 26),
            (26, 25),
            (26, 27),
            (27, 26),
            (0, 5),
            (5, 0),
            (2, 7),
            (7, 2),
            (4, 9),
            (9, 4),
            (6, 12),
            (12, 6),
            (8, 14),
            (14, 8),
            (10, 16),
            (16, 10),
            (11, 17),
            (17, 11),
            (13, 19),
            (19, 13),
            (15, 21),
            (21, 15),
            (18, 23),
            (23, 18),
            (20, 25),
            (25, 20),
            (22, 27),
            (27, 22),
        ]
        self.assertEqual(len(graph), 28)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_hexagonal_graph_2_2(self):
        graph = rustworkx.generators.hexagonal_lattice_graph(2, 2)
        expected_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (0, 5),
            (2, 7),
            (4, 9),
            (6, 11),
            (8, 13),
            (10, 15),
        ]
        self.assertEqual(len(graph), 16)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_hexagonal_graph_3_2(self):
        graph = rustworkx.generators.hexagonal_lattice_graph(3, 2)
        expected_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (0, 7),
            (2, 9),
            (4, 11),
            (6, 13),
            (8, 15),
            (10, 17),
            (12, 19),
            (14, 21),
        ]
        self.assertEqual(len(graph), 22)
        self.assertEqual(len(graph.edges()), 27)
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_hexagonal_graph_2_4(self):
        graph = rustworkx.generators.hexagonal_lattice_graph(2, 4)
        expected_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (23, 24),
            (24, 25),
            (25, 26),
            (26, 27),
            (0, 5),
            (2, 7),
            (4, 9),
            (6, 12),
            (8, 14),
            (10, 16),
            (11, 17),
            (13, 19),
            (15, 21),
            (18, 23),
            (20, 25),
            (22, 27),
        ]
        self.assertEqual(len(graph), 28)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(list(graph.edge_list()), expected_edges)

    def test_hexagonal_graph_0_0(self):
        graph = rustworkx.generators.hexagonal_lattice_graph(0, 0)
        self.assertEqual(len(graph), 0)
        self.assertEqual(len(graph.edges()), 0)

    def test_hexagonal_graph_periodic_subgraphs(self):
        """Check that hexagonal subgraphs of the lattice are isomorphic
        to C6 (idea copied from the networkx test suite)."""
        graph = rustworkx.generators.hexagonal_lattice_graph(2, 4, periodic=True)
        hexagons = [[0, 1, 2, 6, 5, 4], [2, 3, 0, 4, 7, 6], [5, 6, 7, 11, 10, 9]]
        C6 = rustworkx.generators.cycle_graph(6)
        for h in hexagons:
            self.assertTrue(rustworkx.is_isomorphic(graph.subgraph(h), C6))

        # For a periodic graph with two columns, subgraphs are not necessarily
        # isomorphic to C6 because of the boundary connections.
        graph2cols = rustworkx.generators.hexagonal_lattice_graph(2, 2, periodic=True)
        subGraph = graph2cols.subgraph(hexagons[0])
        self.assertFalse(rustworkx.is_isomorphic(subGraph, C6))
        self.assertEqual(len(subGraph.edges()), 7)

    def test_hexagonal_graph_periodic_degree(self):
        """In a periodic hexagonal lattice, all nodes must have
        degree 3 (idea copied from the networkx test suite)."""
        for nRows in range(2, 8):
            for nCols in range(2, 8, 2):
                with self.subTest(nRows=nRows, nCols=nCols):
                    graph = rustworkx.generators.hexagonal_lattice_graph(
                        nRows, nCols, periodic=True
                    )
                    for n in range(graph.num_nodes()):
                        self.assertEqual(graph.degree(n), 3)

    def test_hexagonal_graph_periodic_3_4(self):
        graph = rustworkx.generators.hexagonal_lattice_graph(3, 4, periodic=True)
        expected_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 6),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (17, 12),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 18),
            (0, 6),
            (2, 8),
            (4, 10),
            (7, 13),
            (9, 15),
            (11, 17),
            (12, 18),
            (14, 20),
            (16, 22),
            (19, 1),
            (21, 3),
            (23, 5),
        ]
        self.assertEqual(len(graph), 24)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(set(graph.edge_list()), set(expected_edges))

    def test_directed_hexagonal_graph_periodic_3_2(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(3, 2, periodic=True)
        expected_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 6),
            (0, 6),
            (2, 8),
            (4, 10),
            (7, 1),
            (9, 3),
            (11, 5),
        ]
        self.assertEqual(len(graph), 12)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(set(graph.edge_list()), set(expected_edges))

    def test_directed_hexagonal_graph_bidirectional_periodic_3_2(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(
            3, 2, bidirectional=True, periodic=True
        )
        expected_edges = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
            (5, 0),
            (0, 5),
            (6, 7),
            (7, 6),
            (7, 8),
            (8, 7),
            (8, 9),
            (9, 8),
            (9, 10),
            (10, 9),
            (10, 11),
            (11, 10),
            (11, 6),
            (6, 11),
            (0, 6),
            (6, 0),
            (2, 8),
            (8, 2),
            (4, 10),
            (10, 4),
            (7, 1),
            (1, 7),
            (9, 3),
            (3, 9),
            (11, 5),
            (5, 11),
        ]
        self.assertEqual(len(graph), 12)
        self.assertEqual(len(graph.edges()), len(expected_edges))
        self.assertEqual(set(graph.edge_list()), set(expected_edges))

    def test_hexagonal_graph_periodic_networkx_equivalent(self):
        """Networkx uses different logic to construct a periodic hexagonal
        lattice graph, so here we check that the results from rustworkx and
        networkx are isomorphic for a few cases."""
        for nRows in range(2, 8):
            for nCols in range(2, 8, 2):
                with self.subTest(nRows=nRows, nCols=nCols):
                    graph = rustworkx.generators.hexagonal_lattice_graph(
                        nRows, nCols, periodic=True
                    )

                    nx_graph = rustworkx.networkx_converter(
                        networkx.hexagonal_lattice_graph(nRows, nCols, periodic=True)
                    )

                    self.assertTrue(rustworkx.is_isomorphic(graph, nx_graph))

    def test_hexagonal_graph_periodic_odd_columns(self):
        with self.assertRaises(ValueError):
            rustworkx.generators.hexagonal_lattice_graph(4, 5, periodic=True)
