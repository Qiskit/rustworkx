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


class TestCycleBasis(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
        self.graph.add_nodes_from(list(range(10)))
        self.graph.add_edges_from_no_data(
            [
                (0, 1),
                (0, 3),
                (0, 5),
                (0, 8),
                (1, 2),
                (1, 6),
                (2, 3),
                (3, 4),
                (4, 5),
                (6, 7),
                (7, 8),
                (8, 9),
            ]
        )

    def test_cycle_basis(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(list(range(6)))
        graph.add_edges_from_no_data([(0, 1), (0, 3), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)])
        res = sorted(sorted(c) for c in rustworkx.cycle_basis(graph, 0))
        self.assertEqual([[0, 1, 2, 3], [0, 3, 4, 5]], res)

    def test_cycle_basis_multiple_roots_same_cycles(self):
        res = sorted(sorted(x) for x in rustworkx.cycle_basis(self.graph, 0))
        self.assertEqual(res, [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5]])
        res = sorted(sorted(x) for x in rustworkx.cycle_basis(self.graph, 1))
        self.assertEqual(res, [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5]])
        res = sorted(sorted(x) for x in rustworkx.cycle_basis(self.graph, 9))
        self.assertEqual(res, [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5]])

    def test_cycle_basis_disconnected_graphs(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edges_from_no_data([(10, 11), (10, 12), (11, 12)])
        cycles = rustworkx.cycle_basis(self.graph, 9)
        res = sorted(sorted(x) for x in cycles[:-1]) + [sorted(cycles[-1])]
        self.assertEqual(res, [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5], [10, 11, 12]])

    def test_invalid_types(self):
        digraph = rustworkx.PyDiGraph()
        with self.assertRaises(TypeError):
            rustworkx.cycle_basis(digraph)

    def test_self_loop(self):
        self.graph.add_edge(1, 1, None)
        res = sorted(sorted(c) for c in rustworkx.cycle_basis(self.graph, 0))
        self.assertEqual([[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5], [1]], res)


class TestCycleBasisEdges(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
        self.graph.add_nodes_from(list(range(10)))
        self.graph.add_edges_from_no_data(
            [
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 5),
                (4, 6),
                (6, 7),
                (6, 8),
                (8, 9),
            ]
        )

    def test_cycle_basis_edges(self):
        graph = self.graph
        res = sorted(sorted(c) for c in rustworkx.cycle_basis_edges(graph, 0))
        self.assertEqual([[0, 1, 2], [4, 5, 6]], res)

    def test_cycle_basis_edges_multiple_roots_same_cycles(self):
        res = sorted(sorted(x) for x in rustworkx.cycle_basis_edges(self.graph, 0))
        self.assertEqual([[0, 1, 2], [4, 5, 6]], res)
        res = sorted(sorted(x) for x in rustworkx.cycle_basis_edges(self.graph, 5))
        self.assertEqual([[0, 1, 2], [4, 5, 6]], res)
        res = sorted(sorted(x) for x in rustworkx.cycle_basis_edges(self.graph, 7))
        self.assertEqual([[0, 1, 2], [4, 5, 6]], res)

    def test_cycle_basis_disconnected_graphs(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edges_from_no_data([(10, 11), (10, 12), (11, 12)])
        cycles = rustworkx.cycle_basis_edges(self.graph, 9)
        res = sorted(sorted(x) for x in cycles[:-1]) + [sorted(cycles[-1])]
        self.assertEqual(res, [[0, 1, 2], [4, 5, 6], [11, 12, 13]])

    def test_invalid_types(self):
        digraph = rustworkx.PyDiGraph()
        with self.assertRaises(TypeError):
            rustworkx.cycle_basis_edges(digraph)

    def test_self_loop(self):
        self.graph.add_edge(1, 1, None)
        res = sorted(sorted(c) for c in rustworkx.cycle_basis_edges(self.graph, 0))
        self.assertEqual([[0, 1, 2], [4, 5, 6], [11]], res)
