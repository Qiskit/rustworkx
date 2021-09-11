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


class TestBiconnected(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyGraph()
        self.graph.extend_from_edge_list(
            [
                # back edges
                (0, 2),
                (0, 3),
                (1, 4),
                (4, 9),
                (5, 7),
                # tree edges
                (0, 1),
                (1, 2),
                (2, 3),
                (2, 4),
                (4, 5),
                (4, 8),
                (5, 6),
                (6, 7),
                (8, 9),
            ]
        )

        self.barbell_graph = retworkx.PyGraph()
        self.barbell_graph.extend_from_edge_list(
            [
                (0, 1),
                (0, 2),
                (1, 2),
                (3, 4),
                (3, 5),
                (4, 5),
                (2, 3),
            ]
        )
        return super().setUp()

    def test_null_graph(self):
        graph = retworkx.PyGraph()
        self.assertEqual(retworkx.articulation_points(graph), set())
        self.assertEqual(retworkx.biconnected_components(graph), [])

    def test_graph(self):
        components = [
            {8, 9, 4},
            {5, 6, 7},
            {4, 5},
            {0, 1, 2, 3, 4},
        ]
        self.assertEqual(
            retworkx.biconnected_components(self.graph), components
        )
        self.assertEqual(retworkx.articulation_points(self.graph), {4, 5})

    def test_barbell_graph(self):
        components = [
            {3, 4, 5},
            {2, 3},
            {0, 1, 2},
        ]
        self.assertEqual(
            retworkx.biconnected_components(self.barbell_graph), components
        )
        self.assertEqual(
            retworkx.articulation_points(self.barbell_graph), {2, 3}
        )

    def test_disconnected_graph(self):
        graph = retworkx.union(self.barbell_graph, self.barbell_graph)
        components = [
            {3, 4, 5},
            {2, 3},
            {0, 1, 2},
            {9, 10, 11},
            {8, 9},
            {8, 6, 7},
        ]
        self.assertEqual(retworkx.biconnected_components(graph), components)
        self.assertEqual(retworkx.articulation_points(graph), {2, 3, 8, 9})

    def test_biconnected_graph(self):
        graph = retworkx.PyGraph()
        graph.extend_from_edge_list(
            [
                (0, 1),
                (0, 2),
                (0, 5),
                (1, 5),
                (2, 3),
                (2, 4),
                (3, 4),
                (3, 5),
                (3, 6),
                (4, 5),
                (4, 6),
            ]
        )
        self.assertEqual(retworkx.articulation_points(graph), set())
        nodes = set(graph.node_indexes())
        self.assertEqual(retworkx.biconnected_components(graph), [nodes])
