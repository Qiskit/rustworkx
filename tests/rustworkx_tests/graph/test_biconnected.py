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


def sorted_edges(edges):
    return set([tuple(sorted(edge)) for edge in edges])


class TestBiconnected(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.graph = rustworkx.PyGraph()
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

        self.barbell_graph = rustworkx.PyGraph()
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

    def test_null_graph(self):
        graph = rustworkx.PyGraph()
        self.assertEqual(rustworkx.articulation_points(graph), set())
        self.assertEqual(rustworkx.bridges(graph), set())
        self.assertEqual(rustworkx.biconnected_components(graph), {})

    def test_graph(self):
        components = {
            (4, 8): 0,
            (8, 9): 0,
            (9, 4): 0,
            (5, 6): 1,
            (6, 7): 1,
            (7, 5): 1,
            (4, 5): 2,
            (0, 1): 3,
            (1, 2): 3,
            (2, 3): 3,
            (2, 4): 3,
            (2, 0): 3,
            (3, 0): 3,
            (4, 1): 3,
        }
        self.assertEqual(rustworkx.biconnected_components(self.graph), components)
        self.assertEqual(rustworkx.articulation_points(self.graph), {4, 5})
        self.assertEqual(sorted_edges(rustworkx.bridges(self.graph)), {(4, 5)})

    def test_barbell_graph(self):
        components = {
            (0, 2): 2,
            (2, 1): 2,
            (1, 0): 2,
            (3, 5): 0,
            (5, 4): 0,
            (4, 3): 0,
            (2, 3): 1,
        }
        self.assertEqual(rustworkx.biconnected_components(self.barbell_graph), components)
        self.assertEqual(rustworkx.articulation_points(self.barbell_graph), {2, 3})
        self.assertEqual(sorted_edges(rustworkx.bridges(self.barbell_graph)), {(2, 3)})

    def test_disconnected_graph(self):
        graph = rustworkx.union(self.barbell_graph, self.barbell_graph)
        components = {
            # first copy
            (0, 2): 2,
            (2, 1): 2,
            (1, 0): 2,
            (3, 5): 0,
            (5, 4): 0,
            (4, 3): 0,
            (2, 3): 1,
            # second copy
            (6, 8): 5,
            (8, 7): 5,
            (7, 6): 5,
            (9, 11): 3,
            (11, 10): 3,
            (10, 9): 3,
            (8, 9): 4,
        }
        self.assertEqual(rustworkx.biconnected_components(graph), components)
        self.assertEqual(rustworkx.articulation_points(graph), {2, 3, 8, 9})
        self.assertEqual(sorted_edges(rustworkx.bridges(graph)), {(2, 3), (8, 9)})

    def test_biconnected_graph(self):
        graph = rustworkx.PyGraph()
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
        num_edges = graph.num_edges()
        self.assertEqual(rustworkx.articulation_points(graph), set())
        self.assertEqual(rustworkx.bridges(graph), set())
        bicomp = rustworkx.biconnected_components(graph)
        self.assertEqual(len(bicomp), num_edges)
        self.assertEqual(list(bicomp.values()), [0] * num_edges)
