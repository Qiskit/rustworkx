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


class TestBarbellGraph(unittest.TestCase):
    def test_barbell_graph_count(self):
        graph = retworkx.generators.barbell_graph(17, 3)
        self.assertEqual(len(graph), 37)
        self.assertEqual(len(graph.edges()), 276)

    def test_barbell_graph_weights_count(self):
        graph = retworkx.generators.barbell_graph(
            mesh_weights=list(range(17)), path_weights=list(range(17, 20))
        )
        self.assertEqual(len(graph), 37)
        self.assertEqual(
            list(range(17)) + list(range(17, 20)) + list(range(17)),
            graph.nodes(),
        )
        self.assertEqual(len(graph.edges()), 276)

    def test_barbell_graph_edge(self):
        graph = retworkx.generators.barbell_graph(4, 3)
        edge_list = graph.edge_list()
        expected_edge_list = set(
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (7, 9),
                (7, 10),
                (8, 9),
                (8, 10),
                (9, 10),
            ]
        )
        self.assertEqual(set(edge_list), set(expected_edge_list))

    def test_barbell_graph_weights_edge(self):
        graph = retworkx.generators.barbell_graph(
            mesh_weights=list(range(4)), path_weights=list(range(3))
        )
        weighted_edge_list = graph.weighted_edge_list()
        expected_weighted_edge_list = set(
            [
                (0, 1, None),
                (0, 2, None),
                (0, 3, None),
                (1, 2, None),
                (1, 3, None),
                (2, 3, None),
                (3, 4, None),
                (4, 5, None),
                (5, 6, None),
                (6, 7, None),
                (7, 8, None),
                (7, 9, None),
                (7, 10, None),
                (8, 9, None),
                (8, 10, None),
                (9, 10, None),
            ]
        )
        self.assertEqual(
            set(weighted_edge_list), set(expected_weighted_edge_list)
        )
        self.assertEqual(graph.nodes(), [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3])

    def test_barbell_graph_no_path_weights_or_num(self):
        graph = retworkx.generators.barbell_graph(mesh_weights=list(range(4)))
        mesh = retworkx.generators.mesh_graph(weights=list(range(4)))
        mesh.compose(mesh.copy(), {3: (0, None)})
        self.assertEqual(graph.nodes(), mesh.nodes())
        self.assertEqual(
            set(graph.weighted_edge_list()), set(mesh.weighted_edge_list())
        )

        graph = retworkx.generators.barbell_graph(4)
        mesh = retworkx.generators.mesh_graph(4)
        mesh.compose(mesh.copy(), {3: (0, None)})
        self.assertEqual(set(graph.edge_list()), set(mesh.edge_list()))

    def test_barbell_graph_no_mesh_weights_or_num(self):
        with self.assertRaises(IndexError):
            retworkx.generators.barbell_graph()
