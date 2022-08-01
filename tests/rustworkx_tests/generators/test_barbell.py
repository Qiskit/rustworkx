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


class TestBarbellGraph(unittest.TestCase):
    def test_barbell_graph_count(self):
        graph = rustworkx.generators.barbell_graph(17, 3)
        self.assertEqual(len(graph), 37)
        self.assertEqual(len(graph.edges()), 276)

    def test_barbell_graph_edge(self):
        graph = rustworkx.generators.barbell_graph(4, 3)
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

    def test_barbell_graph_no_path_num(self):
        graph = rustworkx.generators.barbell_graph(4)
        mesh = rustworkx.generators.mesh_graph(4)
        mesh.compose(mesh.copy(), {3: (0, None)})
        self.assertEqual(set(graph.edge_list()), set(mesh.edge_list()))

    def test_barbell_graph_no_mesh_num(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.barbell_graph()
