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


class TestLollipopGraph(unittest.TestCase):
    def test_lollipop_graph_count(self):
        graph = rustworkx.generators.lollipop_graph(17, 3)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 139)

    def test_lollipop_graph_weights_count(self):
        graph = rustworkx.generators.lollipop_graph(
            mesh_weights=list(range(17)), path_weights=list(range(17, 20))
        )
        self.assertEqual(len(graph), 20)
        self.assertEqual(list(range(20)), graph.nodes())
        self.assertEqual(len(graph.edges()), 139)

    def test_lollipop_graph_edge(self):
        graph = rustworkx.generators.lollipop_graph(4, 3)
        edge_list = graph.edge_list()
        expected_edge_list = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
        ]
        self.assertEqual(edge_list, expected_edge_list)

    def test_lollipop_graph_weights_edge(self):
        graph = rustworkx.generators.lollipop_graph(
            mesh_weights=list(range(4)), path_weights=list(range(3))
        )
        weighted_edge_list = graph.weighted_edge_list()
        expected_weighted_edge_list = [
            (0, 1, None),
            (0, 2, None),
            (0, 3, None),
            (1, 2, None),
            (1, 3, None),
            (2, 3, None),
            (3, 4, None),
            (4, 5, None),
            (5, 6, None),
        ]
        self.assertEqual(weighted_edge_list, expected_weighted_edge_list)
        self.assertEqual(graph.nodes(), [0, 1, 2, 3, 0, 1, 2])

    def test_lollipop_graph_no_path_weights_or_num(self):
        graph = rustworkx.generators.lollipop_graph(mesh_weights=list(range(4)))
        mesh = rustworkx.generators.mesh_graph(weights=list(range(4)))
        self.assertEqual(graph.nodes(), mesh.nodes())
        self.assertEqual(graph.weighted_edge_list(), mesh.weighted_edge_list())
        self.assertEqual(
            rustworkx.generators.lollipop_graph(4).edge_list(),
            rustworkx.generators.mesh_graph(4).edge_list(),
        )

    def test_lollipop_graph_no_mesh_weights_or_num(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.lollipop_graph()
