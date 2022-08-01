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


class TestMeshGraph(unittest.TestCase):
    def test_directed_mesh_graph(self):
        graph = rustworkx.generators.directed_mesh_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 380)
        for i in range(20):
            ls = []
            for j in range(19, -1, -1):
                if i != j:
                    ls.append((i, j, None))
            self.assertEqual(graph.out_edges(i), ls)

    def test_directed_mesh_graph_weights(self):
        graph = rustworkx.generators.directed_mesh_graph(weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 380)
        for i in range(20):
            ls = []
            for j in range(19, -1, -1):
                if i != j:
                    ls.append((i, j, None))
            self.assertEqual(graph.out_edges(i), ls)

    def test_mesh_directed_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.directed_mesh_graph()

    def test_mesh_graph(self):
        graph = rustworkx.generators.mesh_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 190)

    def test_mesh_graph_weights(self):
        graph = rustworkx.generators.mesh_graph(weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 190)

    def test_mesh_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.mesh_graph()

    def test_zero_size_mesh_graph(self):
        graph = rustworkx.generators.mesh_graph(0)
        self.assertEqual(0, len(graph))

    def test_zero_size_directed_mesh_graph(self):
        graph = rustworkx.generators.directed_mesh_graph(0)
        self.assertEqual(0, len(graph))
