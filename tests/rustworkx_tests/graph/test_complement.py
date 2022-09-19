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


class TestComplement(unittest.TestCase):
    def test_clique(self):
        N = 5
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list([(i, j) for i in range(N) for j in range(N) if i < j])

        complement_graph = rustworkx.complement(graph)
        self.assertEqual(graph.nodes(), complement_graph.nodes())
        self.assertEqual(0, len(complement_graph.edges()))

    def test_empty(self):
        N = 5
        graph = rustworkx.PyGraph()
        graph.add_nodes_from([i for i in range(N)])

        expected_graph = rustworkx.PyGraph()
        expected_graph.extend_from_edge_list([(i, j) for i in range(N) for j in range(N) if i < j])

        complement_graph = rustworkx.complement(graph)
        self.assertTrue(
            rustworkx.is_isomorphic(
                expected_graph,
                complement_graph,
            )
        )

    def test_null_graph(self):
        graph = rustworkx.PyGraph()
        complement_graph = rustworkx.complement(graph)
        self.assertEqual(0, len(complement_graph.nodes()))
        self.assertEqual(0, len(complement_graph.edges()))

    def test_complement(self):
        N = 8
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list(
            [(j, i) for i in range(N) for j in range(N) if i < j and (i + j) % 3 == 0]
        )

        expected_graph = rustworkx.PyGraph()
        expected_graph.extend_from_edge_list(
            [(i, j) for i in range(N) for j in range(N) if i < j and (i + j) % 3 != 0]
        )

        complement_graph = rustworkx.complement(graph)
        self.assertTrue(
            rustworkx.is_isomorphic(
                expected_graph,
                complement_graph,
            )
        )

    def test_multigraph(self):
        graph = rustworkx.PyGraph(multigraph=True)
        graph.extend_from_edge_list([(0, 0), (0, 1), (1, 1), (2, 2), (1, 0)])

        expected_graph = rustworkx.PyGraph(multigraph=True)
        expected_graph.extend_from_edge_list([(0, 2), (1, 2)])

        complement_graph = rustworkx.complement(graph)
        self.assertTrue(
            rustworkx.is_isomorphic(
                expected_graph,
                complement_graph,
            )
        )
