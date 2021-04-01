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


class TestComplement(unittest.TestCase):
    def test_null_graph(self):
        graph = retworkx.PyDiGraph()
        complement_graph = retworkx.complement(graph)
        self.assertEqual(0, len(complement_graph.nodes()))
        self.assertEqual(0, len(complement_graph.edges()))

    def test_clique_directed(self):
        N = 5
        graph = retworkx.PyDiGraph()
        graph.extend_from_edge_list(
            [(i, j) for i in range(N) for j in range(N) if i != j]
        )

        complement_graph = retworkx.complement(graph)
        self.assertEqual(graph.nodes(), complement_graph.nodes())
        self.assertEqual(0, len(complement_graph.edges()))

    def test_empty_directed(self):
        N = 5
        graph = retworkx.PyDiGraph()
        graph.add_nodes_from([i for i in range(N)])

        expected_graph = retworkx.PyDiGraph()
        expected_graph.extend_from_edge_list(
            [(i, j) for i in range(N) for j in range(N) if i != j]
        )

        complement_graph = retworkx.complement(graph)
        self.assertTrue(
            retworkx.is_isomorphic(
                expected_graph,
                complement_graph,
            )
        )

    def test_complement_directed(self):
        N = 8
        graph = retworkx.PyDiGraph()
        graph.extend_from_edge_list(
            [
                (i, j)
                for i in range(N)
                for j in range(N)
                if i != j and (i + j) % 3 == 0
            ]
        )

        expected_graph = retworkx.PyDiGraph()
        expected_graph.extend_from_edge_list(
            [
                (i, j)
                for i in range(N)
                for j in range(N)
                if i != j and (i + j) % 3 != 0
            ]
        )

        complement_graph = retworkx.complement(graph)
        self.assertTrue(
            retworkx.is_isomorphic(
                expected_graph,
                complement_graph,
            )
        )
