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


class TestLocalComplement(unittest.TestCase):
    def test_multigraph(self):
        graph = rustworkx.PyGraph(multigraph=True)
        node = graph.add_node("")
        with self.assertRaises(ValueError):
            rustworkx.local_complement(graph, node)

    def test_invalid_node(self):
        graph = rustworkx.PyGraph(multigraph=False)
        node = graph.add_node("")
        with self.assertRaises(rustworkx.InvalidNode):
            rustworkx.local_complement(graph, node + 1)

    def test_clique(self):
        N = 5
        graph = rustworkx.generators.complete_graph(N, multigraph=False)

        for node in range(0, N):
            expected_graph = rustworkx.PyGraph(multigraph=False)
            expected_graph.extend_from_edge_list([(i, node) for i in range(0, N) if i != node])

            complement_graph = rustworkx.local_complement(graph, node)

            self.assertTrue(
                rustworkx.is_isomorphic(
                    expected_graph,
                    complement_graph,
                )
            )

    def test_empty(self):
        N = 5
        graph = rustworkx.generators.empty_graph(N, multigraph=False)

        expected_graph = rustworkx.generators.empty_graph(N, multigraph=False)

        complement_graph = rustworkx.local_complement(graph, 0)
        self.assertTrue(
            rustworkx.is_isomorphic(
                expected_graph,
                complement_graph,
            )
        )

    def test_local_complement(self):
        # Example took from https://arxiv.org/abs/1910.03969, figure 1
        graph = rustworkx.PyGraph(multigraph=False)
        graph.extend_from_edge_list(
            [(0, 1), (0, 3), (0, 5), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5)]
        )

        expected_graph = rustworkx.PyGraph(multigraph=False)
        expected_graph.extend_from_edge_list(
            [(0, 1), (0, 3), (0, 5), (1, 2), (1, 3), (1, 5), (2, 3), (2, 4), (3, 4)]
        )

        complement_graph = rustworkx.local_complement(graph, 0)
        self.assertTrue(
            rustworkx.is_isomorphic(
                expected_graph,
                complement_graph,
            )
        )
