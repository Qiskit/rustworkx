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
    def test_clique(self):
        N = 5
        graph = rustworkx.PyGraph()
        graph.extend_from_edge_list([(i, j) for i in range(N) for j in range(N) if i < j])

        for node in range(0, N):
            expected_graph = rustworkx.PyGraph()
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
        graph = rustworkx.PyGraph()
        graph.add_nodes_from([i for i in range(N)])

        expected_graph = rustworkx.PyGraph()
        expected_graph.add_nodes_from([i for i in range(N)])

        complement_graph = rustworkx.local_complement(graph, 0)
        self.assertTrue(
            rustworkx.is_isomorphic(
                expected_graph,
                complement_graph,
            )
        )

    # TODO: More tests!

    # TODO
    # def test_multigraph(self):
    #     graph = rustworkx.PyGraph(multigraph=True)
    #     graph.extend_from_edge_list([(0, 1), (1, 0), (0, 0)])

    #     expected_graph = rustworkx.PyGraph(multigraph=True)
    #     expected_graph.extend_from_edge_list([(0, 1), (1, 0), (0, 0)])

    #     complement_graph = rustworkx.local_complement(graph, 0)
    #     self.assertTrue(
    #         rustworkx.is_isomorphic(
    #             expected_graph,
    #             complement_graph,
    #         )
    #     )
