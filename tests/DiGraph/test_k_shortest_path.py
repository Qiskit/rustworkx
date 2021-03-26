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


class TestKShortestpath(unittest.TestCase):
    def test_digraph_k_shortest_path_lengths(self):
        graph = retworkx.PyDiGraph()
        graph.add_nodes_from(list(range(8)))
        graph.add_edges_from_no_data([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (1, 4),
            (5, 6),
            (6, 7),
            (7, 5)
        ])
        res = retworkx.digraph_k_shortest_path_lengths(graph, 1, 2,
                                                       lambda _: 1)
        expected = {0: 7.0, 1: 4.0, 2: 5.0, 3: 6.0, 4: 5.0, 5: 5.0, 6: 6.0,
                    7: 7.0}
        self.assertEqual(res, expected)

    def test_digraph_k_shortest_path_lengths_with_goal(self):
        graph = retworkx.PyDiGraph()
        graph.add_nodes_from(list(range(8)))
        graph.add_edges_from_no_data([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (1, 4),
            (5, 6),
            (6, 7),
            (7, 5)
        ])
        res = retworkx.digraph_k_shortest_path_lengths(graph, 1, 2,
                                                       lambda _: 1, 3)
        self.assertEqual(res, {3: 6})
