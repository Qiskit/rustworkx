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


class TestDifference(unittest.TestCase):
    def test_null_difference_null(self):
        graph_1 = retworkx.PyDiGraph()
        graph_2 = retworkx.PyDiGraph()

        graph_difference = retworkx.digraph_difference(graph_1, graph_2)

        self.assertEqual(graph_difference.num_nodes(), 0)
        self.assertEqual(graph_difference.num_edges(), 0)

    def test_difference_non_matching(self):
        graph_1 = retworkx.generators.directed_path_graph(2)
        graph_2 = retworkx.generators.directed_path_graph(3)

        with self.assertRaises(IndexError):
            _ = retworkx.digraph_difference(graph_1, graph_2)

    def test_difference_weights_edges(self):
        graph_1 = retworkx.PyDiGraph()
        graph_1.add_nodes_from(["a_1", "a_2", "a_3", "a_4"])
        graph_1.extend_from_weighted_edge_list(
            [
                (0, 1, "e_1"),
                (1, 2, "e_2"),
                (2, 3, "e_3"),
                (3, 0, "e_4"),
                (0, 2, "e_5"),
                (1, 3, "e_6"),
            ]
        )
        graph_2 = retworkx.PyDiGraph()
        graph_2.add_nodes_from(["a_1", "a_2", "a_3", "a_4"])
        graph_2.extend_from_weighted_edge_list(
            [
                (0, 1, "e_1"),
                (1, 2, "e_2"),
                (2, 3, "e_3"),
                (3, 0, "e_4"),
            ]
        )

        graph_difference = retworkx.digraph_difference(graph_1, graph_2)

        expected_edges = [(0, 2, "e_5"), (1, 3, "e_6")]
        self.assertEqual(graph_difference.num_nodes(), 4)
        self.assertEqual(graph_difference.num_edges(), 2)
        self.assertEqual(graph_difference.weighted_edge_list(), expected_edges)
