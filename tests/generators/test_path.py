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


class TestPathGraph(unittest.TestCase):

    def test_directed_path_graph(self):
        graph = retworkx.generators.directed_path_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 19)
        for i in range(19):
            self.assertEqual(graph.out_edges(i), [(i, i + 1, None)])

    def test_directed_path_graph_weights(self):
        graph = retworkx.generators.directed_path_graph(
            weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 19)
        for i in range(19):
            self.assertEqual(graph.out_edges(i), [(i, i + 1, None)])

    def test_path_directed_both_weights_and_num(self):
        with self.assertRaises(IndexError):
            retworkx.generators.directed_path_graph()

    def test_path_graph(self):
        graph = retworkx.generators.path_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 19)

    def test_path_graph_weights(self):
        graph = retworkx.generators.path_graph(weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 19)

    def test_path_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            retworkx.generators.path_graph()
