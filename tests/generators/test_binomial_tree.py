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


class TestBinomailTreeGraph(unittest.TestCase):

    def test_binomial_tree_graph(self):
        for n in range(10):
            with self.subTest(n=n):
                graph = retworkx.generators.binomial_tree_graph(n)
                self.assertEqual(len(graph), 2**n)
                self.assertEqual(len(graph.edges()), 2**n - 1)

    def test_binomial_tree_graph_weights(self):
        graph = retworkx.generators.binomial_tree_graph(
            2, weights=list(range(4)))
        self.assertEqual(len(graph), 4)
        self.assertEqual([x for x in range(4)], graph.nodes())
        self.assertEqual(len(graph.edges()), 3)

    def test_binomial_tree_no_order(self):
        with self.assertRaises(TypeError):
            retworkx.generators.binomial_tree_graph(weights=list(range(4)))

    def test_directed_binomial_tree_graph(self):
        for n in range(10):
            graph = retworkx.generators.directed_binomial_tree_graph(n)
            self.assertEqual(len(graph), 2**n)
            self.assertEqual(len(graph.edges()), 2**n - 1)

    def test_directed_binomial_tree_graph_weights(self):
        graph = retworkx.generators.directed_binomial_tree_graph(
            2, weights=list(range(4)))
        self.assertEqual(len(graph), 4)
        self.assertEqual([x for x in range(4)], graph.nodes())
        self.assertEqual(len(graph.edges()), 3)

    def test_directed_binomial_tree_no_order(self):
        with self.assertRaises(TypeError):
            retworkx.generators.directed_binomial_tree_graph(
                weights=list(range(4)))
