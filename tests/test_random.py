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


class TestGNPRandomGraph(unittest.TestCase):

    def test_random_gnp_directed(self):
        graph = retworkx.directed_gnp_random_graph(20, .5, seed=10)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 104)

    def test_random_gnp_directed_empty_graph(self):
        graph = retworkx.directed_gnp_random_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_gnp_directed_complete_graph(self):
        graph = retworkx.directed_gnp_random_graph(20, 1)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 20 * (20 - 1))

    def test_random_gnp_directed_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            retworkx.directed_gnp_random_graph(-23, .5)

    def test_random_gnp_directed_invalid_probability(self):
        with self.assertRaises(ValueError):
            retworkx.directed_gnp_random_graph(23, 123.5)

    def test_random_gnp_undirected(self):
        graph = retworkx.undirected_gnp_random_graph(20, .5, seed=10)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 105)

    def test_random_gnp_undirected_empty_graph(self):
        graph = retworkx.undirected_gnp_random_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_gnp_undirected_complete_graph(self):
        graph = retworkx.undirected_gnp_random_graph(20, 1)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 20 * (20 - 1) / 2)

    def test_random_gnp_undirected_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            retworkx.undirected_gnp_random_graph(-23, .5)

    def test_random_gnp_undirected_invalid_probability(self):
        with self.assertRaises(ValueError):
            retworkx.undirected_gnp_random_graph(23, 123.5)
