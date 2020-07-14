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

    def test_random_gnp_undirected_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            retworkx.undirected_gnp_random_graph(-23, .5)

    def test_random_gnp_undirected_invalid_probability(self):
        with self.assertRaises(ValueError):
            retworkx.undirected_gnp_random_graph(23, 123.5)

    def test_random_gnp_directed_edges_close_to_probability(self):
        edges = 0
        runs = 100
        for i in range(runs):
            edges += sum(1 for _ in retworkx.directed_gnp_random_graph(
                10, 0.99999).edges())
        self.assertGreaterEqual(abs(edges / float(runs) - 90),
                                runs * 2.0 / 100)

    def test_random_gnp_undirected_edges_close_to_probability(self):
        edges = 0
        runs = 100
        for i in range(runs):
            edges += sum(1 for _ in retworkx.undirected_gnp_random_graph(
                10, 0.99999).edges())
        self.assertGreaterEqual(abs(edges / float(runs) - 90),
                                runs * 2.0 / 100)
