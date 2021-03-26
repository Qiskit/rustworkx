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


class TestGNMRandomGraph(unittest.TestCase):

    def test_random_gnm_directed(self):
        graph = retworkx.directed_gnm_random_graph(20, 100)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 100)
        # with other arguments equal, same seed results in same graph
        graph_s1 = retworkx.directed_gnm_random_graph(20, 100, seed=10)
        graph_s2 = retworkx.directed_gnm_random_graph(20, 100, seed=10)
        self.assertEqual(graph_s1.edge_list(), graph_s2.edge_list())

    def test_random_gnm_directed_empty_graph(self):
        graph = retworkx.directed_gnm_random_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)
        # passing a seed when passing zero edges has no effect
        graph = retworkx.directed_gnm_random_graph(20, 0, 44)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_gnm_directed_complete_graph(self):
        n = 20
        max_m = n * (n - 1)
        # passing the max edges for the passed number of nodes
        graph = retworkx.directed_gnm_random_graph(n, max_m)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)
        # passing m > the max edges n(n-1) still returns the max edges
        graph = retworkx.directed_gnm_random_graph(n, max_m + 1)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)
        # passing a seed when passing max edges has no effect
        graph = retworkx.directed_gnm_random_graph(n, max_m, 55)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)

    def test_random_gnm_directed_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            retworkx.directed_gnm_random_graph(-23, 5)

    def test_random_gnm_directed_invalid_num_edges(self):
        with self.assertRaises(ValueError):
            retworkx.directed_gnm_random_graph(23, -5)

    def test_random_gnm_undirected(self):
        graph = retworkx.undirected_gnm_random_graph(20, 100)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 100)
        # with other arguments equal, same seed results in same graph
        graph_s1 = retworkx.undirected_gnm_random_graph(20, 100, seed=10)
        graph_s2 = retworkx.undirected_gnm_random_graph(20, 100, seed=10)
        self.assertEqual(graph_s1.edge_list(), graph_s2.edge_list())

    def test_random_gnm_undirected_empty_graph(self):
        graph = retworkx.undirected_gnm_random_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)
        # passing a seed when passing zero edges has no effect
        graph = retworkx.undirected_gnm_random_graph(20, 0, 44)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_gnm_undirected_complete_graph(self):
        n = 20
        max_m = n * (n - 1) // 2
        # passing the max edges for the passed number of nodes
        graph = retworkx.undirected_gnm_random_graph(n, max_m)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)
        # passing m > the max edges n(n-1)/2 still returns the max edges
        graph = retworkx.undirected_gnm_random_graph(n, max_m + 1)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)
        # passing a seed when passing max edges has no effect
        graph = retworkx.undirected_gnm_random_graph(n, max_m, 55)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)

    def test_random_gnm_undirected_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            retworkx.undirected_gnm_random_graph(-23, 5)

    def test_random_gnm_undirected_invalid_probability(self):
        with self.assertRaises(ValueError):
            retworkx.undirected_gnm_random_graph(23, -5)


class TestGeometricRandomGraph(unittest.TestCase):

    def test_random_geometric_empty(self):
        graph = retworkx.random_geometric_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_geometric_complete(self):
        r = 1.42 # > sqrt(2)
        graph = retworkx.random_geometric_graph(10, r)
        self.assertEqual(len(graph), 10)
        self.assertEqual(len(graph.edges()), 45)

    def test_random_geometric_same_seed(self):
        # with other arguments equal, same seed results in same graph
        graph_s1 = retworkx.random_geometric_graph(20, 0.5, seed=10)
        graph_s2 = retworkx.random_geometric_graph(20, 0.5, seed=10)
        self.assertEqual(graph_s1.edge_list(), graph_s2.edge_list())

    def test_random_geometric_dim(self):
        graph = retworkx.random_geometric_graph(10, 0.5, dim=3)
        self.assertEqual(len(graph[0]['pos']), 3)

    def test_random_geometric_pos(self):
        pos = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
        graph = retworkx.random_geometric_graph(3, 0.15, pos=pos)
        self.assertEqual(set(graph.edge_list()), {(0, 1), (1, 2)})
        for i in range(3):
            self.assertEqual(graph[i]['pos'], pos[i])

    def test_random_geometric_pos_1norm(self):
        pos = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
        graph = retworkx.random_geometric_graph(3, 0.21, pos=pos, p=1.0)
        self.assertEqual(set(graph.edge_list()), {(0, 1), (1, 2)})

    def test_random_geometric_pos_inf_norm(self):
        pos = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
        graph = retworkx.random_geometric_graph(3, 0.11, pos=pos, p=float('inf'))
        self.assertEqual(set(graph.edge_list()), {(0, 1), (1, 2)})

    def test_random_geometric_num_nodes_invalid(self):
        with self.assertRaises(ValueError):
            graph = retworkx.random_geometric_graph(0, 1.0)

    def test_random_geometric_pos_num_nodes_incomp(self):
        with self.assertRaises(ValueError):
            graph = retworkx.random_geometric_graph(3, 0.15, pos=[[0.5, 0.5]])
