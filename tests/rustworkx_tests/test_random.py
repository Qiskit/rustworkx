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
import random
from ddt import ddt, data

import rustworkx

@ddt
class TestGNPRandomGraph(unittest.TestCase):
    @data((15, 20, 0.7, 156), (20, 10, 0.5, 189), (22, 6, 0.2, 91))
    def test_random_gnp_directed(self, args):
        graph = rustworkx.directed_gnp_random_graph(args[0], args[2], seed=args[1])
        self.assertEqual(len(graph), args[0])
        self.assertEqual(len(graph.edges()), args[3])

    def test_random_gnp_directed_empty_graph(self):
        graph = rustworkx.directed_gnp_random_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_gnp_directed_complete_graph(self):
        graph = rustworkx.directed_gnp_random_graph(20, 1)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 20 * (20 - 1))

    def test_random_gnp_directed_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            rustworkx.directed_gnp_random_graph(0, 0.5)

    def test_random_gnp_directed_invalid_probability(self):
        with self.assertRaises(ValueError):
            rustworkx.directed_gnp_random_graph(23, 123.5)

    def test_random_gnp_directed_payload(self):
        graph = rustworkx.directed_gnp_random_graph(3, 0.5)
        self.assertEqual(graph.nodes(), [0, 1, 2])

    def test_random_gnp_undirected(self):
        graph = rustworkx.undirected_gnp_random_graph(20, 0.5, seed=10)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 105)

    def test_random_gnp_undirected_empty_graph(self):
        graph = rustworkx.undirected_gnp_random_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_gnp_undirected_complete_graph(self):
        graph = rustworkx.undirected_gnp_random_graph(20, 1)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 20 * (20 - 1) / 2)

    def test_random_gnp_undirected_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            rustworkx.undirected_gnp_random_graph(0, 0.5)

    def test_random_gnp_undirected_invalid_probability(self):
        with self.assertRaises(ValueError):
            rustworkx.undirected_gnp_random_graph(23, 123.5)

    def test_random_gnp_undirected_payload(self):
        graph = rustworkx.undirected_gnp_random_graph(3, 0.5)
        self.assertEqual(graph.nodes(), [0, 1, 2])


class TestGNMRandomGraph(unittest.TestCase):
    def test_random_gnm_directed(self):
        graph = rustworkx.directed_gnm_random_graph(20, 100)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 100)
        # with other arguments equal, same seed results in same graph
        graph_s1 = rustworkx.directed_gnm_random_graph(20, 100, seed=10)
        graph_s2 = rustworkx.directed_gnm_random_graph(20, 100, seed=10)
        self.assertEqual(graph_s1.edge_list(), graph_s2.edge_list())

    def test_random_gnm_directed_empty_graph(self):
        graph = rustworkx.directed_gnm_random_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)
        # passing a seed when passing zero edges has no effect
        graph = rustworkx.directed_gnm_random_graph(20, 0, 44)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_gnm_directed_complete_graph(self):
        n = 20
        max_m = n * (n - 1)
        # passing the max edges for the passed number of nodes
        graph = rustworkx.directed_gnm_random_graph(n, max_m)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)
        # passing m > the max edges n(n-1) still returns the max edges
        graph = rustworkx.directed_gnm_random_graph(n, max_m + 1)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)
        # passing a seed when passing max edges has no effect
        graph = rustworkx.directed_gnm_random_graph(n, max_m, 55)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)

    def test_random_gnm_directed_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            rustworkx.directed_gnm_random_graph(0, 0)

    def test_random_gnm_directed_invalid_num_edges(self):
        with self.assertRaises(OverflowError):
            rustworkx.directed_gnm_random_graph(23, -5)

    def test_random_gnm_directed_payload(self):
        graph = rustworkx.directed_gnm_random_graph(3, 3)
        self.assertEqual(graph.nodes(), [0, 1, 2])

    def test_random_gnm_undirected(self):
        graph = rustworkx.undirected_gnm_random_graph(20, 100)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 100)
        # with other arguments equal, same seed results in same graph
        graph_s1 = rustworkx.undirected_gnm_random_graph(20, 100, seed=10)
        graph_s2 = rustworkx.undirected_gnm_random_graph(20, 100, seed=10)
        self.assertEqual(graph_s1.edge_list(), graph_s2.edge_list())

    def test_random_gnm_undirected_empty_graph(self):
        graph = rustworkx.undirected_gnm_random_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)
        # passing a seed when passing zero edges has no effect
        graph = rustworkx.undirected_gnm_random_graph(20, 0, 44)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_gnm_undirected_complete_graph(self):
        n = 20
        max_m = n * (n - 1) // 2
        # passing the max edges for the passed number of nodes
        graph = rustworkx.undirected_gnm_random_graph(n, max_m)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)
        # passing m > the max edges n(n-1)/2 still returns the max edges
        graph = rustworkx.undirected_gnm_random_graph(n, max_m + 1)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)
        # passing a seed when passing max edges has no effect
        graph = rustworkx.undirected_gnm_random_graph(n, max_m, 55)
        self.assertEqual(len(graph), n)
        self.assertEqual(len(graph.edges()), max_m)

    def test_random_gnm_undirected_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            rustworkx.undirected_gnm_random_graph(0, 5)

    def test_random_gnm_undirected_invalid_num_edges(self):
        with self.assertRaises(OverflowError):
            rustworkx.undirected_gnm_random_graph(23, -5)

    def test_random_gnm_undirected_payload(self):
        graph = rustworkx.undirected_gnm_random_graph(3, 3)
        self.assertEqual(graph.nodes(), [0, 1, 2])


class TestGeometricRandomGraph(unittest.TestCase):
    def test_random_geometric_empty(self):
        graph = rustworkx.random_geometric_graph(20, 0)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_random_geometric_complete(self):
        r = 1.42  # > sqrt(2)
        graph = rustworkx.random_geometric_graph(10, r)
        self.assertEqual(len(graph), 10)
        self.assertEqual(len(graph.edges()), 45)

    def test_random_geometric_same_seed(self):
        # with other arguments equal, same seed results in same graph
        graph_s1 = rustworkx.random_geometric_graph(20, 0.5, seed=10)
        graph_s2 = rustworkx.random_geometric_graph(20, 0.5, seed=10)
        self.assertEqual(graph_s1.edge_list(), graph_s2.edge_list())

    def test_random_geometric_dim(self):
        graph = rustworkx.random_geometric_graph(10, 0.5, dim=3)
        self.assertEqual(len(graph[0]["pos"]), 3)

    def test_random_geometric_pos(self):
        pos = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
        graph = rustworkx.random_geometric_graph(3, 0.15, pos=pos)
        self.assertEqual(set(graph.edge_list()), {(0, 1), (1, 2)})
        for i in range(3):
            self.assertEqual(graph[i]["pos"], pos[i])

    def test_random_geometric_pos_1norm(self):
        pos = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
        graph = rustworkx.random_geometric_graph(3, 0.21, pos=pos, p=1.0)
        self.assertEqual(set(graph.edge_list()), {(0, 1), (1, 2)})

    def test_random_geometric_pos_inf_norm(self):
        pos = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
        graph = rustworkx.random_geometric_graph(3, 0.11, pos=pos, p=float("inf"))
        self.assertEqual(set(graph.edge_list()), {(0, 1), (1, 2)})

    def test_random_geometric_num_nodes_invalid(self):
        with self.assertRaises(ValueError):
            rustworkx.random_geometric_graph(0, 1.0)

    def test_random_geometric_pos_num_nodes_incomp(self):
        with self.assertRaises(ValueError):
            rustworkx.random_geometric_graph(3, 0.15, pos=[[0.5, 0.5]])


class TestRandomSubGraphIsomorphism(unittest.TestCase):
    def test_random_gnm_induced_subgraph_isomorphism(self):
        graph = rustworkx.undirected_gnm_random_graph(50, 150)
        nodes = random.sample(range(50), 25)
        subgraph = graph.subgraph(nodes)

        self.assertTrue(
            rustworkx.is_subgraph_isomorphic(graph, subgraph, id_order=True, induced=True)
        )

    def test_random_gnm_non_induced_subgraph_isomorphism(self):
        graph = rustworkx.undirected_gnm_random_graph(50, 150)
        nodes = random.sample(range(50), 25)
        subgraph = graph.subgraph(nodes)

        indexes = list(subgraph.edge_indices())
        for idx in random.sample(indexes, len(indexes) // 2):
            subgraph.remove_edge_from_index(idx)

        self.assertTrue(
            rustworkx.is_subgraph_isomorphic(graph, subgraph, id_order=True, induced=False)
        )
