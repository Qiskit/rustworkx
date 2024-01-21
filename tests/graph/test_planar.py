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

import itertools
import rustworkx as rx


class TestPlanarGraph(unittest.TestCase):
    def test_simple_planar_graph(self):
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 6),
                (6, 7),
                (7, 1),
                (1, 5),
                (5, 2),
                (2, 4),
                (4, 5),
                (5, 7),
            ]
        )
        res = rx.is_planar(graph)
        self.assertTrue(res)

    def test_planar_with_selfloop(self):
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (1, 2),
                (1, 3),
                (1, 5),
                (2, 5),
                (2, 4),
                (3, 4),
                (3, 5),
                (4, 5),
            ]
        )
        res = rx.is_planar(graph)
        self.assertTrue(res)

    def test_grid_graph(self):
        graph = rx.generators.grid_graph(5, 5)
        res = rx.is_planar(graph)
        self.assertTrue(res)

    def test_k3_3(self):
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (2, 4),
                (2, 5),
            ]
        )
        res = rx.is_planar(graph)
        self.assertFalse(res)

    def test_k5(self):
        graph = rx.generators.mesh_graph(5)
        res = rx.is_planar(graph)
        self.assertFalse(res)

    def test_multiple_components_planar(self):
        graph = rx.PyGraph()
        graph.extend_from_edge_list([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
        res = rx.is_planar(graph)
        self.assertTrue(res)

    def test_multiple_components_non_planar(self):
        graph = rx.generators.mesh_graph(5)
        # add another planar component to the non planar component
        # G stays non planar
        graph.extend_from_edge_list([(6, 7), (7, 8), (8, 6)])
        res = rx.is_planar(graph)
        self.assertFalse(res)

    def test_non_planar_with_selfloop(self):
        graph = rx.generators.mesh_graph(5)
        # add self loops
        for i in range(5):
            graph.add_edge(i, i, None)
        res = rx.is_planar(graph)
        self.assertFalse(res)

    def test_non_planar1(self):
        # tests a graph that has no subgraph directly isomorph to K5 or K3_3
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [
                (1, 5),
                (1, 6),
                (1, 7),
                (2, 6),
                (2, 3),
                (3, 5),
                (3, 7),
                (4, 5),
                (4, 6),
                (4, 7),
            ]
        )
        res = rx.is_planar(graph)
        self.assertFalse(res)

    def test_loop(self):
        # test a graph with a selfloop
        graph = rx.PyGraph()
        graph.extend_from_edge_list([(0, 1), (1, 1)])
        res = rx.is_planar(graph)
        self.assertTrue(res)

    def test_goldner_harary(self):
        # test goldner-harary graph (a maximal planar graph)
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 7),
                (1, 8),
                (1, 10),
                (1, 11),
                (2, 3),
                (2, 4),
                (2, 6),
                (2, 7),
                (2, 9),
                (2, 10),
                (2, 11),
                (3, 4),
                (4, 5),
                (4, 6),
                (4, 7),
                (5, 7),
                (6, 7),
                (7, 8),
                (7, 9),
                (7, 10),
                (8, 10),
                (9, 10),
                (10, 11),
            ]
        )
        res = rx.is_planar(graph)
        self.assertTrue(res)

    def test_planar_multigraph(self):
        graph = rx.PyGraph()
        graph.extend_from_edge_list([(1, 2), (1, 2), (1, 2), (1, 2), (2, 3), (3, 1)])
        res = rx.is_planar(graph)
        self.assertTrue(res)

    def test_non_planar_multigraph(self):
        graph = rx.generators.mesh_graph(5)
        graph.add_edges_from_no_data([(1, 2)] * 5)
        res = rx.is_planar(graph)
        self.assertFalse(res)

    def test_single_component(self):
        # Test a graph with only a single node
        graph = rx.PyGraph()
        graph.add_node(1)
        res = rx.is_planar(graph)
        self.assertTrue(res)

    def test_graph1(self):
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [
                (3, 10),
                (2, 13),
                (1, 13),
                (7, 11),
                (0, 8),
                (8, 13),
                (0, 2),
                (0, 7),
                (0, 10),
                (1, 7),
            ]
        )
        res = rx.is_planar(graph)
        self.assertTrue(res)

    def test_graph2(self):
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [
                (1, 2),
                (4, 13),
                (0, 13),
                (4, 5),
                (7, 10),
                (1, 7),
                (0, 3),
                (2, 6),
                (5, 6),
                (7, 13),
                (4, 8),
                (0, 8),
                (0, 9),
                (2, 13),
                (6, 7),
                (3, 6),
                (2, 8),
            ]
        )
        res = rx.is_planar(graph)
        self.assertFalse(res)

    def test_graph3(self):
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [
                (0, 7),
                (3, 11),
                (3, 4),
                (8, 9),
                (4, 11),
                (1, 7),
                (1, 13),
                (1, 11),
                (3, 5),
                (5, 7),
                (1, 3),
                (0, 4),
                (5, 11),
                (5, 13),
            ]
        )
        res = rx.is_planar(graph)
        self.assertFalse(res)

    def test_generalized_petersen_graph_planar_instances(self):
        # see Table 2: https://www.sciencedirect.com/science/article/pii/S0166218X08000371
        planars = itertools.chain(
            iter((n, 1) for n in range(3, 17)),
            iter((n, 2) for n in range(6, 17, 2)),
        )
        for (n, k) in planars:
            with self.subTest(n=n, k=k):
                graph = rx.generators.generalized_petersen_graph(n=n, k=k)
                self.assertTrue(rx.is_planar(graph))

    def test_generalized_petersen_graph_non_planar_instances(self):
        # see Table 2: https://www.sciencedirect.com/science/article/pii/S0166218X08000371
        no_planars = itertools.chain(
            iter((n, 2) for n in range(5, 17, 2)),
            iter((n, k) for k in range(3, 9) for n in range(2 * k + 1, 17)),
        )
        for (n, k) in no_planars:
            with self.subTest(n=n, k=k):
                graph = rx.generators.generalized_petersen_graph(n=n, k=k)
                self.assertFalse(rx.is_planar(graph))
