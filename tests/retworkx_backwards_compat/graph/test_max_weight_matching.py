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

# These tests are adapated from the networkx test cases:
# https://github.com/networkx/networkx/blob/3351206a3ce5b3a39bb2fc451e93ef545b96c95b/networkx/algorithms/tests/test_matching.py

import random
import unittest

import networkx

import retworkx


def match_dict_to_set(match):
    return {(u, v) for (u, v) in set(map(frozenset, match.items()))}


class TestMaxWeightMatching(unittest.TestCase):
    def compare_match_sets(self, rx_match, expected_match):
        for (u, v) in rx_match:
            if (u, v) not in expected_match and (v, u) not in expected_match:
                self.fail(
                    f"Element {(u, v)} and it's reverse {(v, u)} not found in "
                    f"expected output.\nretworkx output: {rx_match}\nexpected "
                    f"output: {expected_match}"
                )

    def compare_rx_nx_sets(self, rx_graph, rx_matches, nx_matches, seed, nx_graph):
        def get_rx_weight(edge):
            weight = rx_graph.get_edge_data(*edge)
            if weight is None:
                return 1
            return weight

        def get_nx_weight(edge):
            weight = nx_graph.get_edge_data(*edge)
            if not weight:
                return 1
            return weight["weight"]

        not_match = False
        for (u, v) in rx_matches:
            if (u, v) not in nx_matches:
                if (v, u) not in nx_matches:
                    print(
                        "seed {} failed. Element {} and it's "
                        "reverse {} not found in networkx output.\nretworkx"
                        " output: {}\nnetworkx output: {}\nedge list: {}\n"
                        "falling back to checking for a valid solution".format(
                            seed,
                            (u, v),
                            (v, u),
                            rx_matches,
                            nx_matches,
                            list(rx_graph.weighted_edge_list()),
                        )
                    )
                    not_match = True
                    break
        if not_match:
            self.assertTrue(
                retworkx.is_matching(rx_graph, rx_matches),
                "%s is not a valid matching" % rx_matches,
            )
            self.assertTrue(
                retworkx.is_maximal_matching(rx_graph, rx_matches),
                "%s is not a maximal matching" % rx_matches,
            )
            self.assertEqual(
                sum(map(get_rx_weight, rx_matches)),
                sum(map(get_nx_weight, nx_matches)),
            )

    def test_empty_graph(self):
        graph = retworkx.PyGraph()
        self.assertEqual(retworkx.max_weight_matching(graph), set())

    def test_single_edge(self):
        graph = retworkx.PyGraph()
        graph.add_nodes_from([0, 1])
        graph.add_edges_from([(0, 1, 1)])
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, verify_optimum=True),
            {
                (0, 1),
            },
        )

    def test_single_edge_no_verification(self):
        graph = retworkx.PyGraph()
        graph.add_nodes_from([0, 1])
        graph.add_edges_from([(0, 1, 1)])
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, verify_optimum=False),
            {
                (0, 1),
            },
        )

    def test_single_self_edge(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list([(0, 0, 100)])
        self.assertEqual(retworkx.max_weight_matching(graph), set())

    def test_small_graph(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list([(1, 2, 10), (2, 3, 11)])
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {
                (2, 3),
            },
        )

    def test_path_graph(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list([(1, 2, 5), (2, 3, 11), (3, 4, 5)])
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {
                (2, 3),
            },
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, True, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 2), (3, 4)},
        )

    def test_negative_weights(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 2),
                (1, 3, -2),
                (2, 3, 1),
                (2, 4, -1),
                (3, 4, -6),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {
                (1, 2),
            },
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, True, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 3), (2, 4)},
        )

    def test_s_blossom(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (0, 1, 8),
                (0, 2, 9),
                (1, 2, 10),
                (2, 3, 7),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(0, 1), (2, 3)},
        )
        graph.extend_from_weighted_edge_list([(0, 5, 5), (3, 4, 6)])
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(0, 5), (1, 2), (3, 4)},
        )

    def test_s_t_blossom(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 9),
                (1, 3, 8),
                (2, 3, 10),
                (1, 4, 5),
                (4, 5, 4),
                (1, 6, 3),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 6), (2, 3), (4, 5)},
        )
        graph.remove_edge(1, 6)
        graph.remove_edge(4, 5)
        graph.extend_from_weighted_edge_list([(4, 5, 3), (1, 6, 4)])
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 6), (2, 3), (4, 5)},
        )
        graph.remove_edge(1, 6)
        graph.add_edge(3, 6, 4)
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 2), (3, 6), (4, 5)},
        )

    def test_s_t_blossom_with_removed_nodes(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 9),
                (1, 3, 8),
                (2, 3, 10),
                (1, 4, 5),
                (4, 5, 4),
                (1, 6, 3),
            ]
        )
        node_id = graph.add_node(None)
        graph.remove_node(5)
        graph.add_edge(4, node_id, 4)
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 6), (2, 3), (4, 7)},
        )
        graph.remove_edge(1, 6)
        graph.remove_edge(4, 7)
        graph.extend_from_weighted_edge_list([(4, node_id, 3), (1, 6, 4)])
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 6), (2, 3), (4, 7)},
        )
        graph.remove_edge(1, 6)
        graph.add_edge(3, 6, 4)
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 2), (3, 6), (4, 7)},
        )

    def test_nested_s_blossom(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 9),
                (1, 3, 9),
                (2, 3, 10),
                (2, 4, 8),
                (3, 5, 8),
                (4, 5, 10),
                (5, 6, 6),
            ]
        )
        expected = {(1, 3), (2, 4), (5, 6)}
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            expected,
        )

    def test_nested_s_blossom_relabel(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 10),
                (1, 7, 10),
                (2, 3, 12),
                (3, 4, 20),
                (3, 5, 20),
                (4, 5, 25),
                (5, 6, 10),
                (6, 7, 10),
                (7, 8, 8),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 2), (3, 4), (5, 6), (7, 8)},
        )

    def test_nested_s_blossom_expand(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 8),
                (1, 3, 8),
                (2, 3, 10),
                (2, 4, 12),
                (3, 5, 12),
                (4, 5, 14),
                (4, 6, 12),
                (5, 7, 12),
                (6, 7, 14),
                (7, 8, 12),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 2), (3, 5), (4, 6), (7, 8)},
        )

    def test_s_blossom_relabel_expand(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 23),
                (1, 5, 22),
                (1, 6, 15),
                (2, 3, 25),
                (3, 4, 22),
                (4, 5, 25),
                (4, 8, 14),
                (5, 7, 13),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            {(1, 6), (2, 3), (4, 8), (5, 7)},
        )

    def test_nested_s_blossom_relabel_expand(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 19),
                (1, 3, 20),
                (1, 8, 8),
                (2, 3, 25),
                (2, 4, 18),
                (3, 5, 18),
                (4, 5, 13),
                (4, 7, 7),
                (5, 6, 7),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            match_dict_to_set({1: 8, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4, 8: 1}),
        )

    def test_blossom_relabel_multiple_paths(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 45),
                (1, 5, 45),
                (2, 3, 50),
                (3, 4, 45),
                (4, 5, 50),
                (1, 6, 30),
                (3, 9, 35),
                (4, 8, 35),
                (5, 7, 26),
                (9, 10, 5),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            match_dict_to_set({1: 6, 2: 3, 3: 2, 4: 8, 5: 7, 6: 1, 7: 5, 8: 4, 9: 10, 10: 9}),
        )

    def test_blossom_relabel_multiple_path_alternate(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 45),
                (1, 5, 45),
                (2, 3, 50),
                (3, 4, 45),
                (4, 5, 50),
                (1, 6, 30),
                (3, 9, 35),
                (4, 8, 26),
                (5, 7, 40),
                (9, 10, 5),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            match_dict_to_set({1: 6, 2: 3, 3: 2, 4: 8, 5: 7, 6: 1, 7: 5, 8: 4, 9: 10, 10: 9}),
        )

    def test_blossom_relabel_multiple_paths_least_slack(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 45),
                (1, 5, 45),
                (2, 3, 50),
                (3, 4, 45),
                (4, 5, 50),
                (1, 6, 30),
                (3, 9, 35),
                (4, 8, 28),
                (5, 7, 26),
                (9, 10, 5),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            match_dict_to_set({1: 6, 2: 3, 3: 2, 4: 8, 5: 7, 6: 1, 7: 5, 8: 4, 9: 10, 10: 9}),
        )

    def test_nested_blossom_expand_recursively(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 40),
                (1, 3, 40),
                (2, 3, 60),
                (2, 4, 55),
                (3, 5, 55),
                (4, 5, 50),
                (1, 8, 15),
                (5, 7, 30),
                (7, 6, 10),
                (8, 10, 10),
                (4, 9, 30),
            ]
        )
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            match_dict_to_set({1: 2, 2: 1, 3: 5, 4: 9, 5: 3, 6: 7, 7: 6, 8: 10, 9: 4, 10: 8}),
        )

    def test_nested_blossom_augmented(self):
        graph = retworkx.PyGraph()
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 45),
                (1, 7, 45),
                (2, 3, 50),
                (3, 4, 45),
                (4, 5, 95),
                (4, 6, 94),
                (5, 6, 94),
                (6, 7, 50),
                (1, 8, 30),
                (3, 11, 35),
                (5, 9, 36),
                (7, 10, 26),
                (11, 12, 5),
            ]
        )
        expected = {
            1: 8,
            2: 3,
            3: 2,
            4: 6,
            5: 9,
            6: 4,
            7: 10,
            8: 1,
            9: 5,
            10: 7,
            11: 12,
            12: 11,
        }
        self.compare_match_sets(
            retworkx.max_weight_matching(graph, weight_fn=lambda x: x, verify_optimum=True),
            match_dict_to_set(expected),
        )

    def test_gnp_random_against_networkx(self):
        for i in range(1024):
            with self.subTest(i=i):
                rx_graph = retworkx.undirected_gnp_random_graph(10, 0.75, seed=42 + i)
                nx_graph = networkx.Graph(list(rx_graph.edge_list()))
                nx_matches = networkx.max_weight_matching(nx_graph)
                rx_matches = retworkx.max_weight_matching(rx_graph, verify_optimum=True)
                self.compare_rx_nx_sets(rx_graph, rx_matches, nx_matches, 42 + i, nx_graph)

    def test_gnp_random_against_networkx_with_weight(self):
        for i in range(1024):
            with self.subTest(i=i):
                random.seed(i)
                rx_graph = retworkx.undirected_gnp_random_graph(10, 0.75, seed=42 + i)
                for edge in rx_graph.edge_list():
                    rx_graph.update_edge(*edge, random.randint(0, 5000))
                nx_graph = networkx.Graph(
                    [(x[0], x[1], {"weight": x[2]}) for x in rx_graph.weighted_edge_list()]
                )
                nx_matches = networkx.max_weight_matching(nx_graph)
                rx_matches = retworkx.max_weight_matching(
                    rx_graph, weight_fn=lambda x: x, verify_optimum=True
                )
                self.compare_rx_nx_sets(rx_graph, rx_matches, nx_matches, 42 + i, nx_graph)

    def test_gnp_random_against_networkx_with_negative_weight(self):
        for i in range(1024):
            with self.subTest(i=i):
                random.seed(i)
                rx_graph = retworkx.undirected_gnp_random_graph(10, 0.75, seed=42 + i)
                for edge in rx_graph.edge_list():
                    rx_graph.update_edge(*edge, random.randint(-5000, 5000))
                nx_graph = networkx.Graph(
                    [(x[0], x[1], {"weight": x[2]}) for x in rx_graph.weighted_edge_list()]
                )
                nx_matches = networkx.max_weight_matching(nx_graph)
                rx_matches = retworkx.max_weight_matching(
                    rx_graph, weight_fn=lambda x: x, verify_optimum=True
                )
                self.compare_rx_nx_sets(rx_graph, rx_matches, nx_matches, 42 + i, nx_graph)

    def test_gnp_random_against_networkx_max_cardinality(self):
        rx_graph = retworkx.undirected_gnp_random_graph(10, 0.78, seed=428)
        nx_graph = networkx.Graph(list(rx_graph.edge_list()))
        nx_matches = networkx.max_weight_matching(nx_graph, maxcardinality=True)
        rx_matches = retworkx.max_weight_matching(
            rx_graph, max_cardinality=True, verify_optimum=True
        )
        self.compare_rx_nx_sets(rx_graph, rx_matches, nx_matches, 428, nx_graph)

    def test_gnp_random_against_networkx_with_weight_max_cardinality(self):
        for i in range(1024):
            with self.subTest(i=i):
                random.seed(i)
                rx_graph = retworkx.undirected_gnp_random_graph(10, 0.75, seed=42 + i)
                for edge in rx_graph.edge_list():
                    rx_graph.update_edge(*edge, random.randint(0, 5000))
                nx_graph = networkx.Graph(
                    [(x[0], x[1], {"weight": x[2]}) for x in rx_graph.weighted_edge_list()]
                )
                nx_matches = networkx.max_weight_matching(nx_graph, maxcardinality=True)
                rx_matches = retworkx.max_weight_matching(
                    rx_graph,
                    weight_fn=lambda x: x,
                    max_cardinality=True,
                    verify_optimum=True,
                )
                self.compare_rx_nx_sets(rx_graph, rx_matches, nx_matches, 42 + i, nx_graph)

    def test_gnp_random__networkx_with_negative_weight_max_cardinality(self):
        for i in range(1024):
            with self.subTest(i=i):
                random.seed(i)
                rx_graph = retworkx.undirected_gnp_random_graph(10, 0.75, seed=42 + i)
                for edge in rx_graph.edge_list():
                    rx_graph.update_edge(*edge, random.randint(-5000, 5000))
                nx_graph = networkx.Graph(
                    [(x[0], x[1], {"weight": x[2]}) for x in rx_graph.weighted_edge_list()]
                )
                nx_matches = networkx.max_weight_matching(nx_graph, maxcardinality=True)
                rx_matches = retworkx.max_weight_matching(
                    rx_graph,
                    weight_fn=lambda x: x,
                    max_cardinality=True,
                    verify_optimum=True,
                )
                self.compare_rx_nx_sets(rx_graph, rx_matches, nx_matches, 42 + i, nx_graph)

    def test_gnm_random_against_networkx(self):
        rx_graph = retworkx.undirected_gnm_random_graph(10, 13, seed=42)
        nx_graph = networkx.Graph(list(rx_graph.edge_list()))
        nx_matches = networkx.max_weight_matching(nx_graph)
        rx_matches = retworkx.max_weight_matching(rx_graph, verify_optimum=True)
        self.compare_rx_nx_sets(rx_graph, rx_matches, nx_matches, 42, nx_graph)

    def test_gnm_random_against_networkx_max_cardinality(self):
        rx_graph = retworkx.undirected_gnm_random_graph(10, 12, seed=42)
        nx_graph = networkx.Graph(list(rx_graph.edge_list()))
        nx_matches = networkx.max_weight_matching(nx_graph, maxcardinality=True)
        rx_matches = retworkx.max_weight_matching(
            rx_graph, max_cardinality=True, verify_optimum=True
        )
        self.compare_rx_nx_sets(rx_graph, rx_matches, nx_matches, 42, nx_graph)
