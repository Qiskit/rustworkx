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

import rustworkx as rx
import networkx as nx


class TestImmediateDominators(unittest.TestCase):
    """Test `rustworkx.immediate_dominators`.

    Test cases adapted from `networkx`:
    https://github.com/networkx/networkx/blob/9c5ca54b7e5310a21568bb2e0104f8c87bf74ff7/networkx/algorithms/tests/test_dominance.py
    (Copyright 2004-2024 NetworkX Developers, 3-clause BSD License)
    """

    def test_empty(self):
        """
        Edge case: empty graph.
        """
        graph = rx.PyDiGraph()

        with self.assertRaises(rx.NullGraph):
            rx.immediate_dominators(graph, 0)

    def test_start_node_not_in_graph(self):
        """
        Edge case: start_node is not in the graph.
        """
        graph = rx.PyDiGraph()
        graph.add_node(0)

        self.assertEqual(list(graph.node_indices()), [0])

        with self.assertRaises(rx.InvalidNode):
            rx.immediate_dominators(graph, 1)

    def test_singleton(self):
        """
        Edge cases: single node, optionally cyclic.
        """
        graph = rx.PyDiGraph()
        graph.add_node(0)
        self.assertDictEqual(rx.immediate_dominators(graph, 0), {0: 0})
        graph.add_edge(0, 0, None)
        self.assertDictEqual(rx.immediate_dominators(graph, 0), {0: 0})

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(graph.edge_list())
        # subset check
        self.assertGreaterEqual(
            rx.immediate_dominators(graph, 0).items(), nx.immediate_dominators(nx_graph, 0).items()
        )

    def test_irreducible1(self):
        """
        Graph taken from figure 2 of "A simple, fast dominance algorithm." (2006).
        https://hdl.handle.net/1911/96345
        """
        edges = [(1, 2), (2, 1), (3, 2), (4, 1), (5, 3), (5, 4)]
        graph = rx.PyDiGraph()
        graph.add_node(0)
        graph.extend_from_edge_list(edges)

        result = rx.immediate_dominators(graph, 5)
        self.assertDictEqual(result, {i: 5 for i in range(1, 6)})

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(graph.edge_list())
        # subset check
        self.assertGreaterEqual(result.items(), nx.immediate_dominators(nx_graph, 5).items())

    def test_irreducible2(self):
        """
        Graph taken from figure 4 of "A simple, fast dominance algorithm." (2006).
        https://hdl.handle.net/1911/96345
        """
        edges = [(1, 2), (2, 1), (2, 3), (3, 2), (4, 2), (4, 3), (5, 1), (6, 4), (6, 5)]
        graph = rx.PyDiGraph()
        graph.add_node(0)
        graph.extend_from_edge_list(edges)

        result = rx.immediate_dominators(graph, 6)
        self.assertDictEqual(result, {i: 6 for i in range(1, 7)})

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(graph.edge_list())
        # subset check
        self.assertGreaterEqual(result.items(), nx.immediate_dominators(nx_graph, 6).items())

    def test_domrel_png(self):
        """
        Graph taken from https://commons.wikipedia.org/wiki/File:Domrel.png
        """
        edges = [(1, 2), (2, 3), (2, 4), (2, 6), (3, 5), (4, 5), (5, 2)]
        graph = rx.PyDiGraph()
        graph.add_node(0)
        graph.extend_from_edge_list(edges)

        result = rx.immediate_dominators(graph, 1)
        self.assertDictEqual(result, {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2})

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(graph.edge_list())
        self.assertDictEqual(nx.immediate_dominators(nx_graph, 1), result)

        # Test postdominance.
        graph.reverse()
        result = rx.immediate_dominators(graph, 6)
        self.assertDictEqual(result, {1: 2, 2: 6, 3: 5, 4: 5, 5: 2, 6: 6})
        # subset check
        self.assertGreaterEqual(
            result.items(), nx.immediate_dominators(nx_graph.reverse(copy=False), 6).items()
        )

    def test_boost_example(self):
        """
        Graph taken from Figure 1 of
        http://www.boost.org/doc/libs/1_56_0/libs/graph/doc/lengauer_tarjan_dominator.htm
        """
        edges = [(0, 1), (1, 2), (1, 3), (2, 7), (3, 4), (4, 5), (4, 6), (5, 7), (6, 4)]
        graph = rx.PyDiGraph()
        graph.extend_from_edge_list(edges)
        result = rx.immediate_dominators(graph, 0)
        self.assertDictEqual(result, {0: 0, 1: 0, 2: 1, 3: 1, 4: 3, 5: 4, 6: 4, 7: 1})

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(graph.edge_list())
        # subset check
        self.assertGreaterEqual(result.items(), nx.immediate_dominators(nx_graph, 0).items())

        # Test postdominance.
        graph.reverse()
        result = rx.immediate_dominators(graph, 7)
        self.assertDictEqual(result, {0: 1, 1: 7, 2: 7, 3: 4, 4: 5, 5: 7, 6: 4, 7: 7})
        # subset check
        self.assertGreaterEqual(
            result.items(), nx.immediate_dominators(nx_graph.reverse(copy=False), 7).items()
        )


class TestDominanceFrontiers(unittest.TestCase):
    """
    Test `rustworkx.dominance_frontiers`.

    Test cases adapted from `networkx`:
    https://github.com/networkx/networkx/blob/9c5ca54b7e5310a21568bb2e0104f8c87bf74ff7/networkx/algorithms/tests/test_dominance.py
    (Copyright 2004-2024 NetworkX Developers, 3-clause BSD License)
    """

    def test_empty(self):
        """
        Edge case: empty graph.
        """
        graph = rx.PyDiGraph()

        with self.assertRaises(rx.NullGraph):
            rx.dominance_frontiers(graph, 0)

    def test_start_node_not_in_graph(self):
        """
        Edge case: start_node is not in the graph.
        """
        graph = rx.PyDiGraph()
        graph.add_node(0)

        self.assertEqual(list(graph.node_indices()), [0])

        with self.assertRaises(rx.InvalidNode):
            rx.dominance_frontiers(graph, 1)

    def test_singleton(self):
        """
        Edge cases: single node, optionally cyclic.
        """
        graph = rx.PyDiGraph()
        graph.add_node(0)
        self.assertDictEqual(rx.dominance_frontiers(graph, 0), {0: set()})

        graph.add_edge(0, 0, None)
        self.assertDictEqual(rx.dominance_frontiers(graph, 0), {0: set()})

    def test_irreducible1(self):
        """
        Graph taken from figure 2 of "A simple, fast dominance algorithm." (2006).
        https://hdl.handle.net/1911/96345
        """
        edges = [(1, 2), (2, 1), (3, 2), (4, 1), (5, 3), (5, 4)]
        graph = rx.PyDiGraph()
        graph.add_node(0)
        graph.extend_from_edge_list(edges)

        result = rx.dominance_frontiers(graph, 5)
        self.assertDictEqual(result, {1: {2}, 2: {1}, 3: {2}, 4: {1}, 5: set()})

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(graph.edge_list())
        self.assertDictEqual(nx.dominance_frontiers(nx_graph, 5), result)

    def test_irreducible2(self):
        """
        Graph taken from figure 4 of "A simple, fast dominance algorithm." (2006).
        https://hdl.handle.net/1911/96345
        """
        edges = [(1, 2), (2, 1), (2, 3), (3, 2), (4, 2), (4, 3), (5, 1), (6, 4), (6, 5)]
        graph = rx.PyDiGraph()
        graph.add_node(0)
        graph.extend_from_edge_list(edges)

        result = rx.dominance_frontiers(graph, 6)

        self.assertDictEqual(
            result,
            {
                1: {2},
                2: {1, 3},
                3: {2},
                4: {2, 3},
                5: {1},
                6: set(),
            },
        )

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(graph.edge_list())
        self.assertDictEqual(nx.dominance_frontiers(nx_graph, 6), result)

    def test_domrel_png(self):
        """
        Graph taken from https://commons.wikipedia.org/wiki/File:Domrel.png
        """
        edges = [(1, 2), (2, 3), (2, 4), (2, 6), (3, 5), (4, 5), (5, 2)]
        graph = rx.PyDiGraph()
        graph.add_node(0)
        graph.extend_from_edge_list(edges)

        result = rx.dominance_frontiers(graph, 1)

        self.assertDictEqual(
            result,
            {
                1: set(),
                2: {2},
                3: {5},
                4: {5},
                5: {2},
                6: set(),
            },
        )

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(graph.edge_list())
        self.assertDictEqual(nx.dominance_frontiers(nx_graph, 1), result)

        # Test postdominance.
        graph.reverse()
        result = rx.dominance_frontiers(graph, 6)
        self.assertDictEqual(
            result,
            {
                1: set(),
                2: {2},
                3: {2},
                4: {2},
                5: {2},
                6: set(),
            },
        )

        self.assertDictEqual(nx.dominance_frontiers(nx_graph.reverse(copy=False), 6), result)

    def test_boost_example(self):
        """
        Graph taken from Figure 1 of
        http://www.boost.org/doc/libs/1_56_0/libs/graph/doc/lengauer_tarjan_dominator.htm
        """
        edges = [(0, 1), (1, 2), (1, 3), (2, 7), (3, 4), (4, 5), (4, 6), (5, 7), (6, 4)]
        graph = rx.PyDiGraph()
        graph.extend_from_edge_list(edges)

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(graph.edge_list())

        result = rx.dominance_frontiers(graph, 0)
        self.assertDictEqual(
            result,
            {
                0: set(),
                1: set(),
                2: {7},
                3: {7},
                4: {4, 7},
                5: {7},
                6: {4},
                7: set(),
            },
        )

        self.assertDictEqual(nx.dominance_frontiers(nx_graph, 0), result)

        # Test postdominance
        graph.reverse()
        result = rx.dominance_frontiers(graph, 7)
        self.assertDictEqual(
            result,
            {
                0: set(),
                1: set(),
                2: {1},
                3: {1},
                4: {1, 4},
                5: {1},
                6: {4},
                7: set(),
            },
        )

        self.assertDictEqual(nx.dominance_frontiers(nx_graph.reverse(copy=False), 7), result)

    def test_missing_immediate_doms(self):
        """
        Test that the `dominance_frontiers` function doesn't regress on
        https://github.com/networkx/networkx/issues/2070
        """
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 3)]
        graph = rx.PyDiGraph()
        graph.extend_from_edge_list(edges)

        idom = rx.immediate_dominators(graph, 0)
        self.assertNotIn(5, idom)

        # In networkx#2070, the call would fail because node 5
        # has no immediate dominators
        result = rx.dominance_frontiers(graph, 0)
        self.assertDictEqual(
            result,
            {
                0: set(),
                1: set(),
                2: set(),
                3: set(),
                4: set(),
                5: {3},
            },
        )
