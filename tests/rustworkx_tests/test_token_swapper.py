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

from numpy import random


def swap_permutation(
    mapping,
    swaps,
) -> None:
    for (sw1, sw2) in list(swaps):
        val1 = mapping.pop(sw1, None)
        val2 = mapping.pop(sw2, None)

        if val1 is not None:
            mapping[sw2] = val1
        if val2 is not None:
            mapping[sw1] = val2


class TestGeneral(unittest.TestCase):
    """The test cases"""

    def setUp(self) -> None:
        """Set up test cases."""
        super().setUp()
        random.seed(0)

    def test_simple(self) -> None:
        """Test a simple permutation on a path graph of size 4."""
        graph = rx.generators.path_graph(4)
        permutation = {0: 0, 1: 3, 3: 1, 2: 2}
        swaps = rx.graph_token_swapper(graph, permutation, 4, 4, 1)
        swap_permutation(permutation, swaps)
        self.assertEqual(3, len(swaps))
        self.assertEqual({i: i for i in range(4)}, permutation)

    def test_small(self) -> None:
        """Test an inverting permutation on a small path graph of size 8"""
        graph = rx.generators.path_graph(8)
        permutation = {i: 7 - i for i in range(8)}
        swaps = rx.graph_token_swapper(graph, permutation, 4, 4, 1)
        swap_permutation(permutation, swaps)
        self.assertEqual({i: i for i in range(8)}, permutation)

    def test_bug1(self) -> None:
        """Tests for a bug that occured in happy swap chains of length >2."""
        graph = rx.PyGraph()
        graph.extend_from_edge_list(
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (3, 6)]
        )
        permutation = {0: 4, 1: 0, 2: 3, 3: 6, 4: 2, 6: 1}
        swaps = rx.graph_token_swapper(graph, permutation, 4, 4, 1)
        swap_permutation(permutation, swaps)
        self.assertEqual({i: i for i in permutation}, permutation)

    def test_partial_simple(self) -> None:
        """Test a partial mapping on a small graph."""
        graph = rx.generators.path_graph(4)
        mapping = {0: 3}
        swaps = rx.graph_token_swapper(graph, mapping, 4, 4, 10)
        swap_permutation(mapping, swaps)
        self.assertEqual(3, len(swaps))
        self.assertEqual({3: 3}, mapping)

    def test_partial_simple_remove_node(self) -> None:
        """Test a partial mapping on a small graph with a node removed."""
        graph = rx.generators.path_graph(5)
        graph.remove_node(2)
        graph.add_edge(1, 3, None)
        mapping = {0: 3}
        swaps = rx.graph_token_swapper(graph, mapping, 4, 4, 10)
        swap_permutation(mapping, swaps)
        self.assertEqual(2, len(swaps))
        self.assertEqual({3: 3}, mapping)

    def test_partial_small(self) -> None:
        """Test an partial inverting permutation on a small path graph of size 5"""
        graph = rx.generators.path_graph(4)
        permutation = {i: 3 - i for i in range(2)}
        swaps = rx.graph_token_swapper(graph, permutation, 4, 4, 10)
        swap_permutation(permutation, swaps)
        self.assertEqual(5, len(swaps))
        self.assertEqual({i: i for i in permutation.values()}, permutation)

    def test_large_partial_random(self) -> None:
        """Test a random (partial) mapping on a large randomly generated graph"""
        size = 100
        # Note that graph may have "gaps" in the node counts, i.e. the numbering is noncontiguous.
        graph = rx.undirected_gnm_random_graph(size, size**2 // 10)
        for i in graph.node_indexes():
            try:
                graph.remove_edge(i, i)  # Remove self-loops.
            except rx.NoEdgeBetweenNodes:
                continue
        # Make sure the graph is connected by adding C_n
        graph.add_edges_from_no_data([(i, i + 1) for i in range(len(graph) - 1)])

        # Generate a randomized permutation.
        rand_perm = random.permutation(graph.nodes())
        permutation = dict(zip(graph.nodes(), rand_perm))
        mapping = dict(itertools.islice(permutation.items(), 0, size, 2))  # Drop every 2nd element.
        swaps = rx.graph_token_swapper(graph, permutation, 4, 4)
        swap_permutation(mapping, swaps)
        self.assertEqual({i: i for i in mapping.values()}, mapping)

    def test_disjoint_graph(self):
        graph = rx.PyGraph()
        graph.extend_from_edge_list([(0, 1), (2, 3)])
        swaps = rx.graph_token_swapper(graph, {1: 0, 0: 1, 2: 3, 3: 2}, 10, seed=42)
        self.assertEqual(len(swaps), 2)
        with self.assertRaises(rx.InvalidMapping):
            rx.graph_token_swapper(graph, {2: 0, 1: 1, 0: 2, 3: 3}, 10)
