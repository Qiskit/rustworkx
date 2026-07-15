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

import rustworkx


class TestNeighbors(unittest.TestCase):
    def test_single_neighbor(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, {"a": 1})
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, {"a": 2})
        res = graph.neighbors(node_a)
        self.assertCountEqual([node_c, node_b], res)

    def test_unique_neighbors_on_graphs(self):
        dag = rustworkx.PyGraph()
        node_a = dag.add_node("a")
        node_b = dag.add_node("b")
        node_c = dag.add_node("c")
        dag.add_edge(node_a, node_b, ["edge a->b"])
        dag.add_edge(node_a, node_b, ["edge a->b bis"])
        dag.add_edge(node_a, node_c, ["edge a->c"])
        res = dag.neighbors(node_a)
        self.assertCountEqual([node_c, node_b], res)

    def test_no_neighbor(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        self.assertEqual([], graph.neighbors(node_a))

    def test_neighbors_deterministic_order(self):
        # Regression test for gh-1501: neighbors must be returned in a stable,
        # deterministic order (graph edge iteration order) rather than a
        # randomized HashSet order.
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(range(6))
        graph.add_edges_from_no_data([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
        expected = [5, 4, 3, 2, 1]
        for _ in range(20):
            self.assertEqual(expected, list(graph.neighbors(0)))

    def test_neighbors_dedup_deterministic_order(self):
        # Parallel edges are deduplicated, and the surviving order is stable.
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(["a", "b", "c"])
        graph.add_edge(0, 1, None)
        graph.add_edge(0, 1, None)
        graph.add_edge(0, 2, None)
        first = list(graph.neighbors(0))
        self.assertCountEqual([1, 2], first)
        self.assertEqual(2, len(first))
        for _ in range(20):
            self.assertEqual(first, list(graph.neighbors(0)))
