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


class TestSubgraph(unittest.TestCase):
    def test_subgraph(self):
        graph = retworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([1, 3])
        self.assertEqual([(0, 1, 4)], subgraph.weighted_edge_list())
        self.assertEqual(["b", "d"], subgraph.nodes())

    def test_subgraph_empty_list(self):
        graph = retworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([])
        self.assertEqual([], subgraph.weighted_edge_list())
        self.assertEqual(0, len(subgraph))

    def test_subgraph_invalid_entry(self):
        graph = retworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([42])
        self.assertEqual([], subgraph.weighted_edge_list())
        self.assertEqual(0, len(subgraph))

    def test_subgraph_pass_by_reference(self):
        graph = retworkx.PyGraph()
        graph.add_node({"a": 0})
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([0, 1, 3])
        self.assertEqual([(0, 1, 1), (0, 2, 3), (1, 2, 4)], subgraph.weighted_edge_list())
        self.assertEqual([{"a": 0}, "b", "d"], subgraph.nodes())
        graph[0]["a"] = 4
        self.assertEqual(subgraph[0]["a"], 4)

    def test_subgraph_replace_weight_no_reference(self):
        graph = retworkx.PyGraph()
        graph.add_node({"a": 0})
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([0, 1, 3])
        self.assertEqual([(0, 1, 1), (0, 2, 3), (1, 2, 4)], subgraph.weighted_edge_list())
        self.assertEqual([{"a": 0}, "b", "d"], subgraph.nodes())
        graph[0] = 4
        self.assertEqual(subgraph[0]["a"], 0)
