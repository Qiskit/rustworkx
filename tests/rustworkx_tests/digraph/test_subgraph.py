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


class TestSubgraph(unittest.TestCase):
    def test_subgraph(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([1, 3])
        self.assertEqual([(0, 1, 4)], subgraph.weighted_edge_list())
        self.assertEqual(["b", "d"], subgraph.nodes())

    def test_subgraph_empty_list(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([])
        self.assertEqual([], subgraph.weighted_edge_list())
        self.assertEqual(0, len(subgraph))

    def test_subgraph_invalid_entry(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([42])
        self.assertEqual([], subgraph.weighted_edge_list())
        self.assertEqual(0, len(subgraph))

    def test_subgraph_pass_by_reference(self):
        graph = rustworkx.PyDiGraph()
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
        graph = rustworkx.PyDiGraph()
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

    def test_edge_subgraph(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.edge_subgraph([(0, 1), (1, 3)])
        self.assertEqual(["a", "b", "d"], subgraph.nodes())
        self.assertEqual([(0, 1, 1), (1, 3, 4)], subgraph.weighted_edge_list())

    def test_edge_subgraph_parallel_edge(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.extend_from_weighted_edge_list(
            [
                (0, 1, 2),
                (0, 1, 3),
                (0, 2, 2),
                (1, 2, 4),
                (0, 3, 5),
                (2, 3, 6),
            ]
        )
        subgraph = graph.edge_subgraph([(0, 1), (1, 2)])
        self.assertEqual([0, 1, 2], subgraph.nodes())
        self.assertEqual([(0, 1, 2), (0, 1, 3), (1, 2, 4)], subgraph.weighted_edge_list())

    def test_edge_subgraph_empty_list(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.extend_from_weighted_edge_list(
            [
                (0, 1, 2),
                (0, 1, 3),
                (0, 2, 2),
                (1, 2, 4),
                (0, 3, 5),
                (2, 3, 6),
            ]
        )
        subgraph = graph.edge_subgraph([])
        self.assertEqual([], subgraph.nodes())

    def test_edge_subgraph_non_edge(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(4)))
        graph.extend_from_weighted_edge_list(
            [
                (0, 1, 2),
                (0, 1, 3),
                (0, 2, 2),
                (1, 2, 4),
                (0, 3, 5),
                (2, 3, 6),
            ]
        )
        # 1->3 isn't an edge in graph
        subgraph = graph.edge_subgraph([(0, 1), (1, 2), (1, 3)])
        self.assertEqual([0, 1, 2], subgraph.nodes())
        self.assertEqual([(0, 1, 2), (0, 1, 3), (1, 2, 4)], subgraph.weighted_edge_list())

    def test_preserve_attrs(self):
        graph = rustworkx.PyGraph(attrs="My attribute")
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([1, 3], preserve_attrs=True)
        self.assertEqual([(0, 1, 4)], subgraph.weighted_edge_list())
        self.assertEqual(["b", "d"], subgraph.nodes())
        self.assertEqual(graph.attrs, subgraph.attrs)
