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
        graph = rustworkx.PyDiGraph(attrs="My attribute")
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 3, 4)])
        subgraph = graph.subgraph([1, 3], preserve_attrs=True)
        self.assertEqual([(0, 1, 4)], subgraph.weighted_edge_list())
        self.assertEqual(["b", "d"], subgraph.nodes())
        self.assertEqual(graph.attrs, subgraph.attrs)

    def test_subgraph_with_nodemap(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(6)))
        graph.add_edges_from([(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 4, 4), (4, 5, 5)])
        
        # Test basic subgraph with node mapping
        subgraph, node_map = graph.subgraph_with_nodemap([0, 2, 4])
        self.assertEqual([], subgraph.weighted_edge_list())  # No edges between disconnected nodes
        self.assertEqual([0, 2, 4], subgraph.nodes())
        self.assertEqual(dict(node_map), {0: 0, 1: 2, 2: 4})

        # Test with connected nodes
        subgraph2, node_map2 = graph.subgraph_with_nodemap([1, 2, 3])
        self.assertEqual([(0, 1, 2), (1, 2, 3)], subgraph2.weighted_edge_list())
        self.assertEqual([1, 2, 3], subgraph2.nodes())
        self.assertEqual(dict(node_map2), {0: 1, 1: 2, 2: 3})

    def test_subgraph_with_nodemap_edge_cases(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(["a", "b", "c"])
        graph.add_edges_from([(0, 1, 1), (1, 2, 2)])
        
        # Test empty node list
        subgraph, node_map = graph.subgraph_with_nodemap([])
        self.assertEqual([], subgraph.weighted_edge_list())
        self.assertEqual(0, len(subgraph))
        self.assertEqual(dict(node_map), {})
        
        # Test invalid node indices (should be silently ignored)
        subgraph, node_map = graph.subgraph_with_nodemap([42, 100])
        self.assertEqual([], subgraph.weighted_edge_list())
        self.assertEqual(0, len(subgraph))
        self.assertEqual(dict(node_map), {})
        
        # Test single node (no edges in subgraph)
        subgraph, node_map = graph.subgraph_with_nodemap([1])
        self.assertEqual([], subgraph.weighted_edge_list())
        self.assertEqual(["b"], subgraph.nodes())
        self.assertEqual(dict(node_map), {0: 1})
        
        # Test all nodes
        subgraph, node_map = graph.subgraph_with_nodemap([0, 1, 2])
        self.assertEqual([(0, 1, 1), (1, 2, 2)], subgraph.weighted_edge_list())
        self.assertEqual(["a", "b", "c"], subgraph.nodes())
        self.assertEqual(dict(node_map), {0: 0, 1: 1, 2: 2})

    def test_subgraph_with_nodemap_preserve_attrs(self):
        graph = rustworkx.PyDiGraph(attrs="test_attrs")
        graph.add_nodes_from(["a", "b", "c"])
        graph.add_edges_from([(0, 1, 1), (1, 2, 2)])
        
        # Test preserve_attrs=False (default)
        subgraph, node_map = graph.subgraph_with_nodemap([0, 1])
        self.assertIsNone(subgraph.attrs)
        
        # Test preserve_attrs=True
        subgraph2, node_map2 = graph.subgraph_with_nodemap([0, 1], preserve_attrs=True)
        self.assertEqual(graph.attrs, subgraph2.attrs)
