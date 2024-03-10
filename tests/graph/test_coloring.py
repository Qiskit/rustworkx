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


class TestGraphColoring(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.graph_greedy_color(graph)
        self.assertEqual({}, res)

    def test_simple_graph(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 1)
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, 1)
        res = rustworkx.graph_greedy_color(graph)
        self.assertEqual({0: 0, 1: 1, 2: 1}, res)

    def test_simple_graph_large_degree(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 1)
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        res = rustworkx.graph_greedy_color(graph)
        self.assertEqual({0: 0, 1: 1, 2: 1}, res)

    def test_simple_graph_with_preset(self):
        def preset(node_idx):
            if node_idx == 0:
                return 1
            return None

        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 1)
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, 1)
        res = rustworkx.graph_greedy_color(graph, preset)
        self.assertEqual({0: 1, 1: 0, 2: 0}, res)

    def test_simple_graph_large_degree_with_preset(self):
        def preset(node_idx):
            if node_idx == 0:
                return 1
            return None

        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 1)
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_c, 1)
        res = rustworkx.graph_greedy_color(graph, preset)
        self.assertEqual({0: 1, 1: 0, 2: 0}, res)

    def test_preset_raises_exception(self):
        def preset(node_idx):
            raise OverflowError("I am invalid")

        graph = rustworkx.generators.path_graph(5)
        with self.assertRaises(OverflowError):
            rustworkx.graph_greedy_color(graph, preset)

    def test_simple_graph_with_strategy(self):
        graph = rustworkx.PyGraph()
        [a, b, c, d, e, f, g, h] = graph.add_nodes_from(["a", "b", "c", "d", "e", "f", "g", "h"])
        graph.add_edges_from(
            [(a, b, 1), (a, c, 1), (a, d, 1), (d, e, 1), (e, f, 1), (f, g, 1), (f, h, 1)]
        )

        with self.subTest():
            res = rustworkx.graph_greedy_color(graph)
            self.assertEqual({a: 0, b: 1, c: 1, d: 1, e: 2, f: 0, g: 1, h: 1}, res)

        with self.subTest(greedy_strategy=rustworkx.GreedyStrategy.Saturation):
            res = rustworkx.graph_greedy_color(
                graph, greedy_strategy=rustworkx.GreedyStrategy.Saturation
            )
            self.assertEqual({a: 0, b: 1, c: 1, d: 1, e: 0, f: 1, g: 0, h: 0}, res)


class TestGraphEdgeColoring(unittest.TestCase):
    def test_graph(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        node_e = graph.add_node("e")

        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_a, node_d, 1)
        graph.add_edge(node_d, node_e, 1)

        edge_colors = rustworkx.graph_greedy_edge_color(graph)
        self.assertEqual({0: 1, 1: 2, 2: 0, 3: 1}, edge_colors)

    def test_graph_with_holes(self):
        """Graph with missing node and edge indices."""
        graph = rustworkx.PyGraph()

        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        node_e = graph.add_node("e")

        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_b, node_c, 1)
        graph.add_edge(node_c, node_d, 1)
        graph.add_edge(node_d, node_e, 1)

        graph.remove_node(node_c)

        edge_colors = rustworkx.graph_greedy_edge_color(graph)
        self.assertEqual({0: 0, 3: 0}, edge_colors)

    def test_graph_without_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        edge_colors = rustworkx.graph_greedy_edge_color(graph)
        self.assertEqual({}, edge_colors)

    def test_graph_multiple_edges(self):
        """Graph with multiple edges between two nodes."""
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_a, node_b, 1)
        edge_colors = rustworkx.graph_greedy_edge_color(graph)
        self.assertEqual({0: 0, 1: 1, 2: 2, 3: 3}, edge_colors)

    def test_cycle_graph(self):
        graph = rustworkx.generators.cycle_graph(7)
        edge_colors = rustworkx.graph_greedy_edge_color(graph)
        self.assertEqual({0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 2}, edge_colors)


class TestMisraGriesColoring(unittest.TestCase):
    def test_simple_graph(self):
        graph = rustworkx.PyGraph()
        node0 = graph.add_node(0)
        node1 = graph.add_node(1)
        node2 = graph.add_node(2)
        node3 = graph.add_node(3)

        graph.add_edge(node0, node1, 1)
        graph.add_edge(node0, node2, 1)
        graph.add_edge(node1, node2, 1)
        graph.add_edge(node2, node3, 1)

        edge_colors = rustworkx.graph_misra_gries_edge_color(graph)
        self.assertEqual({0: 2, 1: 1, 2: 0, 3: 2}, edge_colors)

    def test_graph_with_holes(self):
        """Graph with missing node and edge indices."""
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        node_e = graph.add_node("e")
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_b, node_c, 1)
        graph.add_edge(node_c, node_d, 1)
        graph.add_edge(node_d, node_e, 1)
        graph.remove_node(node_c)
        edge_colors = rustworkx.graph_misra_gries_edge_color(graph)
        self.assertEqual({0: 0, 3: 0}, edge_colors)

    def test_graph_without_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        edge_colors = rustworkx.graph_misra_gries_edge_color(graph)
        self.assertEqual({}, edge_colors)

    def test_graph_multiple_edges(self):
        """Graph with multiple edges between two nodes."""
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_a, node_b, 1)
        edge_colors = rustworkx.graph_misra_gries_edge_color(graph)
        self.assertEqual({0: 0, 1: 1, 2: 2, 3: 3}, edge_colors)

    def test_cycle_graph(self):
        """Test on a small cycle graph with an odd number of virtices."""
        graph = rustworkx.generators.cycle_graph(7)
        edge_colors = rustworkx.graph_misra_gries_edge_color(graph)
        assert edge_colors == {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 0, 6: 2}

    def test_grid(self):
        """Test that Misra-Gries colors the grid with at most 5 colors (max degree + 1)."""
        graph = rustworkx.generators.grid_graph(10, 10)
        edge_colors = rustworkx.graph_misra_gries_edge_color(graph)
        num_colors = max(edge_colors.values()) + 1
        self.assertLessEqual(num_colors, 5)

    def test_heavy_hex(self):
        """Test that Misra-Gries colors the heavy hex with at most 4 colors (max degree + 1)."""
        graph = rustworkx.generators.heavy_hex_graph(9)
        edge_colors = rustworkx.graph_misra_gries_edge_color(graph)
        num_colors = max(edge_colors.values()) + 1
        self.assertLessEqual(num_colors, 4)

    def test_complete_graph(self):
        """Test that Misra-Gries colors the complete graph with at most n+1 colors."""
        graph = rustworkx.generators.complete_graph(10)
        edge_colors = rustworkx.graph_misra_gries_edge_color(graph)
        num_colors = max(edge_colors.values()) + 1
        self.assertLessEqual(num_colors, 11)


class TestBipartiteGraphEdgeColoring(unittest.TestCase):
    def test_graph(self):
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")

        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_b, node_c, 1)
        graph.add_edge(node_c, node_d, 1)
        graph.add_edge(node_a, node_d, 1)

        edge_colors = rustworkx.graph_bipartite_edge_color(graph)
        self.assertEqual({0: 1, 1: 0, 2: 1, 3: 0}, edge_colors)

    def test_graph_multiple_edges(self):
        """Graph with multiple edges between two nodes."""
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_b, node_a, 1)
        graph.add_edge(node_a, node_b, 1)
        edge_colors = rustworkx.graph_bipartite_edge_color(graph)
        self.assertEqual({0: 3, 1: 2, 2: 1, 3: 0}, edge_colors)

    def test_graph_not_bipartite(self):
        """Test that assert is raised on non-bipartite graphs."""
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_a, node_c, 1)
        graph.add_edge(node_b, node_c, 1)
        with self.assertRaises(rustworkx.GraphNotBipartite):
            rustworkx.graph_bipartite_edge_color(graph)

    def test_graph_not_bipartite_self_loop(self):
        """Test that assert is raised on non-bipartite graphs."""
        graph = rustworkx.PyGraph()
        node_a = graph.add_node("a")
        graph.add_edge(node_a, node_a, 1)
        with self.assertRaises(rustworkx.GraphNotBipartite):
            rustworkx.graph_bipartite_edge_color(graph)

    def test_graph_with_holes(self):
        """Graph with missing node and edge indices."""
        graph = rustworkx.PyGraph()

        node_a = graph.add_node("a")
        node_b = graph.add_node("b")
        node_c = graph.add_node("c")
        node_d = graph.add_node("d")
        node_e = graph.add_node("e")

        graph.add_edge(node_a, node_b, 1)
        graph.add_edge(node_b, node_c, 1)
        graph.add_edge(node_c, node_d, 1)
        graph.add_edge(node_d, node_e, 1)

        graph.remove_node(node_c)

        edge_colors = rustworkx.graph_bipartite_edge_color(graph)
        self.assertEqual({0: 0, 3: 0}, edge_colors)

    def test_graph_without_edges(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        edge_colors = rustworkx.rustworkx.graph_bipartite_edge_color(graph)
        self.assertEqual({}, edge_colors)

    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        edge_colors = rustworkx.rustworkx.graph_bipartite_edge_color(graph)
        self.assertEqual({}, edge_colors)

    def test_cycle_graph(self):
        graph = rustworkx.generators.cycle_graph(8)
        edge_colors = rustworkx.rustworkx.graph_bipartite_edge_color(graph)
        self.assertEqual({0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}, edge_colors)

    def test_heavy_hex_graph(self):
        """Test that we color the heavy hex with exactly 3 colors (it's bipartite
        and has max degree 3)."""
        graph = rustworkx.generators.heavy_hex_graph(9)
        edge_colors = rustworkx.rustworkx.graph_bipartite_edge_color(graph)
        num_colors = max(edge_colors.values()) + 1
        self.assertEqual(num_colors, 3)
