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

import os
import unittest
from unittest.mock import patch
import rustworkx
from rustworkx.visualization import plotly_draw
from rustworkx.visualization.utils import has_graphviz

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

HAS_GRAPHVIZ = has_graphviz()

SAVE_IMAGES = os.getenv("RUSTWORKX_TEST_PRESERVE_IMAGES", None)


def _save_html(fig, path):
    fig.write_html(path)
    if not SAVE_IMAGES:
        try:
            os.unlink(path)
        except OSError:
            pass


@unittest.skipUnless(HAS_PLOTLY and HAS_GRAPHVIZ, "plotly and graphviz are required")
class TestPlotlyDraw(unittest.TestCase):
    def test_draw_no_args(self):
        graph = rustworkx.generators.star_graph(24)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_draw_no_args.html")

    def test_draw_directed(self):
        graph = rustworkx.generators.directed_star_graph(weights=list(range(8)))
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        # Directed graphs should have arrow markers on edge traces
        edge_traces = [t for t in fig.data if t.mode and "markers" in t.mode]
        self.assertGreater(len(edge_traces), 0)
        symbols = edge_traces[0].marker.symbol
        self.assertIn("arrow", symbols)
        _save_html(fig, "test_draw_directed.html")

    def test_draw_node_attr_fn(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge(0, 1, None)
        fig = plotly_draw(
            graph,
            node_attr_fn=lambda node: {"color": "red", "size": 20, "label": node},
        )
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_draw_node_attr_fn.html")

    def test_draw_edge_attr_fn(self):
        graph = rustworkx.PyGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge(0, 1, "edge1")
        fig = plotly_draw(
            graph,
            edge_attr_fn=lambda edge: {"color": "blue", "width": 3, "label": edge},
        )
        self.assertIsInstance(fig, go.Figure)
        # Should have an annotation for the edge label
        self.assertEqual(len(fig.layout.annotations), 1)
        _save_html(fig, "test_draw_edge_attr_fn.html")

    def test_draw_graph_attr(self):
        graph = rustworkx.generators.directed_star_graph(weights=list(range(5)))
        fig = plotly_draw(graph, graph_attr={"rankdir": "LR"})
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_draw_graph_attr.html")

    def test_method(self):
        graph = rustworkx.generators.star_graph(10)
        fig = plotly_draw(graph, method="neato")
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_method.html")

    def test_method_sfdp(self):
        graph = rustworkx.generators.star_graph(10)
        fig = plotly_draw(graph, method="sfdp")
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_method_sfdp.html")

    def test_method_invalid(self):
        graph = rustworkx.generators.star_graph(10)
        with self.assertRaises(ValueError):
            plotly_draw(graph, method="invalid")

    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_empty_graph.html")

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node("only")
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_single_node.html")

    def test_all_node_attrs(self):
        graph = rustworkx.PyGraph()
        graph.add_node("x")
        graph.add_node("y")
        graph.add_edge(0, 1, None)
        fig = plotly_draw(
            graph,
            node_attr_fn=lambda _: {
                "color": "green",
                "size": 25,
                "label": "test",
                "symbol": "square",
                "opacity": 0.8,
                "line_color": "black",
                "line_width": 2,
            },
        )
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_all_node_attrs.html")

    def test_all_edge_attrs(self):
        graph = rustworkx.PyGraph()
        graph.add_node("x")
        graph.add_node("y")
        graph.add_edge(0, 1, "e")
        fig = plotly_draw(
            graph,
            edge_attr_fn=lambda _: {
                "color": "red",
                "width": 4,
                "label": "edge",
                "dash": "dash",
                "opacity": 0.5,
            },
        )
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_all_edge_attrs.html")

    def test_directed_cycle(self):
        graph = rustworkx.generators.directed_cycle_graph(6)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_directed_cycle.html")

    def test_hexagonal_lattice(self):
        graph = rustworkx.generators.hexagonal_lattice_graph(3, 4)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_hexagonal_lattice.html")

    def test_directed_hexagonal_lattice(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(3, 4)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_directed_hexagonal_lattice.html")

    def test_grid_graph(self):
        graph = rustworkx.generators.grid_graph(5, 5)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_grid_graph.html")

    def test_heavy_hex_graph(self):
        graph = rustworkx.generators.heavy_hex_graph(3)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_heavy_hex_graph.html")

    def test_heavy_square_graph(self):
        graph = rustworkx.generators.heavy_square_graph(3)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_heavy_square_graph.html")

    def test_binomial_tree(self):
        graph = rustworkx.generators.directed_binomial_tree_graph(4)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_binomial_tree.html")

    def test_petersen_graph(self):
        graph = rustworkx.generators.generalized_petersen_graph(5, 2)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_petersen_graph.html")

    def test_barbell_graph(self):
        graph = rustworkx.generators.barbell_graph(5)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_barbell_graph.html")

    def test_gnp_random_directed(self):
        graph = rustworkx.directed_gnp_random_graph(15, 0.3, seed=42)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_html(fig, "test_gnp_random_directed.html")

    def test_undirected_no_arrowheads(self):
        graph = rustworkx.generators.cycle_graph(6)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        # Undirected graphs should use lines mode (no markers for arrows)
        edge_traces = fig.data[:-1]  # all traces except the node trace
        for trace in edge_traces:
            self.assertEqual(trace.mode, "lines")
        _save_html(fig, "test_undirected_no_arrowheads.html")


@unittest.skipUnless(HAS_PLOTLY, "plotly is required")
class TestPlotlyDrawFallback(unittest.TestCase):
    """Tests for spring_layout fallback when graphviz is unavailable."""

    @patch("rustworkx.visualization.plotly.has_graphviz", return_value=False)
    def test_spring_fallback(self, _mock_gv):
        graph = rustworkx.generators.star_graph(6)
        with self.assertWarns(UserWarning):
            fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        # Should still produce node positions
        node_trace = fig.data[-1]
        self.assertEqual(len(node_trace.x), len(graph.node_indices()))

    @patch("rustworkx.visualization.plotly.has_graphviz", return_value=False)
    def test_spring_fallback_directed(self, _mock_gv):
        graph = rustworkx.generators.directed_star_graph(weights=list(range(5)))
        with self.assertWarns(UserWarning):
            fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        # Should still have arrow markers on directed graphs
        edge_traces = [t for t in fig.data if t.mode and "markers" in t.mode]
        self.assertGreater(len(edge_traces), 0)
        symbols = edge_traces[0].marker.symbol
        self.assertIn("arrow", symbols)

    @patch("rustworkx.visualization.plotly.has_graphviz", return_value=False)
    def test_spring_fallback_warns_on_method(self, _mock_gv):
        graph = rustworkx.generators.star_graph(6)
        with self.assertWarns(UserWarning) as cm:
            plotly_draw(graph, method="neato")
        self.assertIn("ignored", str(cm.warning))
