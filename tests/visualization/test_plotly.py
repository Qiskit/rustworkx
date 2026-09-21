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

import atexit
import os
import unittest

import rustworkx
from rustworkx.visualization import plotly_draw
from rustworkx.visualization.utils import has_graphviz

try:
    import plotly.graph_objects as go
    from plotly.offline.offline import get_plotlyjs_version

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

HAS_GRAPHVIZ = has_graphviz()

SAVE_IMAGES = os.getenv("RUSTWORKX_TEST_PRESERVE_IMAGES", None)
GALLERY_PATH = "test_plotly_gallery.html"

# Collect (title, fig_html) pairs across all test classes
_gallery_entries: list[tuple[str, str]] = []


def _save_figure(fig, title):
    """Register a figure for inclusion in the combined gallery HTML."""
    _gallery_entries.append((title, fig.to_html(full_html=False, include_plotlyjs=False)))


def _write_gallery():
    """Write all collected figures to a single HTML gallery file."""
    if not _gallery_entries:
        return
    parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>',
        "<style>",
        "  body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "  h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; }",
        "  .plot { margin-bottom: 40px; }",
        "</style>",
        "</head><body>",
        f"<h1>plotly_draw gallery ({len(_gallery_entries)} plots)</h1>",
    ]
    for title, fig_html in _gallery_entries:
        parts.append(f'<div class="plot"><h2>{title}</h2>{fig_html}</div>')
    parts.append("</body></html>")

    with open(GALLERY_PATH, "w") as f:
        f.write("\n".join(parts))

    if not SAVE_IMAGES:
        try:
            os.unlink(GALLERY_PATH)
        except OSError:
            pass


atexit.register(_write_gallery)


@unittest.skipUnless(HAS_PLOTLY and HAS_GRAPHVIZ, "plotly and graphviz are required")
class TestPlotlyDraw(unittest.TestCase):
    def test_draw_no_args(self):
        graph = rustworkx.generators.star_graph(24)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "star_graph(24) — no args")

    def test_draw_directed(self):
        graph = rustworkx.generators.directed_star_graph(weights=list(range(8)))
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        # Directed graphs should have arrow markers on edge traces
        edge_traces = [t for t in fig.data if t.mode and "markers" in t.mode]
        self.assertGreater(len(edge_traces), 0)
        symbols = edge_traces[0].marker.symbol
        self.assertIn("arrow-up", symbols)
        _save_figure(fig, "directed_star_graph — arrows")

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
        _save_figure(fig, "node_attr_fn (color, size, label)")

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
        _save_figure(fig, "edge_attr_fn (color, width, label)")

    def test_draw_graph_attr(self):
        graph = rustworkx.generators.directed_star_graph(weights=list(range(5)))
        fig = plotly_draw(graph, graph_attr={"rankdir": "LR"})
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "directed_star_graph — rankdir=LR")

    def test_method(self):
        graph = rustworkx.generators.star_graph(10)
        fig = plotly_draw(graph, method="neato")
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "star_graph — method=neato")

    def test_method_sfdp(self):
        graph = rustworkx.generators.star_graph(10)
        fig = plotly_draw(graph, method="sfdp")
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "star_graph — method=sfdp")

    def test_method_invalid(self):
        graph = rustworkx.generators.star_graph(10)
        with self.assertRaises(ValueError):
            plotly_draw(graph, method="invalid")

    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "empty graph")

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node("only")
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "single node")

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
        _save_figure(fig, "all node attrs")

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
        _save_figure(fig, "all edge attrs")

    def test_custom_hover(self):
        graph = rustworkx.PyGraph()
        graph.add_node("alpha")
        graph.add_node("beta")
        graph.add_edge(0, 1, 3.14)
        fig = plotly_draw(
            graph,
            node_attr_fn=lambda node: {"hover": f"<b>{node}</b><br>custom hover"},
            edge_attr_fn=lambda edge: {"hover": f"weight: <i>{edge}</i>"},
        )
        self.assertIsInstance(fig, go.Figure)
        # Node trace is last; check its hover text
        node_trace = fig.data[-1]
        self.assertIn("<b>alpha</b><br>custom hover", node_trace.hovertext)
        # Edge trace; all points for the single edge should share the same hover
        edge_trace = fig.data[0]
        non_none = [h for h in edge_trace.hovertext if h is not None]
        self.assertTrue(all("weight:" in h for h in non_none))
        _save_figure(fig, "custom hover (HTML)")

    def test_directed_cycle(self):
        graph = rustworkx.generators.directed_cycle_graph(6)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "directed_cycle_graph(6)")

    def test_hexagonal_lattice(self):
        graph = rustworkx.generators.hexagonal_lattice_graph(3, 4)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "hexagonal_lattice_graph(3, 4)")

    def test_directed_hexagonal_lattice(self):
        graph = rustworkx.generators.directed_hexagonal_lattice_graph(3, 4)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "directed_hexagonal_lattice_graph(3, 4)")

    def test_grid_graph(self):
        graph = rustworkx.generators.grid_graph(5, 5)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "grid_graph(5, 5)")

    def test_heavy_hex_graph(self):
        graph = rustworkx.generators.heavy_hex_graph(3)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "heavy_hex_graph(3)")

    def test_heavy_square_graph(self):
        graph = rustworkx.generators.heavy_square_graph(3)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "heavy_square_graph(3)")

    def test_binomial_tree(self):
        graph = rustworkx.generators.directed_binomial_tree_graph(4)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "directed_binomial_tree_graph(4)")

    def test_petersen_graph(self):
        graph = rustworkx.generators.generalized_petersen_graph(5, 2)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "generalized_petersen_graph(5, 2)")

    def test_barbell_graph(self):
        graph = rustworkx.generators.barbell_graph(5)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "barbell_graph(5)")

    def test_gnp_random_directed(self):
        graph = rustworkx.directed_gnp_random_graph(15, 0.3, seed=42)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        _save_figure(fig, "directed_gnp_random_graph(15, 0.3)")

    def test_undirected_no_arrowheads(self):
        graph = rustworkx.generators.cycle_graph(6)
        fig = plotly_draw(graph)
        self.assertIsInstance(fig, go.Figure)
        # Undirected graphs should use lines mode (no markers for arrows)
        edge_traces = fig.data[:-1]  # all traces except the node trace
        for trace in edge_traces:
            self.assertEqual(trace.mode, "lines")
        _save_figure(fig, "cycle_graph(6) — undirected, no arrows")

    def test_graphviz_warns_on_spring_attr(self):
        graph = rustworkx.generators.star_graph(6)
        with self.assertWarns(UserWarning) as cm:
            plotly_draw(graph, spring_attr={"seed": 42})
        self.assertIn("ignored", str(cm.warning))


@unittest.skipUnless(HAS_PLOTLY, "plotly is required")
class TestPlotlyDrawSpring(unittest.TestCase):
    """Tests for spring_layout method."""

    def test_spring_layout(self):
        graph = rustworkx.generators.star_graph(6)
        fig = plotly_draw(graph, method="spring")
        self.assertIsInstance(fig, go.Figure)
        node_trace = fig.data[-1]
        self.assertEqual(len(node_trace.x), len(graph.node_indices()))
        _save_figure(fig, "star_graph(6) — method=spring")

    def test_spring_layout_directed(self):
        graph = rustworkx.generators.directed_star_graph(weights=list(range(5)))
        fig = plotly_draw(graph, method="spring")
        self.assertIsInstance(fig, go.Figure)
        # Should still have arrow markers on directed graphs
        edge_traces = [t for t in fig.data if t.mode and "markers" in t.mode]
        self.assertGreater(len(edge_traces), 0)
        symbols = edge_traces[0].marker.symbol
        self.assertIn("arrow-up", symbols)
        _save_figure(fig, "directed_star_graph — method=spring")

    def test_spring_layout_with_seed(self):
        graph = rustworkx.generators.star_graph(6)
        fig1 = plotly_draw(graph, method="spring", spring_attr={"seed": 42})
        fig2 = plotly_draw(graph, method="spring", spring_attr={"seed": 42})
        # Same seed should produce same positions
        node_trace1 = fig1.data[-1]
        node_trace2 = fig2.data[-1]
        self.assertEqual(list(node_trace1.x), list(node_trace2.x))
        self.assertEqual(list(node_trace1.y), list(node_trace2.y))

    def test_spring_warns_on_graph_attr(self):
        graph = rustworkx.generators.star_graph(6)
        with self.assertWarns(UserWarning) as cm:
            plotly_draw(graph, method="spring", graph_attr={"rankdir": "LR"})
        self.assertIn("ignored", str(cm.warning))
