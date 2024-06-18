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
import subprocess
import tempfile
import unittest
import rustworkx
from rustworkx.visualization import graphviz_draw

try:
    import PIL

    subprocess.run(
        ["dot", "-V"],
        cwd=tempfile.gettempdir(),
        check=True,
        capture_output=True,
    )
    HAS_PILLOW = True
except Exception:
    HAS_PILLOW = False

SAVE_IMAGES = os.getenv("RUSTWORKX_TEST_PRESERVE_IMAGES", None)


def _save_image(image, path):
    if SAVE_IMAGES:
        image.save(path)


@unittest.skipUnless(HAS_PILLOW, "pillow and graphviz are required for running these tests")
class TestGraphvizDraw(unittest.TestCase):
    def test_draw_no_args(self):
        graph = rustworkx.generators.star_graph(24)
        image = graphviz_draw(graph)
        self.assertIsInstance(image, PIL.Image.Image)
        _save_image(image, "test_graphviz_draw.png")

    def test_draw_node_attr_fn(self):
        graph = rustworkx.PyGraph()
        graph.add_node(
            {
                "color": "black",
                "fillcolor": "green",
                "label": "a",
                "style": "filled",
            }
        )
        graph.add_node(
            {
                "color": "black",
                "fillcolor": "red",
                "label": "a",
                "style": "filled",
            }
        )
        graph.add_edge(0, 1, dict(label="1", name="1"))
        image = graphviz_draw(graph, lambda node: node)
        self.assertIsInstance(image, PIL.Image.Image)
        _save_image(image, "test_graphviz_draw_node_attr.png")

    def test_draw_edge_attr_fn(self):
        graph = rustworkx.PyGraph()
        graph.add_node(
            {
                "color": "black",
                "fillcolor": "green",
                "label": "a",
                "style": "filled",
            }
        )
        graph.add_node(
            {
                "color": "black",
                "fillcolor": "red",
                "label": "a",
                "style": "filled",
            }
        )
        graph.add_edge(0, 1, dict(label="1", name="1"))
        image = graphviz_draw(graph, lambda node: node, lambda edge: edge)
        self.assertIsInstance(image, PIL.Image.Image)
        _save_image(image, "test_graphviz_draw_edge_attr.png")

    def test_draw_graph_attr(self):
        graph = rustworkx.PyGraph()
        graph.add_node(
            {
                "color": "black",
                "fillcolor": "green",
                "label": "a",
                "style": "filled",
            }
        )
        graph.add_node(
            {
                "color": "black",
                "fillcolor": "red",
                "label": "a",
                "style": "filled",
            }
        )
        graph.add_edge(0, 1, dict(label="1", name="1"))
        graph_attr = {"bgcolor": "red"}
        image = graphviz_draw(graph, lambda node: node, lambda edge: edge, graph_attr)
        self.assertIsInstance(image, PIL.Image.Image)
        _save_image(image, "test_graphviz_draw_graph_attr.png")

    def test_image_type(self):
        graph = rustworkx.directed_gnp_random_graph(10, 0.8)
        image = graphviz_draw(graph, image_type="jpg")
        self.assertIsInstance(image, PIL.Image.Image)
        _save_image(image, "test_graphviz_draw_image_type.jpg")

    def test_image_type_invalid_type(self):
        graph = rustworkx.directed_gnp_random_graph(50, 0.8)
        with self.assertRaises(ValueError):
            graphviz_draw(graph, image_type="raw")

    def test_method(self):
        graph = rustworkx.directed_gnp_random_graph(10, 0.8)
        image = graphviz_draw(graph, method="sfdp")
        self.assertIsInstance(image, PIL.Image.Image)
        _save_image(image, "test_graphviz_method.png")

    def test_method_invalid_method(self):
        graph = rustworkx.directed_gnp_random_graph(50, 0.8)
        with self.assertRaises(ValueError):
            graphviz_draw(graph, method="special")

    def test_filename(self):
        graph = rustworkx.generators.grid_graph(20, 20)
        graphviz_draw(
            graph,
            filename="test_graphviz_filename.svg",
            image_type="svg",
            method="neato",
        )
        self.assertTrue(os.path.isfile("test_graphviz_filename.svg"))
        if not SAVE_IMAGES:
            self.addCleanup(os.remove, "test_graphviz_filename.svg")

    def test_escape_sequences(self):
        # Create a simple graph
        graph = rustworkx.generators.path_graph(2)

        escape_sequences = {
            "\\n": "\n",  # Newline
            "\\t": "\t",  # Horizontal tab
            "\\'": "'",  # Single quote
            '\\"': '"',  # Double quote
            "\\\\": "\\",  # Backslash
            "\\r": "\r",  # Carriage return
            "\\b": "\b",  # Backspace
            "\\f": "\f",  # Form feed
        }

        for escaped_seq, raw_seq in escape_sequences.items():

            def node_attr(node):
                """
                Define node attributes including escape sequences for labels and tooltips.
                """
                label = f"label{escaped_seq}"
                tooltip = f"tooltip{escaped_seq}"
                return {"label": label, "tooltip": tooltip}

            # Draw the graph using graphviz_draw
            dot_str = graph.to_dot(node_attr)

            # Assert that the escape sequences are correctly placed and escaped in the dot string
            self.assertIn(
                escaped_seq, dot_str, f"Escape sequence {escaped_seq} not found in dot output"
            )
