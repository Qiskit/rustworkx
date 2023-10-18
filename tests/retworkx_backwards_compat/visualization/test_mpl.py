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

# Based on the equivalent tests for the networkx matplotlib drawer:
# https://github.com/networkx/networkx/blob/ead0e65bda59862e329f2e6f1da47919c6b07ca9/networkx/drawing/tests/test_pylab.py

import os
import unittest

import retworkx
from retworkx.visualization import mpl_draw

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("PS")
    plt.rcParams["text.usetex"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

SAVE_IMAGES = os.getenv("RUSTWORKX_TEST_PRESERVE_IMAGES", None)


def _save_images(fig, path):
    fig.savefig(path, dpi=400)
    if not SAVE_IMAGES:
        try:
            os.unlink(path)
        except OSError:
            pass


@unittest.skipUnless(HAS_MPL, "matplotlib is required for running these tests")
class TestMPLDraw(unittest.TestCase):
    def test_draw(self):
        graph = retworkx.generators.star_graph(24)
        options = {"node_color": "black", "node_size": 100, "width": 3}
        fig = mpl_draw(graph, **options)
        _save_images(fig, "test.png")

    def test_node_list(self):
        graph = retworkx.generators.star_graph(24)
        node_list = list(range(4)) + list(range(4, 10)) + list(range(10, 14))
        fig = mpl_draw(graph, node_list=node_list)
        _save_images(fig, "test_node_list.png")

    def test_edge_colormap(self):
        graph = retworkx.generators.star_graph(24)
        colors = range(len(graph.edge_list()))
        fig = mpl_draw(
            graph,
            edge_color=colors,
            width=4,
            edge_cmap=plt.cm.Blues,
            with_labels=True,
        )
        _save_images(fig, "test_edge_colors.png")

    def test_arrows(self):
        graph = retworkx.generators.directed_star_graph(24)
        fig = mpl_draw(graph)
        _save_images(fig, "test_arrows.png")

    def test_empty_graph(self):
        graph = retworkx.PyGraph()
        fig = mpl_draw(graph)
        _save_images(fig, "test_empty.png")

    def test_axes(self):
        fig, ax = plt.subplots()
        graph = retworkx.directed_gnp_random_graph(50, 0.75)
        mpl_draw(graph, ax=ax)
        _save_images(fig, "test_axes.png")

    def test_selfloop_with_single_edge_in_edge_list(self):
        fig, ax = plt.subplots()
        # Graph with selfloop
        graph = retworkx.generators.path_graph(2)
        graph.add_edge(1, 1, None)
        pos = {n: (n, n) for n in graph.node_indexes()}
        mpl_draw(graph, pos, ax=ax, edge_list=[(1, 1)])
        _save_images(fig, "test_self_loop.png")

    def test_draw_edges_min_source_target_margins(self):
        """Test that there is a wider gap between the node and the start of an
        incident edge when min_source_margin is specified.

        This test checks that the use of min_{source/target}_margin kwargs
        result in shorter (more padding) between the edges and source and
        target nodes. As a crude visual example, let 's' and 't' represent
        source and target nodes, respectively:
           Default:
           s-----------------------------t
           With margins:
           s   -----------------------   t
        """
        node_shapes = ["o", "s"]
        graph = retworkx.PyGraph()
        graph.extend_from_edge_list([(0, 1)])
        pos = {0: (0, 0), 1: (1, 0)}  # horizontal layout

        for node_shape in node_shapes:
            with self.subTest(shape=node_shape):
                fig, ax = plt.subplots()
                mpl_draw(
                    graph,
                    pos=pos,
                    ax=ax,
                    node_shape=node_shape,
                    min_source_margin=100,
                    min_target_margin=100,
                )
                _save_images(fig, "test_node_shape_%s.png" % node_shape)

    def test_alpha_iter(self):
        graph = retworkx.generators.grid_graph(4, 6)
        # with fewer alpha elements than nodes
        plt.subplot(131)
        mpl_draw(graph, alpha=[0.1, 0.2])
        # with equal alpha elements and nodes
        num_nodes = len(graph)
        alpha = [x / num_nodes for x in range(num_nodes)]
        colors = range(num_nodes)
        plt.subplot(132)
        mpl_draw(graph, node_color=colors, alpha=alpha)
        # with more alpha elements than nodes
        alpha.append(1)
        plt.subplot(133)
        mpl_draw(graph, alpha=alpha)
        fig = plt.gcf()
        _save_images(fig, "test_alpha_iter.png")

    def test_labels_and_colors(self):
        graph = retworkx.PyGraph()
        graph.add_nodes_from(list(range(8)))
        edge_list = [
            (0, 1, 5),
            (1, 2, 2),
            (2, 3, 7),
            (3, 0, 6),
            (5, 6, 1),
            (4, 5, 7),
            (6, 7, 3),
            (7, 4, 7),
        ]
        labels = {}
        labels[0] = r"$a$"
        labels[1] = r"$b$"
        labels[2] = r"$c$"
        labels[3] = r"$d$"
        labels[4] = r"$\alpha$"
        labels[5] = r"$\beta$"
        labels[6] = r"$\gamma$"
        labels[7] = r"$\delta$"
        graph.add_edges_from(edge_list)
        pos = retworkx.random_layout(graph)
        mpl_draw(
            graph,
            pos=pos,
            node_list=[0, 1, 2, 3],
            node_color="r",
            edge_list=[(0, 1), (1, 2), (2, 3), (3, 0)],
            node_size=500,
            alpha=0.75,
            width=1.0,
            labels=lambda x: labels[x],
            font_size=16,
        )
        mpl_draw(
            graph,
            pos=pos,
            node_list=[4, 5, 6, 7],
            node_color="b",
            node_size=500,
            alpha=0.5,
            edge_list=[(4, 5), (5, 6), (6, 7), (7, 4)],
            width=8,
            edge_color="r",
            rotate=False,
            edge_labels=lambda edge: labels[edge],
        )
        fig = plt.gcf()
        _save_images(fig, "test_labels_and_colors.png")
