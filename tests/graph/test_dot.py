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
import tempfile
import unittest

import rustworkx


class TestDot(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp()
        os.close(fd)
        os.remove(self.path)

    def test_graph_to_dot(self):
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
        expected = (
            'graph {\n0 [color=black, fillcolor=green, label="a", style=filled'
            '];\n1 [color=black, fillcolor=red, label="a", style=filled];'
            '\n0 -- 1 [label="1", name=1];\n}\n'
        )
        res = graph.to_dot(lambda node: node, lambda edge: edge)
        self.assertEqual(expected, res)

    def test_digraph_to_dot(self):
        graph = rustworkx.PyDiGraph()
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
        expected = (
            'digraph {\n0 [color=black, fillcolor=green, label="a", '
            'style=filled];\n1 [color=black, fillcolor=red, label="a", '
            'style=filled];\n0 -> 1 [label="1", name=1];\n}\n'
        )
        res = graph.to_dot(lambda node: node, lambda edge: edge)
        self.assertEqual(expected, res)

    def test_graph_to_dot_to_file(self):
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
        expected = (
            'graph {\n0 [color=black, fillcolor=green, label="a", '
            'style=filled];\n1 [color=black, fillcolor=red, label="a", '
            'style=filled];\n0 -- 1 [label="1", name=1];\n}\n'
        )
        res = graph.to_dot(lambda node: node, lambda edge: edge, filename=self.path)
        self.addCleanup(os.remove, self.path)
        self.assertIsNone(res)
        with open(self.path) as fd:
            res = fd.read()
        self.assertEqual(expected, res)

    def test_graph_empty_dicts(self):
        graph = rustworkx.undirected_gnp_random_graph(3, 0.9, seed=42)
        dot_str = graph.to_dot(lambda _: {}, lambda _: {})
        self.assertEqual(
            "graph {\n0 ;\n1 ;\n2 ;\n1 -- 0 ;\n2 -- 0 ;\n" "2 -- 1 ;\n}\n",
            dot_str,
        )

    def test_graph_graph_attrs(self):
        graph = rustworkx.undirected_gnp_random_graph(3, 0.9, seed=42)
        dot_str = graph.to_dot(lambda _: {}, lambda _: {}, {"bgcolor": "red"})
        self.assertEqual(
            "graph {\nbgcolor=red ;\n0 ;\n1 ;\n2 ;\n1 -- 0 ;\n" "2 -- 0 ;\n2 -- 1 ;\n}\n",
            dot_str,
        )

    def test_graph_no_args(self):
        graph = rustworkx.undirected_gnp_random_graph(3, 0.95, seed=24)
        dot_str = graph.to_dot()
        self.assertEqual("graph {\n0 ;\n1 ;\n2 ;\n2 -- 0 ;\n2 -- 1 ;\n}\n", dot_str)


def test_from_dot():
    expected = (
            'graph {\n0 [color=black, fillcolor=green, label="a", '
            'style=filled];\n1 [color=black, fillcolor=red, label="a", '
            'style=filled];\n0 -- 1 [label="1", name=1];\n}\n'
        )
    g = rustworkx.from_dot(expected)
    print(g)
    print(g.nodes())
    print(g.edges())

test_from_dot()
