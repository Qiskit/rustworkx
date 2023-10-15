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

import retworkx


class TestDot(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp()
        os.close(fd)
        os.remove(self.path)

    def test_digraph_to_dot_to_file(self):
        graph = retworkx.PyDiGraph()
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
        res = graph.to_dot(lambda node: node, lambda edge: edge, filename=self.path)
        self.addCleanup(os.remove, self.path)
        self.assertIsNone(res)
        with open(self.path) as fd:
            res = fd.read()
        self.assertEqual(expected, res)

    def test_digraph_empty_dicts(self):
        graph = retworkx.directed_gnp_random_graph(3, 0.9, seed=42)
        dot_str = graph.to_dot(lambda _: {}, lambda _: {})
        self.assertEqual("digraph {\n0 ;\n1 ;\n2 ;\n0 -> 1 ;\n0 -> 2 ;\n}\n", dot_str)

    def test_digraph_graph_attrs(self):
        graph = retworkx.directed_gnp_random_graph(3, 0.9, seed=42)
        dot_str = graph.to_dot(lambda _: {}, lambda _: {}, {"bgcolor": "red"})
        self.assertEqual(
            "digraph {\nbgcolor=red ;\n0 ;\n1 ;\n2 ;\n0 -> 1 ;\n" "0 -> 2 ;\n}\n",
            dot_str,
        )

    def test_digraph_no_args(self):
        graph = retworkx.directed_gnp_random_graph(3, 0.95, seed=24)
        dot_str = graph.to_dot()
        self.assertEqual("digraph {\n0 ;\n1 ;\n2 ;\n0 -> 1 ;\n0 -> 2 ;\n}\n", dot_str)
