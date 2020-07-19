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


class TestDot(unittest.TestCase):
    def test_graph_to_dot(self):
        graph = retworkx.PyGraph()
        graph.add_node({'color': 'black', 'fillcolor': 'green',
                        'label': "a", 'style': 'filled'})
        graph.add_node({'color': 'black', 'fillcolor': 'red',
                        'label': "a", 'style': 'filled'})
        graph.add_edge(0, 1, dict(label='1', name='1'))
        expected = (
            "graph {\n0 [color=black, fillcolor=green, label=a, style=filled"
            "];\n1 [color=black, fillcolor=red, label=a, style=filled];"
            "\n0 -- 1 [label=1, name=1];\n}\n")
        res = graph.to_dot(lambda node: node, lambda edge: edge)
        self.assertEqual(expected, res.decode('utf8'))

    def test_digraph_to_dot(self):
        graph = retworkx.PyDiGraph()
        graph.add_node({'color': 'black', 'fillcolor': 'green',
                        'label': "a", 'style': 'filled'})
        graph.add_node({'color': 'black', 'fillcolor': 'red',
                        'label': "a", 'style': 'filled'})
        graph.add_edge(0, 1, dict(label='1', name='1'))
        expected = (
            "digraph {\n0 [color=black, fillcolor=green, label=a, style=filled"
            "];\n1 [color=black, fillcolor=red, label=a, style=filled];"
            "\n0 -> 1 [label=1, name=1];\n}\n")
        res = graph.to_dot(lambda node: node, lambda edge: edge)
        self.assertEqual(expected, res.decode('utf8'))
