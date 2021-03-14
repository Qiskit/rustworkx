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


class TestCoreNumber(unittest.TestCase):
    def test_undirected_graph(self):
        graph = retworkx.PyGraph()
        node_a = graph.add_node(0)
        node_b = graph.add_node(1)
        node_c = graph.add_node(2)
        node_d = graph.add_node(3)
        graph.add_node(4)
        graph.add_edge(node_a, node_b, 'ab')
        graph.add_edge(node_b, node_c, 'bc')
        graph.add_edge(node_a, node_c, 'ac')
        graph.add_edge(node_c, node_d, 'cd')
        res = retworkx.core_number(graph)
        self.assertIsInstance(res, dict)
        self.assertTrue({0: 2, 1: 2, 2: 2, 3: 1, 4: 0} == res)

    def test_directed_graph(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node(0)
        node_b = graph.add_node(1)
        node_c = graph.add_node(2)
        node_d = graph.add_node(3)
        node_e = graph.add_node(4)
        graph.add_edge(node_a, node_b, 'ab')
        graph.add_edge(node_b, node_d, 'bd')
        graph.add_edge(node_a, node_c, 'ac')
        graph.add_edge(node_c, node_d, 'cd')
        graph.add_edge(node_d, node_e, 'de')
        res = retworkx.core_number(graph)
        self.assertIsInstance(res, dict)
        self.assertTrue({0: 2, 1: 2, 2: 2, 3: 2, 4: 1} == res)
