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


class TestNumShortestpath(unittest.TestCase):
    def test_num_shortest_path_unweighted(self):
        graph = retworkx.PyDiGraph()
        node_a = graph.add_node(0)
        node_b = graph.add_node('end')
        for i in range(3):
            node = graph.add_child(node_a, i, None)
            graph.add_edge(node, node_b, None)
        res = retworkx.digraph_num_shortest_paths_unweighted(graph, node_a)
        expected = {2: 1, 4: 1, 3: 1, 1: 3}
        self.assertEqual(expected, res)
