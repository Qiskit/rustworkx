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


class TestLongestPath(unittest.TestCase):
    def test_linear(self):
        """Longest depth for a simple dag.

        a
        |
        b
        |\
        c d
        |
        e
        |\
        f g
        """
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        node_c = dag.add_child(node_b, 'c', {})
        dag.add_child(node_b, 'd', {})
        node_e = dag.add_child(node_c, 'e', {})
        dag.add_child(node_e, 'f', {})
        dag.add_child(node_e, 'g', {})
        self.assertEqual(4, retworkx.dag_longest_path_length(dag))

    def test_less_linear(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        node_c = dag.add_child(node_b, 'c', {})
        node_d = dag.add_child(node_c, 'd', {})
        node_e = dag.add_child(node_d, 'e', {})
        dag.add_edge(node_a, node_c, {})
        dag.add_edge(node_a, node_e, {})
        dag.add_edge(node_c, node_e, {})
        self.assertEqual(4, retworkx.dag_longest_path_length(dag))

    def test_degenerate_graph(self):
        dag = retworkx.PyDAG()
        dag.add_node(0)
        self.assertEqual(1, retworkx.dag_longest_path_length(dag))

    def test_empty_graph(self):
        dag = retworkx.PyDAG()
        self.assertEqual(0, retworkx.dag_longest_path_length(dag))
