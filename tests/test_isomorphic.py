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


class TestIsomorphic(unittest.TestCase):

    def test_isomorphic_identical(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('a_1')
        dag_b.add_child(node_b, 'a_2', 'a_1')
        dag_b.add_child(node_b, 'a_3', 'a_2')
        self.assertTrue(retworkx.is_isomorphic(dag_a, dag_b))

    def test_isomorphic_mismatch_node_data(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('b_1')
        dag_b.add_child(node_b, 'b_2', 'b_1')
        dag_b.add_child(node_b, 'b_3', 'b_2')
        self.assertTrue(retworkx.is_isomorphic(dag_a, dag_b))

    def test_isomorphic_compare_nodes_mismatch_node_data(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('b_1')
        dag_b.add_child(node_b, 'b_2', 'b_1')
        dag_b.add_child(node_b, 'b_3', 'b_2')
        self.assertFalse(
            retworkx.is_isomorphic_node_match(
                dag_a, dag_b, lambda x, y: x == y))

    def test_isomorphic_compare_nodes_identical(self):
        dag_a = retworkx.PyDAG()
        dag_b = retworkx.PyDAG()

        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')

        node_b = dag_b.add_node('a_1')
        dag_b.add_child(node_b, 'a_2', 'a_1')
        dag_b.add_child(node_b, 'a_3', 'a_2')
        self.assertTrue(
            retworkx.is_isomorphic_node_match(
                dag_a, dag_b, lambda x, y: x == y))
