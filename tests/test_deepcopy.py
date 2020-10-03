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

import copy
import unittest

import retworkx


class TestDeepcopy(unittest.TestCase):

    def test_isomorphic_compare_nodes_identical(self):
        dag_a = retworkx.PyDAG()
        node_a = dag_a.add_node('a_1')
        dag_a.add_child(node_a, 'a_2', 'a_1')
        dag_a.add_child(node_a, 'a_3', 'a_2')
        dag_b = copy.deepcopy(dag_a)
        self.assertTrue(
            retworkx.is_isomorphic_node_match(
                dag_a, dag_b, lambda x, y: x == y))

    def test_deepcopy_with_holes(self):
        dag_a = retworkx.PyDAG()
        node_a = dag_a.add_node('a_1')
        node_b = dag_a.add_node('a_2')
        dag_a.add_edge(node_a, node_b, 'edge_1')
        node_c = dag_a.add_node('a_3')
        dag_a.add_edge(node_b, node_c, 'edge_2')
        dag_a.remove_node(node_b)
        dag_b = copy.deepcopy(dag_a)
        self.assertIsInstance(dag_b, retworkx.PyDAG)
        self.assertEqual([node_a, node_c], dag_b.node_indexes())
