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


class TestCompoes(unittest.TestCase):
    def test_single_dag_composition(self):
        dag = retworkx.PyDAG()
        dag.check_cycle = True
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_b, 'c', {'a': 2})
        dag_other = retworkx.PyDAG()
        node_d = dag_other.add_node('d')
        dag_other.add_child(node_d, 'e', {'a': 3})
        res = dag.compose(dag_other, {node_c: (node_d, {'b': 1})})
        res = retworkx.topological_sort(dag)
        self.assertEqual([0, 1, 2, 3, 4], res)
