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


class TestEdges(unittest.TestCase):

    def test_nodes(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', "Edgy")
        res = dag.nodes()
        self.assertEqual(['a', 'b'], res)

    def test_no_nodes(self):
        dag = retworkx.PyDAG()
        self.assertEqual([], dag.nodes())

    def test_topo_sort_empty(self):
        dag = retworkx.PyDAG()
        self.assertEqual([], retworkx.topological_sort(dag))

    def test_topo_sort(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        for i in range(5):
            dag.add_child(node_a, i, None)
        dag.add_parent(3, 'A parent', None)
        res = retworkx.topological_sort(dag)
        self.assertEqual([6, 0, 5, 4, 3, 2, 1], res)
