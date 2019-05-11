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


class TestAdj(unittest.TestCase):
    def test_single_neighbor(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_a, 'c', {'a': 2})
        res = dag.adj(node_a)
        self.assertEqual({node_b: {'a': 1}, node_c: {'a': 2}}, res)

    def test_single_neighbor_dir(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_a, 'c', {'a': 2})
        res = dag.adj_direction(node_a, False)
        self.assertEqual({node_b: {'a': 1}, node_c: {'a': 2}}, res)
        res = dag.adj_direction(node_a, True)
        self.assertEqual({}, res)

    def test_in_direction(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        for i in range(5):
            dag.add_parent(node_a, i, None)
        self.assertEqual(5, dag.in_degree(node_a))

    def test_in_direction_none(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        for i in range(5):
            dag.add_child(node_a, i, None)
        self.assertEqual(0, dag.in_degree(node_a))
