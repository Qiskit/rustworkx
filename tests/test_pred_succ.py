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


class TestPredecessors(unittest.TestCase):
    def test_single_predecessor(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_a, 'c', {'a': 2})
        res = dag.predecessors(node_c)
        self.assertEqual(['a'], res)

    def test_single_predecessor_multiple_edges(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_a, 'c', {'a': 2})
        dag.add_edge(node_a, node_c, {'a': 3})
        res = dag.predecessors(node_c)
        self.assertEqual(['a'], res)

    def test_many_parents(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        for i in range(10):
            dag.add_parent(node_a, {'numeral': i}, {'edge': i})
        res = dag.predecessors(node_a)
        self.assertEqual([{'numeral': 9}, {'numeral': 8}, {'numeral': 7},
                          {'numeral': 6}, {'numeral': 5}, {'numeral': 4},
                          {'numeral': 3}, {'numeral': 2}, {'numeral': 1},
                          {'numeral': 0}], res)

class TestSuccessors(unittest.TestCase):
    def test_single_successor(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_b, 'c', {'a': 2})
        dag.add_child(node_c, 'd', {'a': 1})
        res = dag.successors(node_b)
        self.assertEqual(['c'], res)

    def test_single_successor_multiple_edges(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_b, 'c', {'a': 2})
        dag.add_child(node_c, 'd', {'a': 1})
        dag.add_edge(node_b, node_c, {'a': 3})
        res = dag.successors(node_b)
        self.assertEqual(['c'], res)

    def test_many_children(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        for i in range(10):
            dag.add_child(node_a, {'numeral': i}, {'edge': i})
        res = dag.successors(node_a)
        self.assertEqual([{'numeral': 9}, {'numeral': 8}, {'numeral': 7},
                          {'numeral': 6}, {'numeral': 5}, {'numeral': 4},
                          {'numeral': 3}, {'numeral': 2}, {'numeral': 1},
                          {'numeral': 0}], res)

class TestSuccessors(unittest.TestCase):
    def test_single_successor(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_b, 'c', {'a': 2})
        dag.add_child(node_c, 'd', {'a': 1})
        res = retworkx.bfs_successors(dag, node_b)
        self.assertEqual([('b', ['c']), ('c', ['d'])], res)

    def test_many_children(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        for i in range(10):
            dag.add_child(node_a, {'numeral': i}, {'edge': i})
        res = retworkx.bfs_successors(dag, node_a)
        self.assertEqual([('a', [{'numeral': 9}, {'numeral': 8}, {'numeral': 7},
                          {'numeral': 6}, {'numeral': 5}, {'numeral': 4},
                          {'numeral': 3}, {'numeral': 2}, {'numeral': 1},
                          {'numeral': 0}])], res)

    def test_bfs_succesors(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node(0)
        node_b = dag.add_child(node_a, 1, {})
        node_c = dag.add_child(node_b, 2, {})
        node_d = dag.add_child(node_c, 3, {})
        node_e = dag.add_child(node_d, 4, {})
        node_f = dag.add_child(node_e, 5, {})
        dag.add_child(node_f, 6, {})
        node_h = dag.add_child(node_c, 7, {})
        node_i = dag.add_child(node_h, 8, {})
        node_j = dag.add_child(node_i, 9, {})
        dag.add_child(node_j, 10, {})
        res = {n: sorted(s) for n, s in retworkx.bfs_successors(dag, node_b)}
        expected = {
            1: [2],
            2: [3, 7],
            3: [4],
            4: [5],
            5: [6],
            7: [8],
            8: [9],
            9: [10]
        }
        self.assertEqual(expected, res)
        self.assertEqual([(7, [8]), (8, [9]), (9, [10])],
                         retworkx.bfs_successors(dag, node_h))
