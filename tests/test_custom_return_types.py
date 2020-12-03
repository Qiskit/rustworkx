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
import pickle
import unittest

import retworkx


class TestBFSSuccessorsComparisons(unittest.TestCase):

    def setUp(self):
        self.dag = retworkx.PyDAG()
        node_a = self.dag.add_node('a')
        self.dag.add_child(node_a, 'b', "Edgy")

    def test__eq__match(self):
        self.assertTrue(retworkx.bfs_successors(self.dag, 0) == [('a', ['b'])])

    def test__eq__not_match(self):
        self.assertFalse(retworkx.bfs_successors(
            self.dag, 0) == [('b', ['c'])])

    def test_eq_not_match_inner(self):
        self.assertFalse(retworkx.bfs_successors(
            self.dag, 0) == [('a', ['c'])])

    def test__eq__different_length(self):
        self.assertFalse(retworkx.bfs_successors(
            self.dag, 0) == [('a', ['b']), ('b', ['c'])])

    def test__eq__invalid_type(self):
        with self.assertRaises(TypeError):
            retworkx.bfs_successors(self.dag, 0) == ['a']

    def test__ne__match(self):
        self.assertFalse(retworkx.bfs_successors(
            self.dag, 0) != [('a', ['b'])])

    def test__ne__not_match(self):
        self.assertTrue(retworkx.bfs_successors(self.dag, 0) != [('b', ['c'])])

    def test_ne_not_match_inner(self):
        self.assertTrue(retworkx.bfs_successors(self.dag, 0) != [('a', ['c'])])

    def test__ne__different_length(self):
        self.assertTrue(retworkx.bfs_successors(
            self.dag, 0) != [('a', ['b']), ('b', ['c'])])

    def test__ne__invalid_type(self):
        with self.assertRaises(TypeError):
            retworkx.bfs_successors(self.dag, 0) != ['a']

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            retworkx.bfs_successors(self.dag, 0) > [('b', ['c'])]

    def test_deepcopy(self):
        bfs = retworkx.bfs_successors(self.dag, 0)
        bfs_copy = copy.deepcopy(bfs)
        self.assertEqual(bfs, bfs_copy)

    def test_pickle(self):
        bfs = retworkx.bfs_successors(self.dag, 0)
        bfs_pickle = pickle.dumps(bfs)
        bfs_copy = pickle.loads(bfs_pickle)
        self.assertEqual(bfs, bfs_copy)


class TestNodeIndicesComparisons(unittest.TestCase):

    def setUp(self):
        self.dag = retworkx.PyDAG()
        node_a = self.dag.add_node('a')
        self.dag.add_child(node_a, 'b', "Edgy")

    def test__eq__match(self):
        self.assertTrue(self.dag.node_indexes() == [0, 1])

    def test__eq__not_match(self):
        self.assertFalse(self.dag.node_indexes() == [1, 2])

    def test__eq__different_length(self):
        self.assertFalse(self.dag.node_indexes() == [0, 1, 2, 3])

    def test__eq__invalid_type(self):
        with self.assertRaises(TypeError):
            self.dag.node_indexes() == ['a', None]

    def test__ne__match(self):
        self.assertFalse(self.dag.node_indexes() != [0, 1])

    def test__ne__not_match(self):
        self.assertTrue(self.dag.node_indexes() != [1, 2])

    def test__ne__different_length(self):
        self.assertTrue(self.dag.node_indexes() != [0, 1, 2, 3])

    def test__ne__invalid_type(self):
        with self.assertRaises(TypeError):
            self.dag.node_indexes() != ['a', None]

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.dag.node_indexes() > [2, 1]

    def test_deepcopy(self):
        nodes = self.dag.node_indexes()
        nodes_copy = copy.deepcopy(nodes)
        self.assertEqual(nodes, nodes_copy)

    def test_pickle(self):
        nodes = self.dag.node_indexes()
        nodes_pickle = pickle.dumps(nodes)
        nodes_copy = pickle.loads(nodes_pickle)
        self.assertEqual(nodes, nodes_copy)

class TestEdgeListComparisons(unittest.TestCase):

    def setUp(self):
        self.dag = retworkx.PyDAG()
        node_a = self.dag.add_node('a')
        self.dag.add_child(node_a, 'b', "Edgy")

    def test__eq__match(self):
        self.assertTrue(self.dag.edge_list() == [(0, 1)])

    def test__eq__not_match(self):
        self.assertFalse(self.dag.edge_list() == [(1, 2)])

    def test__eq__different_length(self):
        self.assertFalse(self.dag.edge_list() == [(0, 1), (2, 3)])

    def test__eq__invalid_type(self):
        self.assertFalse(self.dag.edge_list() == ['a', None])

    def test__ne__match(self):
        self.assertFalse(self.dag.edge_list() != [(0, 1)])

    def test__ne__not_match(self):
        self.assertTrue(self.dag.edge_list() != [(1, 2)])

    def test__ne__different_length(self):
        self.assertTrue(self.dag.edge_list() != [(0, 1), (2, 3)])

    def test__ne__invalid_type(self):
        self.assertTrue(self.dag.edge_list() != ['a', None])

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.dag.edge_list() > [(2, 1)]

    def test_deepcopy(self):
        edges = self.dag.edge_list()
        edges_copy = copy.deepcopy(edges)
        self.assertEqual(edges, edges_copy)

    def test_pickle(self):
        edges = self.dag.edge_list()
        edges_pickle = pickle.dumps(edges)
        edges_copy = pickle.loads(edges_pickle)
        self.assertEqual(edges, edges_copy)

class TestWeightedEdgeListComparisons(unittest.TestCase):

    def setUp(self):
        self.dag = retworkx.PyDAG()
        node_a = self.dag.add_node('a')
        self.dag.add_child(node_a, 'b', "Edgy")

    def test__eq__match(self):
        self.assertTrue(self.dag.weighted_edge_list() == [(0, 1, 'Edgy')])

    def test__eq__not_match(self):
        self.assertFalse(self.dag.weighted_edge_list() == [(1, 2, None)])

    def test__eq__different_length(self):
        self.assertFalse(
            self.dag.weighted_edge_list() == [
                (0, 1, 'Edgy'), (2, 3, 'Not Edgy')])

    def test__eq__invalid_type(self):
        self.assertFalse(self.dag.weighted_edge_list() == ['a', None])

    def test__ne__match(self):
        self.assertFalse(self.dag.weighted_edge_list() != [(0, 1, 'Edgy')])

    def test__ne__not_match(self):
        self.assertTrue(self.dag.weighted_edge_list() != [(1, 2, 'Not Edgy')])

    def test__ne__different_length(self):
        self.assertTrue(self.dag.node_indexes() != [0, 1, 2, 3])

    def test__ne__invalid_type(self):
        self.assertTrue(self.dag.weighted_edge_list() != ['a', None])

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.dag.weighted_edge_list() > [(2, 1, 'Not Edgy')]

    def test_deepcopy(self):
        edges = self.dag.weighted_edge_list()
        edges_copy = copy.deepcopy(edges)
        self.assertEqual(edges, edges_copy)

    def test_pickle(self):
        edges = self.dag.weighted_edge_list()
        edges_pickle = pickle.dumps(edges)
        edges_copy = pickle.loads(edges_pickle)
        self.assertEqual(edges, edges_copy)
