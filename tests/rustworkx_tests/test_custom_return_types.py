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

import rustworkx
import numpy as np


class TestBFSSuccessorsComparisons(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        self.node_a = self.dag.add_node("a")
        self.node_b = self.dag.add_child(self.node_a, "b", "Edgy")

    def test__eq__match(self):
        self.assertTrue(rustworkx.bfs_successors(self.dag, 0) == [("a", ["b"])])

    def test__eq__not_match(self):
        self.assertFalse(rustworkx.bfs_successors(self.dag, 0) == [("b", ["c"])])

    def test_eq_not_match_inner(self):
        self.assertFalse(rustworkx.bfs_successors(self.dag, 0) == [("a", ["c"])])

    def test__eq__different_length(self):
        self.assertFalse(rustworkx.bfs_successors(self.dag, 0) == [("a", ["b"]), ("b", ["c"])])

    def test__eq__invalid_type(self):
        with self.assertRaises(TypeError):
            rustworkx.bfs_successors(self.dag, 0) == ["a"]

    def test__ne__match(self):
        self.assertFalse(rustworkx.bfs_successors(self.dag, 0) != [("a", ["b"])])

    def test__ne__not_match(self):
        self.assertTrue(rustworkx.bfs_successors(self.dag, 0) != [("b", ["c"])])

    def test_ne_not_match_inner(self):
        self.assertTrue(rustworkx.bfs_successors(self.dag, 0) != [("a", ["c"])])

    def test__ne__different_length(self):
        self.assertTrue(rustworkx.bfs_successors(self.dag, 0) != [("a", ["b"]), ("b", ["c"])])

    def test__ne__invalid_type(self):
        with self.assertRaises(TypeError):
            rustworkx.bfs_successors(self.dag, 0) != ["a"]

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            rustworkx.bfs_successors(self.dag, 0) > [("b", ["c"])]

    def test_deepcopy(self):
        bfs = rustworkx.bfs_successors(self.dag, 0)
        bfs_copy = copy.deepcopy(bfs)
        self.assertEqual(bfs, bfs_copy)

    def test_pickle(self):
        bfs = rustworkx.bfs_successors(self.dag, 0)
        bfs_pickle = pickle.dumps(bfs)
        bfs_copy = pickle.loads(bfs_pickle)
        self.assertEqual(bfs, bfs_copy)

    def test_str(self):
        res = rustworkx.bfs_successors(self.dag, 0)
        self.assertEqual("BFSSuccessors[(a, [b])]", str(res))

    def test_hash(self):
        res = rustworkx.bfs_successors(self.dag, 0)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_hash_invalid_type(self):
        self.dag.add_child(0, [1, 2, 3], "edgy")
        res = rustworkx.bfs_successors(self.dag, 0)
        with self.assertRaises(TypeError):
            hash(res)

    def test_slices(self):
        self.dag.add_child(self.node_a, "c", "New edge")
        self.dag.add_child(self.node_b, "d", "New edge to d")
        successors = rustworkx.bfs_successors(self.dag, 0)
        slice_return = successors[0:3:2]
        self.assertEqual([("a", ["c", "b"])], slice_return)


class TestNodeIndicesComparisons(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")

    def test__eq__match(self):
        self.assertTrue(self.dag.node_indexes() == [0, 1])

    def test__eq__not_match(self):
        self.assertFalse(self.dag.node_indexes() == [1, 2])

    def test__eq__different_length(self):
        self.assertFalse(self.dag.node_indexes() == [0, 1, 2, 3])

    def test__eq__invalid_type(self):
        with self.assertRaises(TypeError):
            self.dag.node_indexes() == ["a", None]

    def test__ne__match(self):
        self.assertFalse(self.dag.node_indexes() != [0, 1])

    def test__ne__not_match(self):
        self.assertTrue(self.dag.node_indexes() != [1, 2])

    def test__ne__different_length(self):
        self.assertTrue(self.dag.node_indexes() != [0, 1, 2, 3])

    def test__ne__invalid_type(self):
        with self.assertRaises(TypeError):
            self.dag.node_indexes() != ["a", None]

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

    def test_str(self):
        res = self.dag.node_indexes()
        self.assertEqual("NodeIndices[0, 1]", str(res))

    def test_hash(self):
        res = self.dag.node_indexes()
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_slices(self):
        self.dag.add_node("new")
        self.dag.add_node("fun")
        nodes = self.dag.node_indices()
        slice_return = nodes[0:3:2]
        self.assertEqual([0, 2], slice_return)
        self.assertEqual(nodes[0:-1], [0, 1, 2])

    def test_slices_negatives(self):
        graph = rustworkx.PyGraph()
        graph.add_nodes_from(range(5))
        indices = graph.node_indices()
        slice_return = indices[-1:-3:-1]
        self.assertEqual([4, 3], slice_return)
        slice_return = indices[3:1:-2]
        self.assertEqual([3], slice_return)
        slice_return = indices[-3:-1]
        self.assertEqual([2, 3], slice_return)
        self.assertEqual([], indices[-1:-2])

    def test_reversed(self):
        indices = self.dag.node_indices()
        reversed_slice = indices[::-1]
        reversed_elems = list(reversed(indices))
        self.assertEqual(reversed_slice, reversed_elems)

    def test_numpy_conversion(self):
        res = self.dag.node_indexes()
        np.testing.assert_array_equal(np.asarray(res, dtype=np.uintp), np.array([0, 1]))


class TestNodesCountMapping(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")

    def test__eq__match(self):
        self.assertTrue(rustworkx.num_shortest_paths_unweighted(self.dag, 0) == {1: 1})

    def test__eq__not_match_keys(self):
        self.assertFalse(rustworkx.num_shortest_paths_unweighted(self.dag, 0) == {2: 1})

    def test__eq__not_match_values(self):
        self.assertFalse(rustworkx.num_shortest_paths_unweighted(self.dag, 0) == {1: 2})

    def test__eq__different_length(self):
        self.assertFalse(rustworkx.num_shortest_paths_unweighted(self.dag, 0) == {1: 1, 2: 2})

    def test_eq__same_type(self):
        self.assertEqual(
            rustworkx.num_shortest_paths_unweighted(self.dag, 0),
            rustworkx.num_shortest_paths_unweighted(self.dag, 0),
        )

    def test__eq__invalid_type(self):
        self.assertFalse(rustworkx.num_shortest_paths_unweighted(self.dag, 0) == ["a", None])

    def test__eq__invalid_inner_type(self):
        self.assertFalse(rustworkx.num_shortest_paths_unweighted(self.dag, 0) == {0: "a"})

    def test__ne__match(self):
        self.assertFalse(rustworkx.num_shortest_paths_unweighted(self.dag, 0) != {1: 1})

    def test__ne__not_match(self):
        self.assertTrue(rustworkx.num_shortest_paths_unweighted(self.dag, 0) != {2: 1})

    def test__ne__not_match_values(self):
        self.assertTrue(rustworkx.num_shortest_paths_unweighted(self.dag, 0) != {1: 2})

    def test__ne__different_length(self):
        self.assertTrue(rustworkx.num_shortest_paths_unweighted(self.dag, 0) != {1: 1, 2: 2})

    def test__ne__invalid_type(self):
        self.assertTrue(rustworkx.num_shortest_paths_unweighted(self.dag, 0) != ["a", None])

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            rustworkx.num_shortest_paths_unweighted(self.dag, 0) > {1: 1}

    def test_deepcopy(self):
        paths = rustworkx.num_shortest_paths_unweighted(self.dag, 0)
        paths_copy = copy.deepcopy(paths)
        self.assertEqual(paths, paths_copy)

    def test_pickle(self):
        paths = rustworkx.num_shortest_paths_unweighted(self.dag, 0)
        paths_pickle = pickle.dumps(paths)
        paths_copy = pickle.loads(paths_pickle)
        self.assertEqual(paths, paths_copy)

    def test_str(self):
        res = rustworkx.num_shortest_paths_unweighted(self.dag, 0)
        self.assertEqual("NodesCountMapping{1: 1}", str(res))

    def test_hash(self):
        res = rustworkx.num_shortest_paths_unweighted(self.dag, 0)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = rustworkx.num_shortest_paths_unweighted(self.dag, 0)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = rustworkx.num_shortest_paths_unweighted(self.dag, 0).keys()
        self.assertEqual([1], list(keys))

    def test_values(self):
        values = rustworkx.num_shortest_paths_unweighted(self.dag, 0).values()
        self.assertEqual([1], list(values))

    def test_items(self):
        items = rustworkx.num_shortest_paths_unweighted(self.dag, 0).items()
        self.assertEqual([(1, 1)], list(items))

    def test_iter(self):
        mapping_iter = iter(rustworkx.num_shortest_paths_unweighted(self.dag, 0))
        output = list(mapping_iter)
        self.assertEqual(output, [1])

    def test_contains(self):
        res = rustworkx.num_shortest_paths_unweighted(self.dag, 0)
        self.assertIn(1, res)

    def test_not_contains(self):
        res = rustworkx.num_shortest_paths_unweighted(self.dag, 0)
        self.assertNotIn(0, res)


class TestEdgeIndicesComparisons(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDiGraph()
        node_a = self.dag.add_node("a")
        node_b = self.dag.add_child(node_a, "b", "Edgy")
        self.dag.add_child(node_b, "c", "Super Edgy")

    def test__eq__match(self):
        self.assertTrue(self.dag.edge_indices() == [0, 1])

    def test__eq__not_match(self):
        self.assertFalse(self.dag.edge_indices() == [1, 2])

    def test__eq__different_length(self):
        self.assertFalse(self.dag.edge_indices() == [0, 1, 2, 3])

    def test__eq__invalid_type(self):
        with self.assertRaises(TypeError):
            self.dag.edge_indices() == ["a", None]

    def test__ne__match(self):
        self.assertFalse(self.dag.edge_indices() != [0, 1])

    def test__ne__not_match(self):
        self.assertTrue(self.dag.edge_indices() != [1, 2])

    def test__ne__different_length(self):
        self.assertTrue(self.dag.edge_indices() != [0, 1, 2, 3])

    def test__ne__invalid_type(self):
        with self.assertRaises(TypeError):
            self.dag.edge_indices() != ["a", None]

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.dag.edge_indices() > [2, 1]

    def test_deepcopy(self):
        edges = self.dag.edge_indices()
        edges_copy = copy.deepcopy(edges)
        self.assertEqual(edges, edges_copy)

    def test_pickle(self):
        edges = self.dag.edge_indices()
        edges_pickle = pickle.dumps(edges)
        edges_copy = pickle.loads(edges_pickle)
        self.assertEqual(edges, edges_copy)

    def test_str(self):
        res = self.dag.edge_indices()
        self.assertEqual("EdgeIndices[0, 1]", str(res))

    def test_hash(self):
        res = self.dag.edge_indices()
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_slices(self):
        self.dag.add_edge(0, 1, None)
        edges = self.dag.edge_indices()
        slice_return = edges[0:-1]
        self.assertEqual([0, 1], slice_return)


class TestEdgeListComparisons(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")

    def test__eq__match(self):
        self.assertTrue(self.dag.edge_list() == [(0, 1)])

    def test__eq__not_match(self):
        self.assertFalse(self.dag.edge_list() == [(1, 2)])

    def test__eq__different_length(self):
        self.assertFalse(self.dag.edge_list() == [(0, 1), (2, 3)])

    def test__eq__invalid_type(self):
        self.assertFalse(self.dag.edge_list() == ["a", None])

    def test__ne__match(self):
        self.assertFalse(self.dag.edge_list() != [(0, 1)])

    def test__ne__not_match(self):
        self.assertTrue(self.dag.edge_list() != [(1, 2)])

    def test__ne__different_length(self):
        self.assertTrue(self.dag.edge_list() != [(0, 1), (2, 3)])

    def test__ne__invalid_type(self):
        self.assertTrue(self.dag.edge_list() != ["a", None])

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

    def test_str(self):
        res = self.dag.edge_list()
        self.assertEqual("EdgeList[(0, 1)]", str(res))

    def test_hash(self):
        res = self.dag.edge_list()
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_slice(self):
        self.dag.add_edge(0, 1, None)
        self.dag.add_edge(0, 1, None)
        edges = self.dag.edge_list()
        slice_return = edges[0:3:2]
        self.assertEqual([(0, 1), (0, 1)], slice_return)

    @staticmethod
    def test_numpy_conversion():
        g = rustworkx.generators.directed_star_graph(5)
        res = g.edge_list()

        np.testing.assert_array_equal(
            np.asarray(res, dtype=np.uintp), np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        )


class TestWeightedEdgeListComparisons(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")

    def test__eq__match(self):
        self.assertTrue(self.dag.weighted_edge_list() == [(0, 1, "Edgy")])

    def test__eq__not_match(self):
        self.assertFalse(self.dag.weighted_edge_list() == [(1, 2, None)])

    def test__eq__different_length(self):
        self.assertFalse(self.dag.weighted_edge_list() == [(0, 1, "Edgy"), (2, 3, "Not Edgy")])

    def test__eq__invalid_type(self):
        self.assertFalse(self.dag.weighted_edge_list() == ["a", None])

    def test__ne__match(self):
        self.assertFalse(self.dag.weighted_edge_list() != [(0, 1, "Edgy")])

    def test__ne__not_match(self):
        self.assertTrue(self.dag.weighted_edge_list() != [(1, 2, "Not Edgy")])

    def test__ne__different_length(self):
        self.assertTrue(self.dag.node_indexes() != [0, 1, 2, 3])

    def test__ne__invalid_type(self):
        self.assertTrue(self.dag.weighted_edge_list() != ["a", None])

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.dag.weighted_edge_list() > [(2, 1, "Not Edgy")]

    def test_deepcopy(self):
        edges = self.dag.weighted_edge_list()
        edges_copy = copy.deepcopy(edges)
        self.assertEqual(edges, edges_copy)

    def test_pickle(self):
        edges = self.dag.weighted_edge_list()
        edges_pickle = pickle.dumps(edges)
        edges_copy = pickle.loads(edges_pickle)
        self.assertEqual(edges, edges_copy)

    def test_str(self):
        res = self.dag.weighted_edge_list()
        self.assertEqual("WeightedEdgeList[(0, 1, Edgy)]", str(res))

    def test_hash(self):
        res = self.dag.weighted_edge_list()
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_hash_invalid_type(self):
        self.dag.add_child(0, "c", ["edgy", "not_edgy"])
        res = self.dag.weighted_edge_list()
        with self.assertRaises(TypeError):
            hash(res)

    def test_slice(self):
        self.dag.add_edge(0, 1, None)
        self.dag.add_edge(0, 1, None)
        edges = self.dag.weighted_edge_list()
        slice_return = edges[0:3:2]
        self.assertEqual([(0, 1, "Edgy"), (0, 1, None)], slice_return)

    def test_numpy_conversion(self):
        np.testing.assert_array_equal(
            np.asarray(self.dag.weighted_edge_list()), np.array([(0, 1, "Edgy")], dtype=object)
        )


class TestPathMapping(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")

    def test__eq__match(self):
        self.assertTrue(rustworkx.dijkstra_shortest_paths(self.dag, 0) == {1: [0, 1]})

    def test__eq__not_match_keys(self):
        self.assertFalse(rustworkx.dijkstra_shortest_paths(self.dag, 0) == {2: [0, 1]})

    def test__eq__not_match_values(self):
        self.assertFalse(rustworkx.dijkstra_shortest_paths(self.dag, 0) == {1: [0, 2]})

    def test__eq__different_length(self):
        self.assertFalse(rustworkx.dijkstra_shortest_paths(self.dag, 0) == {1: [0, 1], 2: [0, 2]})

    def test_eq__same_type(self):
        self.assertEqual(
            rustworkx.dijkstra_shortest_paths(self.dag, 0),
            rustworkx.dijkstra_shortest_paths(self.dag, 0),
        )

    def test__eq__invalid_type(self):
        self.assertFalse(rustworkx.dijkstra_shortest_paths(self.dag, 0) == ["a", None])

    def test__eq__invalid_inner_type(self):
        self.assertFalse(rustworkx.dijkstra_shortest_paths(self.dag, 0) == {0: {"a": None}})

    def test__ne__match(self):
        self.assertFalse(rustworkx.dijkstra_shortest_paths(self.dag, 0) != {1: [0, 1]})

    def test__ne__not_match(self):
        self.assertTrue(rustworkx.dijkstra_shortest_paths(self.dag, 0) != {2: [0, 1]})

    def test__ne__not_match_values(self):
        self.assertTrue(rustworkx.dijkstra_shortest_paths(self.dag, 0) != {1: [0, 2]})

    def test__ne__different_length(self):
        self.assertTrue(rustworkx.dijkstra_shortest_paths(self.dag, 0) != {1: [0, 1], 2: [0, 2]})

    def test__ne__invalid_type(self):
        self.assertTrue(rustworkx.dijkstra_shortest_paths(self.dag, 0) != ["a", None])

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            rustworkx.dijkstra_shortest_paths(self.dag, 0) > {1: [0, 2]}

    def test_deepcopy(self):
        paths = rustworkx.dijkstra_shortest_paths(self.dag, 0)
        paths_copy = copy.deepcopy(paths)
        self.assertEqual(paths, paths_copy)

    def test_pickle(self):
        paths = rustworkx.dijkstra_shortest_paths(self.dag, 0)
        paths_pickle = pickle.dumps(paths)
        paths_copy = pickle.loads(paths_pickle)
        self.assertEqual(paths, paths_copy)

    def test_str(self):
        res = rustworkx.dijkstra_shortest_paths(self.dag, 0)
        self.assertEqual("PathMapping{1: [0, 1]}", str(res))

    def test_hash(self):
        res = rustworkx.dijkstra_shortest_paths(self.dag, 0)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = rustworkx.dijkstra_shortest_paths(self.dag, 0)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = rustworkx.dijkstra_shortest_paths(self.dag, 0).keys()
        self.assertEqual([1], list(keys))

    def test_values(self):
        values = rustworkx.dijkstra_shortest_paths(self.dag, 0).values()
        self.assertEqual([[0, 1]], list(values))

    def test_items(self):
        items = rustworkx.dijkstra_shortest_paths(self.dag, 0).items()
        self.assertEqual([(1, [0, 1])], list(items))

    def test_iter(self):
        mapping_iter = iter(rustworkx.dijkstra_shortest_paths(self.dag, 0))
        output = list(mapping_iter)
        self.assertEqual(output, [1])

    def test_contains(self):
        res = rustworkx.dijkstra_shortest_paths(self.dag, 0)
        self.assertIn(1, res)

    def test_not_contains(self):
        res = rustworkx.dijkstra_shortest_paths(self.dag, 0)
        self.assertNotIn(0, res)


class TestPathLengthMapping(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")
        self.fn = lambda _: 1.0

    def test__eq__match(self):
        self.assertTrue(rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) == {1: 1.0})

    def test__eq__not_match_keys(self):
        self.assertFalse(rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) == {2: 1.0})

    def test__eq__not_match_values(self):
        self.assertFalse(rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) == {1: 2.0})

    def test__eq__different_length(self):
        self.assertFalse(
            rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) == {1: 1.0, 2: 2.0}
        )

    def test_eq__same_type(self):
        self.assertEqual(
            rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn),
            rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn),
        )

    def test__eq__invalid_type(self):
        self.assertFalse(
            rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) == ["a", None]
        )

    def test__eq__invalid_inner_type(self):
        self.assertFalse(rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) == {0: "a"})

    def test__ne__match(self):
        self.assertFalse(rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) != {1: 1.0})

    def test__ne__not_match(self):
        self.assertTrue(rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) != {2: 1.0})

    def test__ne__not_match_values(self):
        self.assertTrue(rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) != {1: 2.0})

    def test__ne__different_length(self):
        self.assertTrue(
            rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) != {1: 1.0, 2: 2.0}
        )

    def test__ne__invalid_type(self):
        self.assertTrue(
            rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) != ["a", None]
        )

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) > {1: 1.0}

    def test_deepcopy(self):
        paths = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        paths_copy = copy.deepcopy(paths)
        self.assertEqual(paths, paths_copy)

    def test_pickle(self):
        paths = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        paths_pickle = pickle.dumps(paths)
        paths_copy = pickle.loads(paths_pickle)
        self.assertEqual(paths, paths_copy)

    def test_str(self):
        res = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, lambda _: 3.14)
        self.assertEqual("PathLengthMapping{1: 3.14}", str(res))

    def test_hash(self):
        res = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn).keys()
        self.assertEqual([1], list(keys))

    def test_values(self):
        values = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn).values()
        self.assertEqual([1.0], list(values))

    def test_items(self):
        items = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn).items()
        self.assertEqual([(1, 1.0)], list(items))

    def test_iter(self):
        mapping_iter = iter(rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn))
        output = list(mapping_iter)
        self.assertEqual(output, [1])

    def test_contains(self):
        res = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        self.assertIn(1, res)

    def test_not_contains(self):
        res = rustworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        self.assertNotIn(0, res)


class TestPos2DMapping(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDiGraph()
        self.dag.add_node("a")

    def test__eq__match(self):
        res = rustworkx.random_layout(self.dag, seed=10244242)
        self.assertTrue(res == {0: (0.4883489113112722, 0.6545867364101975)})

    def test__eq__not_match_keys(self):
        self.assertFalse(rustworkx.random_layout(self.dag, seed=10244242) == {2: 1.0})

    def test__eq__not_match_values(self):
        self.assertFalse(rustworkx.random_layout(self.dag, seed=10244242) == {1: 2.0})

    def test__eq__different_length(self):
        res = rustworkx.random_layout(self.dag, seed=10244242)
        self.assertFalse(res == {1: 1.0, 2: 2.0})

    def test_eq__same_type(self):
        self.assertEqual(
            rustworkx.random_layout(self.dag, seed=10244242),
            rustworkx.random_layout(self.dag, seed=10244242),
        )

    def test__eq__invalid_type(self):
        self.assertFalse(rustworkx.random_layout(self.dag, seed=10244242) == {"a": None})

    def test__ne__match(self):
        res = rustworkx.random_layout(self.dag, seed=10244242)
        self.assertFalse(res != {0: (0.4883489113112722, 0.6545867364101975)})

    def test__ne__not_match(self):
        self.assertTrue(rustworkx.random_layout(self.dag, seed=10244242) != {2: 1.0})

    def test__ne__not_match_values(self):
        self.assertTrue(rustworkx.random_layout(self.dag, seed=10244242) != {1: 2.0})

    def test__ne__different_length(self):
        res = rustworkx.random_layout(self.dag, seed=10244242)

        self.assertTrue(res != {1: 1.0, 2: 2.0})

    def test__ne__invalid_type(self):
        self.assertTrue(rustworkx.random_layout(self.dag, seed=10244242) != ["a", None])

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            rustworkx.random_layout(self.dag, seed=10244242) > {1: 1.0}

    def test_deepcopy(self):
        positions = rustworkx.random_layout(self.dag)
        positions_copy = copy.deepcopy(positions)
        self.assertEqual(positions_copy, positions)

    def test_pickle(self):
        pos = rustworkx.random_layout(self.dag)
        pos_pickle = pickle.dumps(pos)
        pos_copy = pickle.loads(pos_pickle)
        self.assertEqual(pos, pos_copy)

    def test_str(self):
        res = rustworkx.random_layout(self.dag, seed=10244242)
        self.assertEqual(
            "Pos2DMapping{0: [0.4883489113112722, 0.6545867364101975]}",
            str(res),
        )

    def test_hash(self):
        res = rustworkx.random_layout(self.dag, seed=10244242)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = rustworkx.random_layout(self.dag, seed=10244242)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = rustworkx.random_layout(self.dag, seed=10244242).keys()
        self.assertEqual([0], list(keys))

    def test_values(self):
        values = rustworkx.random_layout(self.dag, seed=10244242).values()
        expected = [[0.4883489113112722, 0.6545867364101975]]
        self.assertEqual(expected, list(values))

    def test_items(self):
        items = rustworkx.random_layout(self.dag, seed=10244242).items()
        self.assertEqual([(0, [0.4883489113112722, 0.6545867364101975])], list(items))

    def test_iter(self):
        mapping_iter = iter(rustworkx.random_layout(self.dag, seed=10244242))
        output = list(mapping_iter)
        self.assertEqual(output, [0])

    def test_contains(self):
        res = rustworkx.random_layout(self.dag, seed=10244242)
        self.assertIn(0, res)

    def test_not_contains(self):
        res = rustworkx.random_layout(self.dag, seed=10244242)
        self.assertNotIn(1, res)


class TestEdgeIndices(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDiGraph()
        self.dag.add_node("a")
        self.dag.add_child(0, "b", "edge")

    def test__eq__match(self):
        res = self.dag.edge_index_map()
        self.assertTrue(res == {0: (0, 1, "edge")})

    def test__eq__not_match_keys(self):
        res = self.dag.edge_index_map()
        self.assertFalse(res == {2: (0, 1, "edge")})

    def test__eq__not_match_values(self):
        res = self.dag.edge_index_map()
        self.assertFalse(res == {0: (1, 2, "edge")})
        self.assertFalse(res == {0: (0, 1, "not edge")})

    def test__eq__different_length(self):
        res = self.dag.edge_index_map()
        self.assertFalse(res == {1: (0, 1, "edge"), 0: (0, 1, "double edge")})

    def test_eq__same_type(self):
        self.assertEqual(self.dag.edge_index_map(), self.dag.edge_index_map())

    def test__eq__invalid_type(self):
        res = self.dag.edge_index_map()
        self.assertFalse(res == {"a": ("a", "b", "c")})

    def test__ne__match(self):
        res = self.dag.edge_index_map()
        self.assertFalse(res != {0: (0, 1, "edge")})

    def test__ne__not_match(self):
        res = self.dag.edge_index_map()
        self.assertTrue(res, {2: (0, 1, "edge")})

    def test__ne__not_match_values(self):
        res = self.dag.edge_index_map()
        self.assertTrue(res, {0: (0, 2, "edge")})

    def test__ne__different_length(self):
        res = self.dag.edge_index_map()
        self.assertTrue(res != {1: (0, 1, "double edge"), 0: (0, 1, "edge")})

    def test__ne__invalid_type(self):
        res = self.dag.edge_index_map()
        self.assertTrue(res != {"a": ("a", "b", "c")})

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.dag.edge_index_map() > {0: (0, 1, "edge")}

    def test_deepcopy(self):
        edge_map = self.dag.edge_index_map()
        edge_map_copy = copy.deepcopy(edge_map)
        self.assertEqual(edge_map_copy, edge_map)

    def test_pickle(self):
        edge_map = self.dag.edge_index_map()
        edge_map_pickle = pickle.dumps(edge_map)
        edge_map_copy = pickle.loads(edge_map_pickle)
        self.assertEqual(edge_map, edge_map_copy)

    def test_str(self):
        res = self.dag.edge_index_map()
        self.assertEqual(
            "EdgeIndexMap{0: (0, 1, edge)}",
            str(res),
        )

    def test_hash(self):
        res = self.dag.edge_index_map()
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = self.dag.edge_index_map()
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = self.dag.edge_index_map().keys()
        self.assertEqual([0], list(keys))

    def test_values(self):
        values = self.dag.edge_index_map().values()
        expected = [(0, 1, "edge")]
        self.assertEqual(expected, list(values))

    def test_items(self):
        items = self.dag.edge_index_map().items()
        self.assertEqual([(0, (0, 1, "edge"))], list(items))

    def test_iter(self):
        mapping_iter = iter(self.dag.edge_index_map())
        output = list(mapping_iter)
        self.assertEqual(output, [0])

    def test_contains(self):
        res = self.dag.edge_index_map()
        self.assertIn(0, res)

    def test_not_contains(self):
        res = self.dag.edge_index_map()
        self.assertNotIn(1, res)


class TestAllPairsPathMapping(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")
        self.fn = lambda _: 1.0

    def test__eq__match(self):
        self.assertTrue(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
            == {0: {1: [0, 1]}, 1: {}}
        )

    def test__eq__not_match_keys(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
            == {2: {2: [0, 1]}, 1: {}}
        )

    def test__eq__not_match_values(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
            == {0: {1: [0, 2]}, 1: {}}
        )

    def test__eq__different_length(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn) == {1: [0, 1], 2: [0, 2]}
        )

    def test_eq__same_type(self):
        self.assertEqual(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn),
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn),
        )

    def test__eq__invalid_type(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn) == {"a": []}
        )

    def test__eq__invalid_inner_type(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn) == {0: {1: None}}
        )

    def test__ne__match(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
            != {0: {1: [0, 1]}, 1: {}}
        )

    def test__ne__not_match(self):
        self.assertTrue(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn) != {2: [0, 1]}
        )

    def test__ne__not_match_values(self):
        self.assertTrue(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn) != {1: [0, 2]}
        )

    def test__ne__different_length(self):
        self.assertTrue(
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn) != {1: [0, 1], 2: [0, 2]}
        )

    def test__ne__invalid_type(self):
        self.assertTrue(rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn) != {"a": {}})

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn) > {1: [0, 2]}

    def test_deepcopy(self):
        paths = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
        paths_copy = copy.deepcopy(paths)
        self.assertEqual(paths, paths_copy)

    def test_pickle(self):
        paths = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
        paths_pickle = pickle.dumps(paths)
        paths_copy = pickle.loads(paths_pickle)
        self.assertEqual(paths, paths_copy)

    def test_str(self):
        res = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
        # Since run in parallel the order is not deterministic
        expected_valid = [
            "AllPairsPathMapping{1: PathMapping{}, 0: PathMapping{1: [0, 1]}}",
            "AllPairsPathMapping{0: PathMapping{1: [0, 1]}, 1: PathMapping{}}",
        ]
        self.assertIn(str(res), expected_valid)

    def test_hash(self):
        res = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn).keys()
        self.assertEqual([0, 1], list(sorted(keys)))

    def test_values(self):
        values = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn).values()
        # Since run in parallel the order is not deterministic
        expected_valid = [[{1: [0, 1]}, {}], [{}, {1: [0, 1]}]]
        self.assertIn(list(values), expected_valid)

    def test_items(self):
        items = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn).items()
        # Since run in parallel the order is not deterministic
        expected_valid = [
            [(0, {1: [0, 1]}), (1, {})],
            [(1, {}), (0, {1: [0, 1]})],
        ]
        self.assertIn(list(items), expected_valid)

    def test_iter(self):
        mapping_iter = iter(rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn))
        output = list(sorted(mapping_iter))
        self.assertEqual(output, [0, 1])

    def test_contains(self):
        res = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
        self.assertIn(1, res)

    def test_not_contains(self):
        res = rustworkx.all_pairs_dijkstra_shortest_paths(self.dag, self.fn)
        self.assertNotIn(2, res)


class TestAllPairsPathLengthMapping(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")
        self.fn = lambda _: 1.0

    def test__eq__match(self):
        self.assertTrue(
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) == {0: {1: 1.0}, 1: {}}
        )

    def test__eq__not_match_keys(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) == {1: {2: 1.0}}
        )

    def test__eq__not_match_values(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) == {0: {2: 2.0}}
        )

    def test__eq__different_length(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) == {0: {1: 1.0, 2: 2.0}}
        )

    def test_eq__same_type(self):
        self.assertEqual(
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn),
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn),
        )

    def test__eq__invalid_type(self):
        self.assertFalse(rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) == {"a": 2})

    def test__eq__invalid_inner_type(self):
        self.assertFalse(rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) == {0: "a"})

    def test__ne__match(self):
        self.assertFalse(
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) != {0: {1: 1.0}, 1: {}}
        )

    def test__ne__not_match(self):
        self.assertTrue(
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) != {0: {2: 1.0}}
        )

    def test__ne__not_match_values(self):
        self.assertTrue(
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) != {0: {1: 2.0}}
        )

    def test__ne__different_length(self):
        self.assertTrue(
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn)
            != {0: {1: 1.0}, 2: {1: 2.0}}
        )

    def test__ne__invalid_type(self):
        self.assertTrue(rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) != {1: []})

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn) > {1: 1.0}

    def test_deepcopy(self):
        paths = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn)
        paths_copy = copy.deepcopy(paths)
        self.assertEqual(paths, paths_copy)

    def test_pickle(self):
        paths = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn)
        paths_pickle = pickle.dumps(paths)
        paths_copy = pickle.loads(paths_pickle)
        self.assertEqual(paths, paths_copy)

    def test_str(self):
        res = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, lambda _: 3.14)
        # Since all_pairs_dijkstra_path_lengths() is parallel the order of the
        # output is non-determinisitic
        valid_values = [
            "AllPairsPathLengthMapping{1: PathLengthMapping{}, " "0: PathLengthMapping{1: 3.14}}",
            "AllPairsPathLengthMapping{"
            "0: PathLengthMapping{1: 3.14}, "
            "1: PathLengthMapping{}}",
        ]
        self.assertIn(str(res), valid_values)

    def test_hash(self):
        res = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn).keys()
        self.assertEqual([0, 1], list(sorted(keys)))

    def test_values(self):
        values = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn).values()
        # Since run in parallel the order is not deterministic
        valid_expected = [[{}, {1: 1.0}], [{1: 1.0}, {}]]
        self.assertIn(list(values), valid_expected)

    def test_items(self):
        items = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn).items()
        # Since run in parallel the order is not deterministic
        valid_expected = [[(0, {1: 1.0}), (1, {})], [(1, {}), (0, {1: 1.0})]]
        self.assertIn(list(items), valid_expected)

    def test_iter(self):
        mapping_iter = iter(rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn))
        output = list(sorted(mapping_iter))
        self.assertEqual(output, [0, 1])

    def test_contains(self):
        res = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn)
        self.assertIn(0, res)

    def test_not_contains(self):
        res = rustworkx.all_pairs_dijkstra_path_lengths(self.dag, self.fn)
        self.assertNotIn(2, res)


class TestNodeMap(unittest.TestCase):
    def setUp(self):
        self.dag = rustworkx.PyDAG()
        self.dag.add_node("a")
        self.in_dag = rustworkx.generators.directed_path_graph(1)

    def test__eq__match(self):
        self.assertTrue(
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None) == {0: 1}
        )

    def test__eq__not_match_keys(self):
        self.assertFalse(
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None) == {2: 1}
        )

    def test__eq__not_match_values(self):
        self.assertFalse(
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None) == {0: 2}
        )

    def test__eq__different_length(self):
        self.assertFalse(
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
            == {0: 1, 1: 2}
        )

    def test_eq__same_type(self):
        res = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
        self.assertEqual(res, res)

    def test__ne__match(self):
        self.assertFalse(
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None) != {0: 1}
        )

    def test__ne__not_match(self):
        self.assertTrue(
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None) != {2: 2}
        )

    def test__ne__not_match_values(self):
        self.assertTrue(
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None) != {0: 2}
        )

    def test__ne__different_length(self):
        self.assertTrue(
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
            != {0: 1, 1: 2}
        )

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None) > {1: 2}

    def test__len__(self):
        in_dag = rustworkx.generators.directed_grid_graph(5, 5)
        node_map = self.dag.substitute_node_with_subgraph(0, in_dag, lambda *args: None)
        self.assertEqual(25, len(node_map))

    def test_deepcopy(self):
        node_map = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
        node_map_copy = copy.deepcopy(node_map)
        self.assertEqual(node_map, node_map_copy)

    def test_pickle(self):
        node_map = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
        node_map_pickle = pickle.dumps(node_map)
        node_map_copy = pickle.loads(node_map_pickle)
        self.assertEqual(node_map, node_map_copy)

    def test_str(self):
        res = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
        self.assertEqual("NodeMap{0: 1}", str(res))

    def test_hash(self):
        res = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None).keys()
        self.assertEqual([0], list(keys))

    def test_values(self):
        values = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None).values()
        self.assertEqual([1], list(values))

    def test_items(self):
        items = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None).items()
        self.assertEqual([(0, 1)], list(items))

    def test_iter(self):
        mapping_iter = iter(
            self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
        )
        output = list(mapping_iter)
        self.assertEqual(output, [0])

    def test_contains(self):
        res = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
        self.assertIn(0, res)

    def test_not_contains(self):
        res = self.dag.substitute_node_with_subgraph(0, self.in_dag, lambda *args: None)
        self.assertNotIn(2, res)

    def test_iter_stable_for_same_obj(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        in_graph = rustworkx.generators.directed_path_graph(5)
        res = self.dag.substitute_node_with_subgraph(0, in_graph, lambda *args: None)
        first_iter = list(iter(res))
        second_iter = list(iter(res))
        third_iter = list(iter(res))
        self.assertEqual(first_iter, second_iter)
        self.assertEqual(first_iter, third_iter)


class TestChainsComparisons(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.generators.cycle_graph(3)
        self.chains = rustworkx.chain_decomposition(self.graph)

    def test__eq__match(self):
        self.assertTrue(self.chains == [[(0, 2), (2, 1), (1, 0)]])

    def test__eq__not_match(self):
        self.assertFalse(self.chains == [[(0, 2), (2, 1), (2, 0)]])

    def test__eq__different_length(self):
        self.assertFalse(self.chains == [[(0, 2)]])

    def test__eq__invalid_type(self):
        with self.assertRaises(TypeError):
            self.chains == [0]

    def test__ne__match(self):
        self.assertFalse(self.chains != [[(0, 2), (2, 1), (1, 0)]])

    def test__ne__not_match(self):
        self.assertTrue(self.chains != [[(0, 2), (2, 1), (2, 0)]])

    def test__ne__different_length(self):
        self.assertTrue(self.chains != [[(0, 2)]])

    def test__ne__invalid_type(self):
        with self.assertRaises(TypeError):
            self.chains != [0]

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.chains > [[(0, 2)]]

    def test_deepcopy(self):
        chains_copy = copy.deepcopy(self.chains)
        self.assertEqual(self.chains, chains_copy)

    def test_pickle(self):
        chains_pickle = pickle.dumps(self.chains)
        chains_copy = pickle.loads(chains_pickle)
        self.assertEqual(self.chains, chains_copy)

    def test_str(self):
        self.assertEqual("Chains[EdgeList[(0, 2), (2, 1), (1, 0)]]", str(self.chains))

    def test_hash(self):
        hash_res = hash(self.chains)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(self.chains))

    def test_numpy_conversion(self):
        # this test assumes the array is 1-dimensional which avoids issues with jagged arrays
        self.assertTrue(np.asarray(self.chains).shape, (1,))


class TestProductNodeMap(unittest.TestCase):
    def setUp(self):
        self.first = rustworkx.PyGraph()
        self.first.add_node("a0")
        self.first.add_node("a1")

        self.second = rustworkx.PyGraph()
        self.second.add_node("b")
        _, self.node_map = rustworkx.graph_cartesian_product(self.first, self.second)

    def test__eq__match(self):
        self.assertTrue(self.node_map == {(0, 0): 0, (1, 0): 1})

    def test__eq__not_match_keys(self):
        self.assertFalse(self.node_map == {(0, 0): 0, (2, 0): 1})

    def test__eq__not_match_values(self):
        self.assertFalse(self.node_map == {(0, 0): 0, (1, 0): 2})

    def test__eq__different_length(self):
        self.assertFalse(self.node_map == {(0, 0): 0})

    def test_eq__same_type(self):
        _, res = rustworkx.graph_cartesian_product(self.first, self.second)
        self.assertEqual(self.node_map, res)

    def test__ne__match(self):
        self.assertFalse(self.node_map != {(0, 0): 0, (1, 0): 1})

    def test__ne__not_match(self):
        self.assertTrue(self.node_map != {(0, 0): 0, (2, 0): 1})

    def test__ne__not_match_values(self):
        self.assertTrue(self.node_map != {(0, 0): 0, (1, 0): 2})

    def test__ne__different_length(self):
        self.assertTrue(self.node_map != {(0, 0): 0})

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.node_map > {1: 2}

    def test__len__(self):
        self.assertEqual(2, len(self.node_map))

    def test_deepcopy(self):
        node_map_copy = copy.deepcopy(self.node_map)
        self.assertEqual(self.node_map, node_map_copy)

    def test_pickle(self):
        node_map_pickle = pickle.dumps(self.node_map)
        node_map_copy = pickle.loads(node_map_pickle)
        self.assertEqual(self.node_map, node_map_copy)

    def test_str(self):
        valid_str_output = [
            "ProductNodeMap{(0, 0): 0, (1, 0): 1}",
            "ProductNodeMap{(1, 0): 1, (0, 0): 0}",
        ]
        self.assertTrue(str(self.node_map) in valid_str_output)

    def test_hash(self):
        hash_res = hash(self.node_map)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(self.node_map))

    def test_index_error(self):
        with self.assertRaises(IndexError):
            self.node_map[(1, 1)]

    def test_keys(self):
        keys = self.node_map.keys()
        self.assertEqual(set([(0, 0), (1, 0)]), set(keys))

    def test_values(self):
        values = self.node_map.values()
        self.assertEqual(set([0, 1]), set(values))

    def test_items(self):
        items = self.node_map.items()
        self.assertEqual(set([((0, 0), 0), ((1, 0), 1)]), set(items))

    def test_iter(self):
        mapping_iter = iter(self.node_map)
        output = set(mapping_iter)
        self.assertEqual(output, set([(0, 0), (1, 0)]))

    def test_contains(self):
        self.assertIn((0, 0), self.node_map)

    def test_not_contains(self):
        self.assertNotIn((1, 1), self.node_map)


class TestBiconnectedComponentsMap(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.generators.path_graph(3)
        self.bicon_map = rustworkx.biconnected_components(self.graph)

    def test__eq__match(self):
        self.assertTrue(self.bicon_map == {(0, 1): 1, (1, 2): 0})

    def test__eq__not_match_keys(self):
        self.assertFalse(self.bicon_map == {(0, 0): 1, (2, 0): 0})

    def test__eq__not_match_values(self):
        self.assertFalse(self.bicon_map == {(0, 1): 2, (1, 2): 0})

    def test__eq__different_length(self):
        self.assertFalse(self.bicon_map == {(0, 1): 1})

    def test_eq__same_type(self):
        res = rustworkx.biconnected_components(self.graph)
        self.assertEqual(self.bicon_map, res)

    def test__ne__match(self):
        self.assertFalse(self.bicon_map != {(0, 1): 1, (1, 2): 0})

    def test__ne__not_match(self):
        self.assertTrue(self.bicon_map != {(0, 2): 1, (1, 2): 0})

    def test__ne__not_match_values(self):
        self.assertTrue(self.bicon_map != {(0, 1): 0, (1, 2): 0})

    def test__ne__different_length(self):
        self.assertTrue(self.bicon_map != {(0, 1): 1})

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.bicon_map > {1: 2}

    def test__len__(self):
        self.assertEqual(2, len(self.bicon_map))

    def test_deepcopy(self):
        bicon_map_copy = copy.deepcopy(self.bicon_map)
        self.assertEqual(self.bicon_map, bicon_map_copy)

    def test_pickle(self):
        bicon_map_pickle = pickle.dumps(self.bicon_map)
        bicon_map_copy = pickle.loads(bicon_map_pickle)
        self.assertEqual(self.bicon_map, bicon_map_copy)

    def test_str(self):
        valid_str_output = [
            "BiconnectedComponents{(0, 1): 1, (1, 2): 0}",
            "BiconnectedComponents{(1, 2): 0, (0, 1): 1}",
        ]
        self.assertTrue(str(self.bicon_map) in valid_str_output)

    def test_hash(self):
        hash_res = hash(self.bicon_map)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(self.bicon_map))

    def test_index_error(self):
        with self.assertRaises(IndexError):
            self.bicon_map[(1, 1)]

    def test_keys(self):
        keys = self.bicon_map.keys()
        self.assertEqual(set([(0, 1), (1, 2)]), set(keys))

    def test_values(self):
        values = self.bicon_map.values()
        self.assertEqual(set([0, 1]), set(values))

    def test_items(self):
        items = self.bicon_map.items()
        self.assertEqual(set([((0, 1), 1), ((1, 2), 0)]), set(items))

    def test_iter(self):
        mapping_iter = iter(self.bicon_map)
        output = set(mapping_iter)
        self.assertEqual(output, set([(0, 1), (1, 2)]))

    def test_contains(self):
        self.assertIn((0, 1), self.bicon_map)

    def test_not_contains(self):
        self.assertNotIn((0, 2), self.bicon_map)
