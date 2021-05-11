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
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")

    def test__eq__match(self):
        self.assertTrue(retworkx.bfs_successors(self.dag, 0) == [("a", ["b"])])

    def test__eq__not_match(self):
        self.assertFalse(retworkx.bfs_successors(self.dag, 0) == [("b", ["c"])])

    def test_eq_not_match_inner(self):
        self.assertFalse(retworkx.bfs_successors(self.dag, 0) == [("a", ["c"])])

    def test__eq__different_length(self):
        self.assertFalse(
            retworkx.bfs_successors(self.dag, 0) == [("a", ["b"]), ("b", ["c"])]
        )

    def test__eq__invalid_type(self):
        with self.assertRaises(TypeError):
            retworkx.bfs_successors(self.dag, 0) == ["a"]

    def test__ne__match(self):
        self.assertFalse(retworkx.bfs_successors(self.dag, 0) != [("a", ["b"])])

    def test__ne__not_match(self):
        self.assertTrue(retworkx.bfs_successors(self.dag, 0) != [("b", ["c"])])

    def test_ne_not_match_inner(self):
        self.assertTrue(retworkx.bfs_successors(self.dag, 0) != [("a", ["c"])])

    def test__ne__different_length(self):
        self.assertTrue(
            retworkx.bfs_successors(self.dag, 0) != [("a", ["b"]), ("b", ["c"])]
        )

    def test__ne__invalid_type(self):
        with self.assertRaises(TypeError):
            retworkx.bfs_successors(self.dag, 0) != ["a"]

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            retworkx.bfs_successors(self.dag, 0) > [("b", ["c"])]

    def test_deepcopy(self):
        bfs = retworkx.bfs_successors(self.dag, 0)
        bfs_copy = copy.deepcopy(bfs)
        self.assertEqual(bfs, bfs_copy)

    def test_pickle(self):
        bfs = retworkx.bfs_successors(self.dag, 0)
        bfs_pickle = pickle.dumps(bfs)
        bfs_copy = pickle.loads(bfs_pickle)
        self.assertEqual(bfs, bfs_copy)

    def test_str(self):
        res = retworkx.bfs_successors(self.dag, 0)
        self.assertEqual("BFSSuccessors[(a, [b])]", str(res))

    def test_hash(self):
        res = retworkx.bfs_successors(self.dag, 0)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_hash_invalid_type(self):
        self.dag.add_child(0, [1, 2, 3], "edgy")
        res = retworkx.bfs_successors(self.dag, 0)
        with self.assertRaises(TypeError):
            hash(res)


class TestNodeIndicesComparisons(unittest.TestCase):
    def setUp(self):
        self.dag = retworkx.PyDAG()
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


class TestEdgeListComparisons(unittest.TestCase):
    def setUp(self):
        self.dag = retworkx.PyDAG()
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


class TestWeightedEdgeListComparisons(unittest.TestCase):
    def setUp(self):
        self.dag = retworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")

    def test__eq__match(self):
        self.assertTrue(self.dag.weighted_edge_list() == [(0, 1, "Edgy")])

    def test__eq__not_match(self):
        self.assertFalse(self.dag.weighted_edge_list() == [(1, 2, None)])

    def test__eq__different_length(self):
        self.assertFalse(
            self.dag.weighted_edge_list()
            == [(0, 1, "Edgy"), (2, 3, "Not Edgy")]
        )

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


class TestPathMapping(unittest.TestCase):
    def setUp(self):
        self.dag = retworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")

    def test__eq__match(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_paths(self.dag, 0) == {1: [0, 1]}
        )

    def test__eq__not_match_keys(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_paths(self.dag, 0) == {2: [0, 1]}
        )

    def test__eq__not_match_values(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_paths(self.dag, 0) == {1: [0, 2]}
        )

    def test__eq__different_length(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_paths(self.dag, 0)
            == {1: [0, 1], 2: [0, 2]}
        )

    def test_eq__same_type(self):
        self.assertEqual(
            retworkx.dijkstra_shortest_paths(self.dag, 0),
            retworkx.dijkstra_shortest_paths(self.dag, 0),
        )

    def test__eq__invalid_type(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_paths(self.dag, 0) == ["a", None]
        )

    def test__eq__invalid_inner_type(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_paths(self.dag, 0) == {0: {"a": None}}
        )

    def test__ne__match(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_paths(self.dag, 0) != {1: [0, 1]}
        )

    def test__ne__not_match(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_paths(self.dag, 0) != {2: [0, 1]}
        )

    def test__ne__not_match_values(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_paths(self.dag, 0) != {1: [0, 2]}
        )

    def test__ne__different_length(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_paths(self.dag, 0)
            != {1: [0, 1], 2: [0, 2]}
        )

    def test__ne__invalid_type(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_paths(self.dag, 0) != ["a", None]
        )

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            retworkx.dijkstra_shortest_paths(self.dag, 0) > {1: [0, 2]}

    def test_deepcopy(self):
        paths = retworkx.dijkstra_shortest_paths(self.dag, 0)
        paths_copy = copy.deepcopy(paths)
        self.assertEqual(paths, paths_copy)

    def test_pickle(self):
        paths = retworkx.dijkstra_shortest_paths(self.dag, 0)
        paths_pickle = pickle.dumps(paths)
        paths_copy = pickle.loads(paths_pickle)
        self.assertEqual(paths, paths_copy)

    def test_str(self):
        res = retworkx.dijkstra_shortest_paths(self.dag, 0)
        self.assertEqual("PathMapping{1: [0, 1]}", str(res))

    def test_hash(self):
        res = retworkx.dijkstra_shortest_paths(self.dag, 0)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = retworkx.dijkstra_shortest_paths(self.dag, 0)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = retworkx.dijkstra_shortest_paths(self.dag, 0).keys()
        self.assertEqual([1], list(keys))

    def test_values(self):
        values = retworkx.dijkstra_shortest_paths(self.dag, 0).values()
        self.assertEqual([[0, 1]], list(values))

    def test_items(self):
        items = retworkx.dijkstra_shortest_paths(self.dag, 0).items()
        self.assertEqual([(1, [0, 1])], list(items))

    def test_iter(self):
        mapping_iter = iter(retworkx.dijkstra_shortest_paths(self.dag, 0))
        output = list(mapping_iter)
        self.assertEqual(output, [1])

    def test_contains(self):
        res = retworkx.dijkstra_shortest_paths(self.dag, 0)
        self.assertIn(1, res)

    def test_not_contains(self):
        res = retworkx.dijkstra_shortest_paths(self.dag, 0)
        self.assertNotIn(0, res)


class TestPathLengthMapping(unittest.TestCase):
    def setUp(self):
        self.dag = retworkx.PyDAG()
        node_a = self.dag.add_node("a")
        self.dag.add_child(node_a, "b", "Edgy")
        self.fn = lambda _: 1.0

    def test__eq__match(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            == {1: 1.0}
        )

    def test__eq__not_match_keys(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            == {2: 1.0}
        )

    def test__eq__not_match_values(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            == {1: 2.0}
        )

    def test__eq__different_length(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            == {1: 1.0, 2: 2.0}
        )

    def test_eq__same_type(self):
        self.assertEqual(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn),
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn),
        )

    def test__eq__invalid_type(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            == ["a", None]
        )

    def test__eq__invalid_inner_type(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            == {0: "a"}
        )

    def test__ne__match(self):
        self.assertFalse(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            != {1: 1.0}
        )

    def test__ne__not_match(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            != {2: 1.0}
        )

    def test__ne__not_match_values(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            != {1: 2.0}
        )

    def test__ne__different_length(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            != {1: 1.0, 2: 2.0}
        )

    def test__ne__invalid_type(self):
        self.assertTrue(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
            != ["a", None]
        )

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn) > {
                1: 1.0
            }

    def test_deepcopy(self):
        paths = retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        paths_copy = copy.deepcopy(paths)
        self.assertEqual(paths, paths_copy)

    def test_pickle(self):
        paths = retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        paths_pickle = pickle.dumps(paths)
        paths_copy = pickle.loads(paths_pickle)
        self.assertEqual(paths, paths_copy)

    def test_str(self):
        res = retworkx.dijkstra_shortest_path_lengths(
            self.dag, 0, lambda _: 3.14
        )
        self.assertEqual("PathLengthMapping{1: 3.14}", str(res))

    def test_hash(self):
        res = retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = retworkx.dijkstra_shortest_path_lengths(
            self.dag, 0, self.fn
        ).keys()
        self.assertEqual([1], list(keys))

    def test_values(self):
        values = retworkx.dijkstra_shortest_path_lengths(
            self.dag, 0, self.fn
        ).values()
        self.assertEqual([1.0], list(values))

    def test_items(self):
        items = retworkx.dijkstra_shortest_path_lengths(
            self.dag, 0, self.fn
        ).items()
        self.assertEqual([(1, 1.0)], list(items))

    def test_iter(self):
        mapping_iter = iter(
            retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        )
        output = list(mapping_iter)
        self.assertEqual(output, [1])

    def test_contains(self):
        res = retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        self.assertIn(1, res)

    def test_not_contains(self):
        res = retworkx.dijkstra_shortest_path_lengths(self.dag, 0, self.fn)
        self.assertNotIn(0, res)


class TestPos2DMapping(unittest.TestCase):
    def setUp(self):
        self.dag = retworkx.PyDiGraph()
        self.dag.add_node("a")

    def test__eq__match(self):
        res = retworkx.random_layout(self.dag, seed=10244242)
        self.assertTrue(res == {0: (0.4883489113112722, 0.6545867364101975)})

    def test__eq__not_match_keys(self):
        self.assertFalse(
            retworkx.random_layout(self.dag, seed=10244242) == {2: 1.0}
        )

    def test__eq__not_match_values(self):
        self.assertFalse(
            retworkx.random_layout(self.dag, seed=10244242) == {1: 2.0}
        )

    def test__eq__different_length(self):
        res = retworkx.random_layout(self.dag, seed=10244242)
        self.assertFalse(res == {1: 1.0, 2: 2.0})

    def test_eq__same_type(self):
        self.assertEqual(
            retworkx.random_layout(self.dag, seed=10244242),
            retworkx.random_layout(self.dag, seed=10244242),
        )

    def test__eq__invalid_type(self):
        self.assertFalse(
            retworkx.random_layout(self.dag, seed=10244242) == {"a": None}
        )

    def test__ne__match(self):
        res = retworkx.random_layout(self.dag, seed=10244242)
        self.assertFalse(res != {0: (0.4883489113112722, 0.6545867364101975)})

    def test__ne__not_match(self):
        self.assertTrue(
            retworkx.random_layout(self.dag, seed=10244242) != {2: 1.0}
        )

    def test__ne__not_match_values(self):
        self.assertTrue(
            retworkx.random_layout(self.dag, seed=10244242) != {1: 2.0}
        )

    def test__ne__different_length(self):
        res = retworkx.random_layout(self.dag, seed=10244242)

        self.assertTrue(res != {1: 1.0, 2: 2.0})

    def test__ne__invalid_type(self):
        self.assertTrue(
            retworkx.random_layout(self.dag, seed=10244242) != ["a", None]
        )

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            retworkx.random_layout(self.dag, seed=10244242) > {1: 1.0}

    def test_deepcopy(self):
        positions = retworkx.random_layout(self.dag)
        positions_copy = copy.deepcopy(positions)
        self.assertEqual(positions_copy, positions)

    def test_pickle(self):
        pos = retworkx.random_layout(self.dag)
        pos_pickle = pickle.dumps(pos)
        pos_copy = pickle.loads(pos_pickle)
        self.assertEqual(pos, pos_copy)

    def test_str(self):
        res = retworkx.random_layout(self.dag, seed=10244242)
        self.assertEqual(
            "Pos2DMapping{0: (0.4883489113112722, 0.6545867364101975)}",
            str(res),
        )

    def test_hash(self):
        res = retworkx.random_layout(self.dag, seed=10244242)
        hash_res = hash(res)
        self.assertIsInstance(hash_res, int)
        # Assert hash is stable
        self.assertEqual(hash_res, hash(res))

    def test_index_error(self):
        res = retworkx.random_layout(self.dag, seed=10244242)
        with self.assertRaises(IndexError):
            res[42]

    def test_keys(self):
        keys = retworkx.random_layout(self.dag, seed=10244242).keys()
        self.assertEqual([0], list(keys))

    def test_values(self):
        values = retworkx.random_layout(self.dag, seed=10244242).values()
        expected = [[0.4883489113112722, 0.6545867364101975]]
        self.assertEqual(expected, list(values))

    def test_items(self):
        items = retworkx.random_layout(self.dag, seed=10244242).items()
        self.assertEqual(
            [(0, [0.4883489113112722, 0.6545867364101975])], list(items)
        )

    def test_iter(self):
        mapping_iter = iter(retworkx.random_layout(self.dag, seed=10244242))
        output = list(mapping_iter)
        self.assertEqual(output, [0])

    def test_contains(self):
        res = retworkx.random_layout(self.dag, seed=10244242)
        self.assertIn(0, res)

    def test_not_contains(self):
        res = retworkx.random_layout(self.dag, seed=10244242)
        self.assertNotIn(1, res)
