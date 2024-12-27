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

import rustworkx


class TestPredecessors(unittest.TestCase):
    def test_single_predecessor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})
        res = dag.predecessors(node_c)
        self.assertEqual(["a"], res)

    def test_single_predecessor_multiple_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})
        dag.add_edge(node_a, node_c, {"a": 3})
        res = dag.predecessors(node_c)
        self.assertEqual(["a"], res)

    def test_many_parents(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(10):
            dag.add_parent(node_a, {"numeral": i}, {"edge": i})
        res = dag.predecessors(node_a)
        self.assertEqual(
            [
                {"numeral": 9},
                {"numeral": 8},
                {"numeral": 7},
                {"numeral": 6},
                {"numeral": 5},
                {"numeral": 4},
                {"numeral": 3},
                {"numeral": 2},
                {"numeral": 1},
                {"numeral": 0},
            ],
            res,
        )


class TestSuccessors(unittest.TestCase):
    def test_single_successor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        dag.add_child(node_c, "d", {"a": 1})
        res = dag.successors(node_b)
        self.assertEqual(["c"], res)

    def test_single_successor_multiple_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        dag.add_child(node_c, "d", {"a": 1})
        dag.add_edge(node_b, node_c, {"a": 3})
        res = dag.successors(node_b)
        self.assertEqual(["c"], res)

    def test_many_children(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(10):
            dag.add_child(node_a, {"numeral": i}, {"edge": i})
        res = dag.successors(node_a)
        self.assertEqual(
            [
                {"numeral": 9},
                {"numeral": 8},
                {"numeral": 7},
                {"numeral": 6},
                {"numeral": 5},
                {"numeral": 4},
                {"numeral": 3},
                {"numeral": 2},
                {"numeral": 1},
                {"numeral": 0},
            ],
            res,
        )


class TestFindPredecessorsByEdge(unittest.TestCase):
    def test_single_predecessor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})

        res_even = dag.find_predecessors_by_edge(node_c, lambda x: x["a"] % 2 == 0)

        res_odd = dag.find_predecessors_by_edge(node_c, lambda x: x["a"] % 2 != 0)

        self.assertEqual(["a"], res_even)
        self.assertEqual([], res_odd)

    def test_single_predecessor_multiple_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})
        dag.add_edge(node_a, node_c, {"a": 3})

        res_even = dag.find_predecessors_by_edge(node_c, lambda x: x["a"] % 2 == 0)

        res_odd = dag.find_predecessors_by_edge(node_c, lambda x: x["a"] % 2 == 0)

        self.assertEqual(["a"], res_even)
        self.assertEqual(["a"], res_odd)

    def test_many_parents(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(10):
            dag.add_parent(node_a, {"numeral": i}, {"edge": i})

        res_even = dag.find_predecessors_by_edge(node_a, lambda x: x["edge"] % 2 == 0)

        res_odd = dag.find_predecessors_by_edge(node_a, lambda x: x["edge"] % 2 != 0)

        self.assertEqual(
            [
                {"numeral": 8},
                {"numeral": 6},
                {"numeral": 4},
                {"numeral": 2},
                {"numeral": 0},
            ],
            res_even,
        )

        self.assertEqual(
            [
                {"numeral": 9},
                {"numeral": 7},
                {"numeral": 5},
                {"numeral": 3},
                {"numeral": 1},
            ],
            res_odd,
        )

    def test_no_parents(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")

        res = dag.find_predecessors_by_edge(node_a, lambda _: True)

        self.assertEqual([], res)


class TestFindSuccessorsByEdge(unittest.TestCase):
    def test_single_successor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        dag.add_child(node_c, "d", {"a": 1})

        res_even = dag.find_successors_by_edge(node_b, lambda x: x["a"] % 2 == 0)
        res_odd = dag.find_successors_by_edge(node_b, lambda x: x["a"] % 2 != 0)

        self.assertEqual(["c"], res_even)
        self.assertEqual([], res_odd)

    def test_single_successor_multiple_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        dag.add_child(node_c, "d", {"a": 1})
        dag.add_edge(node_b, node_c, {"a": 3})

        res_even = dag.find_successors_by_edge(node_b, lambda x: x["a"] % 2 == 0)
        res_odd = dag.find_successors_by_edge(node_b, lambda x: x["a"] % 2 != 0)

        self.assertEqual(["c"], res_even)
        self.assertEqual(["c"], res_odd)

    def test_many_children(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(10):
            dag.add_child(node_a, {"numeral": i}, {"edge": i})

        res_even = dag.find_successors_by_edge(node_a, lambda x: x["edge"] % 2 == 0)

        res_odd = dag.find_successors_by_edge(node_a, lambda x: x["edge"] % 2 != 0)

        self.assertEqual(
            [
                {"numeral": 8},
                {"numeral": 6},
                {"numeral": 4},
                {"numeral": 2},
                {"numeral": 0},
            ],
            res_even,
        )

        self.assertEqual(
            [
                {"numeral": 9},
                {"numeral": 7},
                {"numeral": 5},
                {"numeral": 3},
                {"numeral": 1},
            ],
            res_odd,
        )

    def test_no_children(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")

        res = dag.find_successors_by_edge(node_a, lambda _: True)

        self.assertEqual([], res)


class TestBfsSuccessors(unittest.TestCase):
    def test_single_successor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        dag.add_child(node_c, "d", {"a": 1})
        res = rustworkx.bfs_successors(dag, node_b)
        self.assertEqual([("b", ["c"]), ("c", ["d"])], res)

    def test_many_children(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(10):
            dag.add_child(node_a, {"numeral": i}, {"edge": i})
        res = rustworkx.bfs_successors(dag, node_a)
        self.assertEqual(
            [
                (
                    "a",
                    [
                        {"numeral": 9},
                        {"numeral": 8},
                        {"numeral": 7},
                        {"numeral": 6},
                        {"numeral": 5},
                        {"numeral": 4},
                        {"numeral": 3},
                        {"numeral": 2},
                        {"numeral": 1},
                        {"numeral": 0},
                    ],
                )
            ],
            res,
        )

    def test_bfs_successors(self):
        dag = rustworkx.PyDAG()
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
        res = {n: sorted(s) for n, s in rustworkx.bfs_successors(dag, node_b)}
        expected = {
            1: [2],
            2: [3, 7],
            3: [4],
            4: [5],
            5: [6],
            7: [8],
            8: [9],
            9: [10],
        }
        self.assertEqual(expected, res)
        self.assertEqual(
            [(7, [8]), (8, [9]), (9, [10])],
            rustworkx.bfs_successors(dag, node_h),
        )

    def test_bfs_successors_sequence(self):
        dag = rustworkx.PyDAG()
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
        res = rustworkx.bfs_successors(dag, node_b)
        expected = [
            (1, [2]),
            (2, [7, 3]),
            (7, [8]),
            (3, [4]),
            (8, [9]),
            (4, [5]),
            (9, [10]),
            (5, [6]),
        ]
        for index, expected_value in enumerate(expected):
            self.assertEqual((res[index][0], res[index][1]), expected_value)

    def test_bfs_successors_sequence_invalid_index(self):
        dag = rustworkx.PyDAG()
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
        res = rustworkx.bfs_successors(dag, node_b)
        with self.assertRaises(IndexError):
            res[8]

    def test_bfs_successors_sequence_negative_index(self):
        dag = rustworkx.PyDAG()
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
        res = rustworkx.bfs_successors(dag, node_b)
        self.assertEqual((5, [6]), res[-1])
        self.assertEqual((4, [5]), res[-3])

    def test_bfs_successors_sequence_stop_iterator(self):
        dag = rustworkx.PyDAG()
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
        res = iter(rustworkx.bfs_successors(dag, node_b))
        for _ in range(8):
            next(res)
        with self.assertRaises(StopIteration):
            next(res)


class TestBfsPredecessors(unittest.TestCase):
    def test_single_predecessor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        dag.add_child(node_c, "d", {"a": 1})
        res = rustworkx.bfs_predecessors(dag, node_c)
        res = rustworkx.bfs_predecessors(dag, node_c)
        self.assertEqual([("c", ["b"]), ("b", ["a"])], res)

    def test_many_parents(self):
        dag = rustworkx.PyDAG()
        parent_nodes = [dag.add_node({"parent": i}) for i in range(10)]
        child = dag.add_node("child")
        for i, parent in enumerate(parent_nodes):
            dag.add_edge(parent, child, {"edge": i})
        for i in range(10):
            dag.add_child(child, {"grand_child": i}, {"gc_edge": i})
        res = rustworkx.bfs_predecessors(dag, child)
        self.assertEqual(
            [
                (
                    "child",
                    [
                        {"parent": 9},
                        {"parent": 8},
                        {"parent": 7},
                        {"parent": 6},
                        {"parent": 5},
                        {"parent": 4},
                        {"parent": 3},
                        {"parent": 2},
                        {"parent": 1},
                        {"parent": 0},
                    ],
                )
            ],
            res,
        )

    def test_breadth_first(self):
        dag = rustworkx.PyDAG()
        layers = []
        parent_cnt = 8
        layers.append([dag.add_node({"layer1": i}) for i in range(parent_cnt)])
        child_cnt = parent_cnt / 2
        layers.append(
            [
                dag.add_child(parent1, {"layer2": i}, {})
                for i, parent1 in enumerate(layers[-1][0::2])
            ]
        )
        for parent2, child in zip(layers[-2][1::2], layers[-1]):
            dag.add_edge(parent2, child, {})

        parent_cnt = child_cnt
        child_cnt = parent_cnt / 2
        layers.append(
            [
                dag.add_child(parent1, {"layer3": i}, {})
                for i, parent1 in enumerate(layers[-1][0::2])
            ]
        )
        for parent2, child in zip(layers[-2][1::2], layers[-1]):
            dag.add_edge(parent2, child, {})

        parent_cnt = child_cnt
        child_cnt = parent_cnt / 2
        layers.append(
            [
                dag.add_child(parent1, {"layer4": i}, {})
                for i, parent1 in enumerate(layers[-1][0::2])
            ]
        )
        for parent2, child in zip(layers[-2][1::2], layers[-1]):
            dag.add_edge(parent2, child, {})

        res = rustworkx.bfs_predecessors(dag, child)
        self.assertEqual(
            res,
            [
                (
                    {"layer4": 0},
                    [
                        {"layer3": 1},
                        {"layer3": 0},
                    ],
                ),
                (
                    {"layer3": 1},
                    [
                        {"layer2": 3},
                        {"layer2": 2},
                    ],
                ),
                (
                    {"layer3": 0},
                    [
                        {"layer2": 1},
                        {"layer2": 0},
                    ],
                ),
                (
                    {"layer2": 3},
                    [
                        {"layer1": 7},
                        {"layer1": 6},
                    ],
                ),
                (
                    {"layer2": 2},
                    [
                        {"layer1": 5},
                        {"layer1": 4},
                    ],
                ),
                (
                    {"layer2": 1},
                    [
                        {"layer1": 3},
                        {"layer1": 2},
                    ],
                ),
                (
                    {"layer2": 0},
                    [
                        {"layer1": 1},
                        {"layer1": 0},
                    ],
                ),
            ],
        )
