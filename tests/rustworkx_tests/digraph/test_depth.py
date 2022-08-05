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


class TestLongestPath(unittest.TestCase):
    def test_linear(self):
        """Longest depth for a simple dag.

        a
        |
        b
        |\
        c d
        |\
        e |
        | |
        f g
        """
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        node_c = dag.add_child(node_b, "c", {})
        dag.add_child(node_b, "d", {})
        node_e = dag.add_child(node_c, "e", {})
        node_f = dag.add_child(node_e, "f", {})
        dag.add_child(node_c, "g", {})
        self.assertEqual(4, rustworkx.dag_longest_path_length(dag))
        self.assertEqual(
            [node_a, node_b, node_c, node_e, node_f],
            rustworkx.dag_longest_path(dag),
        )

    def test_less_linear(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {})
        node_c = dag.add_child(node_b, "c", {})
        node_d = dag.add_child(node_c, "d", {})
        node_e = dag.add_child(node_d, "e", {})
        dag.add_edge(node_a, node_c, {})
        dag.add_edge(node_a, node_e, {})
        dag.add_edge(node_c, node_e, {})
        self.assertEqual(4, rustworkx.dag_longest_path_length(dag))
        self.assertEqual(
            [node_a, node_b, node_c, node_d, node_e],
            rustworkx.dag_longest_path(dag),
        )

    def test_degenerate_graph(self):
        dag = rustworkx.PyDAG()
        dag.add_node(0)
        self.assertEqual(0, rustworkx.dag_longest_path_length(dag))
        self.assertEqual([0], rustworkx.dag_longest_path(dag))

    def test_empty_graph(self):
        dag = rustworkx.PyDAG()
        self.assertEqual(0, rustworkx.dag_longest_path_length(dag))
        self.assertEqual([], rustworkx.dag_longest_path(dag))

    def test_parallel_edges(self):
        dag = rustworkx.PyDiGraph()
        dag.extend_from_weighted_edge_list(
            [
                (0, 1, 1),
                (0, 3, 1),
                (3, 4, 1),
                (4, 5, 1),
                (1, 2, 1),
                (0, 1, 3),
            ]
        )
        self.assertEqual(
            [0, 3, 4, 5],
            rustworkx.dag_longest_path(dag),
        )

    def test_linear_with_weight(self):
        """Longest depth for a simple dag.

        a
        |
        b
        |\
        c d
        |\
        e |
        | |
        f g
        """
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", 4)
        node_c = dag.add_child(node_b, "c", 4)
        dag.add_child(node_b, "d", 5)
        node_e = dag.add_child(node_c, "e", 2)
        dag.add_child(node_e, "f", 2)
        node_g = dag.add_child(node_c, "g", 15)
        self.assertEqual(
            [node_a, node_b, node_c, node_g],
            rustworkx.dag_longest_path(dag, lambda _, __, weight: weight),
        )
        self.assertEqual(
            23,
            rustworkx.dag_longest_path_length(dag, lambda _, __, weight: weight),
        )

    def test_parallel_edges_with_weights(self):
        dag = rustworkx.PyDiGraph()
        dag.extend_from_weighted_edge_list(
            [
                (0, 1, 1),
                (0, 3, 1),
                (3, 4, 1),
                (4, 5, 1),
                (1, 2, 1),
                (0, 1, 3),
            ]
        )
        self.assertEqual(
            [0, 1, 2],
            rustworkx.dag_longest_path(dag, lambda _, __, weight: weight),
        )
        self.assertEqual(
            4,
            rustworkx.dag_longest_path_length(dag, weight_fn=lambda _, __, weight: weight),
        )

    def test_less_linear_with_weight(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", 1)
        node_c = dag.add_child(node_b, "c", 1)
        node_d = dag.add_child(node_c, "d", 1)
        node_e = dag.add_child(node_d, "e", 1)
        dag.add_edge(node_a, node_c, 3)
        dag.add_edge(node_a, node_e, 3)
        dag.add_edge(node_c, node_e, 3)
        self.assertEqual(
            6,
            rustworkx.dag_longest_path_length(dag, weight_fn=lambda _, __, weight: weight),
        )
        self.assertEqual(
            [node_a, node_c, node_e],
            rustworkx.dag_longest_path(dag, weight_fn=lambda _, __, weight: weight),
        )

    def test_degenerate_graph_with_weight(self):
        dag = rustworkx.PyDAG()
        dag.add_node(0)
        self.assertEqual([0], rustworkx.dag_longest_path(dag, weight_fn=weight_fn))
        self.assertEqual(0, rustworkx.dag_longest_path_length(dag, weight_fn=weight_fn))

    def test_empty_graph_with_weights(self):
        dag = rustworkx.PyDAG()
        self.assertEqual([], rustworkx.dag_longest_path(dag, weight_fn=weight_fn))
        self.assertEqual(0, rustworkx.dag_longest_path_length(dag, weight_fn=weight_fn))

    def test_cycle(self):
        not_a_dag = rustworkx.generators.directed_cycle_graph(250)
        with self.assertRaises(rustworkx.DAGHasCycle):
            rustworkx.dag_longest_path_length(not_a_dag, lambda *_: 1.0)
        with self.assertRaises(rustworkx.DAGHasCycle):
            rustworkx.dag_longest_path(not_a_dag, lambda *_: 1.0)


def weight_fn(_, __, weight):
    return int(weight)


class TestWeightedLongestPath(unittest.TestCase):
    def test_linear_with_weight(self):
        """Longest depth for a simple dag.

        a
        |
        b
        |\
        c d
        |\
        e |
        | |
        f g
        """
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", 4)
        node_c = dag.add_child(node_b, "c", 4)
        dag.add_child(node_b, "d", 5)
        node_e = dag.add_child(node_c, "e", 2)
        dag.add_child(node_e, "f", 2)
        node_g = dag.add_child(node_c, "g", 15)
        self.assertEqual(
            23.0,
            rustworkx.dag_weighted_longest_path_length(dag, lambda _, __, weight: float(weight)),
        )
        self.assertEqual(
            [node_a, node_b, node_c, node_g],
            rustworkx.dag_weighted_longest_path(dag, lambda _, __, weight: float(weight)),
        )

    def test_parallel_edges_with_weights(self):
        dag = rustworkx.PyDiGraph()
        dag.extend_from_weighted_edge_list(
            [
                (0, 1, 1),
                (0, 3, 1),
                (3, 4, 1),
                (4, 5, 1),
                (1, 2, 1),
                (0, 1, 3),
            ]
        )
        self.assertEqual(
            4.0,
            rustworkx.dag_weighted_longest_path_length(
                dag, weight_fn=lambda _, __, weight: float(weight)
            ),
        )
        self.assertEqual(
            [0, 1, 2],
            rustworkx.dag_weighted_longest_path(dag, lambda _, __, weight: float(weight)),
        )

    def test_less_linear_with_weight(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", 1)
        node_c = dag.add_child(node_b, "c", 1)
        node_d = dag.add_child(node_c, "d", 1)
        node_e = dag.add_child(node_d, "e", 1)
        dag.add_edge(node_a, node_c, 3)
        dag.add_edge(node_a, node_e, 3)
        dag.add_edge(node_c, node_e, 3)
        self.assertEqual(
            6.0,
            rustworkx.dag_weighted_longest_path_length(
                dag, weight_fn=lambda _, __, weight: float(weight)
            ),
        )
        self.assertEqual(
            [node_a, node_c, node_e],
            rustworkx.dag_weighted_longest_path(dag, weight_fn=lambda _, __, weight: float(weight)),
        )

    def test_degenerate_graph_with_weight(self):
        dag = rustworkx.PyDAG()
        dag.add_node(0)
        self.assertEqual(
            0.0,
            rustworkx.dag_weighted_longest_path_length(dag, lambda x: float(weight_fn(x))),
        )
        self.assertEqual(
            [0],
            rustworkx.dag_weighted_longest_path(dag, lambda x: float(weight_fn(x))),
        )

    def test_empty_graph_with_weights(self):
        dag = rustworkx.PyDAG()
        self.assertEqual(
            0.0,
            rustworkx.dag_weighted_longest_path_length(dag, lambda x: float(weight_fn(x))),
        )
        self.assertEqual(
            [],
            rustworkx.dag_weighted_longest_path(dag, lambda x: float(weight_fn(x))),
        )

    def test_nan_not_valid_weight(self):
        dag = rustworkx.generators.directed_path_graph(526)

        def weight_fn(*_):
            return float("nan")

        with self.assertRaises(ValueError):
            rustworkx.dag_weighted_longest_path_length(dag, weight_fn)
        with self.assertRaises(ValueError):
            rustworkx.dag_weighted_longest_path(dag, weight_fn)

    def test_cycle(self):
        not_a_dag = rustworkx.generators.directed_cycle_graph(250)
        with self.assertRaises(rustworkx.DAGHasCycle):
            rustworkx.dag_weighted_longest_path_length(not_a_dag, lambda *_: 1.0)
        with self.assertRaises(rustworkx.DAGHasCycle):
            rustworkx.dag_weighted_longest_path(not_a_dag, lambda *_: 1.0)
