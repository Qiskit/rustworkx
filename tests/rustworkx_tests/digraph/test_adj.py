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


class TestAdj(unittest.TestCase):
    def test_single_neighbor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})
        res = dag.adj(node_a)
        self.assertEqual({node_b: {"a": 1}, node_c: {"a": 2}}, res)

    def test_in_and_out_adj_neighbor(self):
        dag = rustworkx.PyDAG()
        dag.extend_from_weighted_edge_list([(0, 1, "a"), (1, 2, "b")])
        res = dag.adj(1)
        self.assertEqual({0: "a", 2: "b"}, res)

    def test_single_neighbor_dir(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})
        res = dag.adj_direction(node_a, False)
        self.assertEqual({node_b: {"a": 1}, node_c: {"a": 2}}, res)
        res = dag.adj_direction(node_a, True)
        self.assertEqual({}, res)

    def test_neighbor_dir_surrounded(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        res = dag.adj_direction(node_b, False)
        self.assertEqual({node_c: {"a": 2}}, res)
        res = dag.adj_direction(node_b, True)
        self.assertEqual({node_a: {"a": 1}}, res)

    def test_single_neighbor_dir_out_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})
        res = dag.out_edges(node_a)
        self.assertEqual([(node_a, node_c, {"a": 2}), (node_a, node_b, {"a": 1})], res)

    def test_neighbor_dir_surrounded_in_out_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        res = dag.out_edges(node_b)
        self.assertEqual([(node_b, node_c, {"a": 2})], res)
        res = dag.in_edges(node_b)
        self.assertEqual([(node_a, node_b, {"a": 1})], res)

    def test_no_neighbor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        self.assertEqual({}, dag.adj(node_a))

    def test_in_direction(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(5):
            dag.add_parent(node_a, i, None)
        self.assertEqual(5, dag.in_degree(node_a))

    def test_in_direction_none(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(5):
            dag.add_child(node_a, i, None)
        self.assertEqual(0, dag.in_degree(node_a))

    def test_out_direction(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(5):
            dag.add_parent(node_a, i, None)
        self.assertEqual(0, dag.out_degree(node_a))

    def test_out_direction_none(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        for i in range(5):
            dag.add_child(node_a, i, None)
        self.assertEqual(5, dag.out_degree(node_a))
