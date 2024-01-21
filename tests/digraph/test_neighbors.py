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
        res = dag.neighbors(node_a)
        self.assertCountEqual([node_c, node_b], res)

    def test_unique_neighbors_on_dags(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", ["edge a->b"])
        node_c = dag.add_child(node_a, "c", ["edge a->c"])
        dag.add_edge(node_a, node_b, ["edge a->b bis"])
        res = dag.neighbors(node_a)
        self.assertCountEqual([node_c, node_b], res)

    def test_single_neighbor_dir(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})
        res = dag.successor_indices(node_a)
        self.assertEqual([node_c, node_b], res)
        res = dag.predecessor_indices(node_a)
        self.assertEqual([], res)

    def test_neighbor_dir_surrounded(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        res = dag.successor_indices(node_b)
        self.assertEqual([node_c], res)
        res = dag.predecessor_indices(node_b)
        self.assertEqual([node_a], res)

    def test_no_neighbor(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        self.assertEqual([], dag.neighbors(node_a))
