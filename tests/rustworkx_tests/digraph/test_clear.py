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


class TestClear(unittest.TestCase):
    def test_clear(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})
        dag.clear() # clear nodes and edges
        self.assertEqual(dag.num_nodes(), 0)
        self.assertEqual(dag.num_edges(), 0)
        self.assertEqual(dag.nodes(), [])
        self.assertEqual(dag.edges(), [])


    def test_clear_edges(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_a, "c", {"a": 2})
        res = dag.adj_direction(node_a, False)
        self.assertEqual(graph.num_nodes(), 3)
        self.assertEqual(graph.num_edges(), 0)
        self.assertEqual(graph.nodes(), ["a", "b", "c"])
        self.assertEqual(graph.edges(), [])

