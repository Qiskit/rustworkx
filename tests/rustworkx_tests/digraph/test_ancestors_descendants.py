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


class TestAncestors(unittest.TestCase):
    def test_ancestors(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        res = rustworkx.ancestors(dag, node_c)
        self.assertEqual({node_a, node_b}, res)

    def test_no_ancestors(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        dag.add_child(node_a, "b", {"a": 1})
        res = rustworkx.ancestors(dag, node_a)
        self.assertEqual(set(), res)

    def test_ancestors_no_descendants(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        dag.add_child(node_b, "c", {"b": 1})
        res = rustworkx.ancestors(dag, node_b)
        self.assertEqual({node_a}, res)


class TestDescendants(unittest.TestCase):
    def test_descendants(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"a": 2})
        res = rustworkx.descendants(dag, node_a)
        self.assertEqual({node_b, node_c}, res)

    def test_no_descendants(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        res = rustworkx.descendants(dag, node_a)
        self.assertEqual(set(), res)

    def test_descendants_no_ancestors(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", {"a": 1})
        node_c = dag.add_child(node_b, "c", {"b": 1})
        res = rustworkx.descendants(dag, node_b)
        self.assertEqual({node_c}, res)
