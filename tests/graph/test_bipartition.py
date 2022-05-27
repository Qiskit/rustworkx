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


class TestBipartition(unittest.TestCase):
    def setUp(self):
        self.line = retworkx.PyGraph()
        a = self.line.add_node(0)
        b = self.line.add_node(1)
        c = self.line.add_node(2)
        d = self.line.add_node(3)
        e = self.line.add_node(4)
        f = self.line.add_node(5)

        self.line.add_edges_from(
            [
                (a, b, 1),
                (b, c, 1),
                (c, d, 1),
                (d, e, 1),
                (e, f, 1),
            ]
        )

        self.tree = retworkx.PyGraph()
        a = self.tree.add_node(0)
        b = self.tree.add_node(1)
        c = self.tree.add_node(2)
        d = self.tree.add_node(3)
        e = self.tree.add_node(4)
        f = self.tree.add_node(5)

        self.tree.add_edges_from(
            [
                (a, b, 1),
                (a, d, 1),
                (c, d, 1),
                (a, f, 1),
                (d, e, 1),
            ]
        )

    def test_one_balanced_edge_tree(self):
        balanced_edges = retworkx.bipartition_tree(
            self.tree,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3.0,
            0.2,
        )
        self.assertEqual(len(balanced_edges), 1)

        # Since this is already a spanning tree, bipartition_graph should
        # behave identically. That is, it should be invariant to weight_fn
        graph_balanced_edges = retworkx.bipartition_graph(
            self.tree,
            lambda _: 1,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3.0,
            0.2,
        )
        self.assertEqual(balanced_edges, graph_balanced_edges)

    def test_two_balanced_edges_tree(self):
        balanced_edges = retworkx.bipartition_tree(
            self.tree,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3.0,
            0.5,
        )
        self.assertEqual(len(balanced_edges), 1)

        graph_balanced_edges = retworkx.bipartition_graph(
            self.tree,
            lambda _: 1,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3.0,
            0.5,
        )
        self.assertEqual(balanced_edges, graph_balanced_edges)

    def test_three_balanced_edges_line(self):
        balanced_edges = retworkx.bipartition_tree(
            self.line,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3.0,
            0.5,
        )
        self.assertEqual(len(balanced_edges), 3)

        graph_balanced_edges = retworkx.bipartition_graph(
            self.line,
            lambda _: 1,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3.0,
            0.5,
        )
        self.assertEqual(balanced_edges, graph_balanced_edges)

    def test_one_balanced_edges_line(self):
        balanced_edges = retworkx.bipartition_tree(
            self.line,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3.0,
            0.01,
        )
        self.assertEqual(len(balanced_edges), 1)

        graph_balanced_edges = retworkx.bipartition_graph(
            self.line,
            lambda _: 1,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3.0,
            0.01,
        )
        self.assertEqual(balanced_edges, graph_balanced_edges)
