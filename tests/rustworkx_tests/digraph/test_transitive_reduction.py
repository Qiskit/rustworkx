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


class TestTransitiveReduction(unittest.TestCase):
    def test_tr1(self):
        graph = rustworkx.PyDiGraph()
        a = graph.add_node("a")
        b = graph.add_node("b")
        c = graph.add_node("c")
        d = graph.add_node("d")
        e = graph.add_node("e")
        graph.add_edges_from(
            [(a, b, 1), (a, d, 1), (a, c, 1), (a, e, 1), (b, d, 1), (c, d, 1), (c, e, 1), (d, e, 1)]
        )
        tr, _ = rustworkx.transitive_reduction(graph)
        self.assertCountEqual(list(tr.edge_list()), [(0, 2), (0, 1), (1, 3), (2, 3), (3, 4)])

    def test_tr2(self):
        graph2 = rustworkx.PyDiGraph()
        a = graph2.add_node("a")
        b = graph2.add_node("b")
        c = graph2.add_node("c")
        graph2.add_edges_from(
            [
                (a, b, 1),
                (b, c, 1),
                (a, c, 1),
            ]
        )
        tr2, _ = rustworkx.transitive_reduction(graph2)
        self.assertCountEqual(list(tr2.edge_list()), [(0, 1), (1, 2)])

    def test_tr3(self):
        graph3 = rustworkx.PyDiGraph()
        graph3.add_nodes_from([0, 1, 2, 3])
        graph3.add_edges_from([(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1)])
        tr3, _ = rustworkx.transitive_reduction(graph3)
        self.assertCountEqual(list(tr3.edge_list()), [(0, 1), (1, 2), (1, 3)])

    def test_tr_with_deletion(self):
        graph = rustworkx.PyDiGraph()
        a = graph.add_node("a")
        b = graph.add_node("b")
        c = graph.add_node("c")
        d = graph.add_node("d")
        e = graph.add_node("e")

        graph.add_edges_from(
            [(a, b, 1), (a, d, 1), (a, c, 1), (a, e, 1), (b, d, 1), (c, d, 1), (c, e, 1), (d, e, 1)]
        )

        graph.remove_node(3)

        tr, index_map = rustworkx.transitive_reduction(graph)

        self.assertCountEqual(list(tr.edge_list()), [(0, 1), (0, 2), (2, 3)])
        self.assertEqual(index_map[4], 3)

    def test_tr_error(self):
        digraph = rustworkx.generators.directed_cycle_graph(1000)
        with self.assertRaises(ValueError):
            rustworkx.transitive_reduction(digraph)
