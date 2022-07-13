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


class TestAstarDigraph(unittest.TestCase):
    def test_astar_null_heuristic(self):
        g = rustworkx.PyDAG()
        a = g.add_node("A")
        b = g.add_node("B")
        c = g.add_node("C")
        d = g.add_node("D")
        e = g.add_node("E")
        f = g.add_node("F")
        g.add_edge(a, b, 7)
        g.add_edge(c, a, 9)
        g.add_edge(a, d, 14)
        g.add_edge(b, c, 10)
        g.add_edge(d, c, 2)
        g.add_edge(d, e, 9)
        g.add_edge(b, f, 15)
        g.add_edge(c, f, 11)
        g.add_edge(e, f, 6)
        path = rustworkx.digraph_astar_shortest_path(
            g, a, lambda goal: goal == "E", lambda x: float(x), lambda y: 0
        )
        expected = [a, d, e]
        self.assertEqual(expected, path)

    def test_astar_manhattan_heuristic(self):
        g = rustworkx.PyDAG()
        a = g.add_node((0.0, 0.0))
        b = g.add_node((2.0, 0.0))
        c = g.add_node((1.0, 1.0))
        d = g.add_node((0.0, 2.0))
        e = g.add_node((3.0, 3.0))
        f = g.add_node((4.0, 2.0))
        no_path = g.add_node((5.0, 5.0))  # no path to node
        g.add_edge(a, b, 2.0)
        g.add_edge(a, d, 4.0)
        g.add_edge(b, c, 1.0)
        g.add_edge(b, f, 7.0)
        g.add_edge(c, e, 5.0)
        g.add_edge(e, f, 1.0)
        g.add_edge(d, e, 1.0)

        def heuristic_func(f):
            x1, x2 = f
            return abs(x2 - x1)

        def finish_func(node, x):
            return x == g.get_node_data(node)

        expected = [
            [0],
            [0, 1],
            [0, 1, 2],
            [0, 3],
            [0, 3, 4],
            [0, 3, 4, 5],
        ]

        for index, end in enumerate([a, b, c, d, e, f]):
            path = rustworkx.digraph_astar_shortest_path(
                g,
                a,
                lambda finish: finish_func(end, finish),
                lambda x: float(x),
                heuristic_func,
            )
            self.assertEqual(expected[index], path)

        with self.assertRaises(rustworkx.NoPathFound):
            rustworkx.digraph_astar_shortest_path(
                g,
                a,
                lambda finish: finish_func(no_path, finish),
                lambda x: float(x),
                heuristic_func,
            )

    def test_astar_digraph_with_graph_input(self):
        g = rustworkx.PyGraph()
        g.add_node(0)
        with self.assertRaises(TypeError):
            rustworkx.digraph_astar_shortest_path(g, 0, lambda x: x, lambda y: 1, lambda z: 0)

    def test_astar_with_invalid_weights(self):
        g = rustworkx.PyDAG()
        a = g.add_node("A")
        b = g.add_node("B")
        g.add_edge(a, b, 7)
        for invalid_weight in [float("nan"), -1]:
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises(ValueError):
                    rustworkx.digraph_astar_shortest_path(
                        g,
                        a,
                        goal_fn=lambda goal: goal == "B",
                        edge_cost_fn=lambda _: invalid_weight,
                        estimate_cost_fn=lambda _: 0,
                    )
