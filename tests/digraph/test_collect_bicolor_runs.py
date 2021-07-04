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


class TestCollectBicolorRuns(unittest.TestCase):
    def test_cycle(self):
        dag = retworkx.PyDiGraph()
        dag.extend_from_edge_list([(0, 1), (1, 2), (2, 0)])
        with self.assertRaises(retworkx.DAGHasCycle):
            retworkx.collect_bicolor_runs(dag, lambda _: True, lambda _: -1)

    def test_filter_function_inner_exception(self):
        dag = retworkx.PyDiGraph()
        dag.add_node("a")
        dag.add_child(0, "b", None)
        dag.add_child(1, "c", None)

        def fail_function(node):
            raise IndexError("Things fail from time to time")

        with self.assertRaises(IndexError):
            retworkx.collect_bicolor_runs(dag, fail_function, lambda _: -1)

        with self.assertRaises(IndexError):
            retworkx.collect_bicolor_runs(dag, lambda _: True, fail_function)

    def test_empty(self):
        dag = retworkx.PyDAG()
        self.assertEqual(
            [], retworkx.collect_bicolor_runs(dag, lambda _: True, lambda _: -1)
        )

    def test_single_color(self):
        """
        Input:
        ┌──────┐      ┌─────┐     ┌──────┐     ┌─────┐     ┌──────┐
        │      │      │     │     │      │     │     │     │      │
        │      │  q0  │     │ q0  │      │ q0  │     │ q0  │      │
        │  q0  ├─────►│  h  ├────►│  q0  ├────►│  x  ├────►│  q0  |
        │      │      │     │     │      │     │     │     │      │
        │      │      │     │     │      │     │     │     │      │
        └──────┘      └─────┘     └──────┘     └─────┘     └──────┘

        Expected: [[h, x]]
        """
        dag = retworkx.PyDAG()
        q0_first = dag.add_node("q0")
        q0_second = dag.add_node("q0")
        q0_third = dag.add_node("q0")
        h_gate = dag.add_node("h")
        x_gate = dag.add_node("x")

        dag.add_edge(q0_first, h_gate, "q0")
        dag.add_edge(h_gate, q0_second, "q0")
        dag.add_edge(q0_second, x_gate, "q0")
        dag.add_edge(x_gate, q0_third, "q0")

        def filter_function(node):
            if node in ["h", "x"]:
                return True
            else:
                return None

        def color_function(node):
            if "q" in node:
                return int(node[1:])
            else:
                return -1

        self.assertEqual(
            [["h", "x"]],
            retworkx.collect_bicolor_runs(dag, filter_function, color_function),
        )

    def test_two_colors(self):
        """
        Input:
        ┌─────────────┐                 ┌─────────────┐
        │             │                 │             │
        │    q0       │                 │    q1       │
        │             │                 │             │
        └───┬─────────┘                 └──────┬──────┘
            │          ┌─────────────┐         │
        q0  │          │             │         │  q1
            │          │             │         │
            └─────────►│     cx      │◄────────┘
            ┌──────────┤             ├─────────┐
            │          │             │         │
        q0  │          └─────────────┘         │  q1
            │                                  │
        ┌───▼─────────┐                 ┌──────▼──────┐
        │             │                 │             │
        │    q0       │                 │    q1       │
        │             │                 │             │
        └───┬─────────┘                 └──────┬──────┘
        q0  │          ┌─────────────┐         │  q1
            │          │             │         │
            └─────────►│      cz     │◄────────┘
             ┌─────────┤             ├─────────┐
             │         └─────────────┘         │
         q0  │                                 │ q1
             │                                 │
         ┌───▼─────────┐                ┌──────▼──────┐
         │             │                │             │
         │    q0       │                │    q1       │
         │             │                │             │
         └─────────────┘                └─────────────┘

        Expected: [[cx, cz]]
        """
        dag = retworkx.PyDAG()
        q0_list = []
        q1_list = []
        for _ in range(3):
            q0_list.append(dag.add_node("q0"))
            q1_list.append(dag.add_node("q1"))

        cx_gate = dag.add_node("cx")
        cz_gate = dag.add_node("cz")

        dag.add_edge(q0_list[0], cx_gate, "q0")
        dag.add_edge(q1_list[0], cx_gate, "q1")
        dag.add_edge(cx_gate, q0_list[1], "q0")
        dag.add_edge(cx_gate, q1_list[1], "q1")
        dag.add_edge(q0_list[1], cz_gate, "q0")
        dag.add_edge(q1_list[1], cz_gate, "q1")
        dag.add_edge(cz_gate, q0_list[2], "q0")
        dag.add_edge(cz_gate, q1_list[2], "q1")

        def filter_function(node):
            if node in ["cx", "cz"]:
                return True
            else:
                return None

        def color_function(node):
            if "q" in node:
                return int(node[1:])
            else:
                return -1

        self.assertEqual(
            [["cx", "cz"]],
            retworkx.collect_bicolor_runs(dag, filter_function, color_function),
        )

    def test_two_colors_with_pending(self):
        """
        Input:
        ┌─────────────┐
        │             │
        │    q0       │
        │             │
        └───┬─────────┘
            | q0
            │
        ┌───▼─────────┐
        │             │
        │    h        │
        │             │
        └───┬─────────┘
            | q0
            │
        ┌───▼─────────┐                 ┌─────────────┐
        │             │                 │             │
        │    q0       │                 │    q1       │
        │             │                 │             │
        └───┬─────────┘                 └──────┬──────┘
            │          ┌─────────────┐         │
        q0  │          │             │         │  q1
            │          │             │         │
            └─────────►│     cx      │◄────────┘
            ┌──────────┤             ├─────────┐
            │          │             │         │
        q0  │          └─────────────┘         │  q1
            │                                  │
        ┌───▼─────────┐                 ┌──────▼──────┐
        │             │                 │             │
        │    q0       │                 │    q1       │
        │             │                 │             │
        └───┬─────────┘                 └──────┬──────┘
        q0  │          ┌─────────────┐         │  q1
            │          │             │         │
            └─────────►│      cz     │◄────────┘
             ┌─────────┤             ├─────────┐
             │         └─────────────┘         │
         q0  │                                 │ q1
             │                                 │
         ┌───▼─────────┐                ┌──────▼──────┐
         │             │                │             │
         │    q0       │                │    q1       │
         │             │                │             │
         └─────────────┘                └─────────────┘
                                            | q1
                                            │
                                        ┌───▼─────────┐
                                        │             │
                                        │    y        │
                                        │             │
                                        └─────────────┘
                                            | q1
                                            │
                                        ┌───▼─────────┐
                                        │             │
                                        │    q1       │
                                        │             │
                                        └─────────────┘

        Expected: [[h, cx, cz, y]]
        """
        dag = retworkx.PyDAG()
        q0_list = []
        q1_list = []
        for _ in range(4):
            q0_list.append(dag.add_node("q0"))
            q1_list.append(dag.add_node("q1"))

        h_gate = dag.add_node("h")
        cx_gate = dag.add_node("cx")
        cz_gate = dag.add_node("cz")
        y_gate = dag.add_node("y")

        dag.add_edge(q0_list[0], h_gate, "q0")
        dag.add_edge(h_gate, q0_list[1], "q0")
        dag.add_edge(q0_list[1], cx_gate, "q0")
        dag.add_edge(q1_list[0], cx_gate, "q1")
        dag.add_edge(cx_gate, q0_list[2], "q0")
        dag.add_edge(cx_gate, q1_list[1], "q1")
        dag.add_edge(q0_list[2], cz_gate, "q0")
        dag.add_edge(q1_list[1], cz_gate, "q1")
        dag.add_edge(cz_gate, q0_list[3], "q0")
        dag.add_edge(cz_gate, q1_list[2], "q1")
        dag.add_edge(q1_list[2], y_gate, "q1")
        dag.add_edge(y_gate, q1_list[3], "q1")

        def filter_function(node):
            if node in ["cx", "cz", "h", "y"]:
                return True
            else:
                return None

        def color_function(node):
            if "q" in node:
                return int(node[1:])
            else:
                return -1

        self.assertEqual(
            [["h", "cx", "cz", "y"]],
            retworkx.collect_bicolor_runs(dag, filter_function, color_function),
        )

    def test_two_colors_with_barrier(self):
        """
        Input:
        ┌─────────────┐                 ┌─────────────┐
        │             │                 │             │
        │    q0       │                 │    q1       │
        │             │                 │             │
        └───┬─────────┘                 └──────┬──────┘
            │          ┌─────────────┐         │
        q0  │          │             │         │  q1
            └─────────►│     cx      │◄────────┘
            ┌──────────┤             ├─────────┐
        q0  │          └─────────────┘         │  q1
            │                                  │
        ┌───▼─────────┐                 ┌──────▼──────┐
        │             │                 │             │
        │    q0       │                 │    q1       │
        │             │                 │             │
        └───┬─────────┘                 └──────┬──────┘
        q0  │          ┌─────────────┐         │  q1
            │          │             │         │
            └─────────►│  barrier    │◄────────┘
             ┌─────────┤             ├─────────┐
             │         └─────────────┘         │
         q0  │                                 │ q1
             │                                 │
        ┌────▼────────┐                 ┌──────▼──────┐
        │             │                 │             │
        │    q0       │                 │    q1       │
        │             │                 │             │
        └───┬─────────┘                 └──────┬──────┘
            │          ┌─────────────┐         │
        q0  │          │             │         │  q1
            └─────────►│     cz      │◄────────┘
            ┌──────────┤             ├─────────┐
        q0  │          └─────────────┘         │  q1
            │                                  │
        ┌───▼─────────┐                 ┌──────▼──────┐
        │             │                 │             │
        │    q0       │                 │    q1       │
        │             │                 │             │
        └─────────────┘                 └─────────────┘

        Expected: [[cx], [cz]]
        """
        dag = retworkx.PyDAG()
        q0_list = []
        q1_list = []
        for _ in range(4):
            q0_list.append(dag.add_node("q0"))
            q1_list.append(dag.add_node("q1"))

        cx_gate = dag.add_node("cx")
        barrier = dag.add_node("barrier")
        cz_gate = dag.add_node("cz")

        # CX
        dag.add_edge(q0_list[0], cx_gate, "q0")
        dag.add_edge(q1_list[0], cx_gate, "q1")
        dag.add_edge(cx_gate, q0_list[1], "q0")
        dag.add_edge(cx_gate, q1_list[1], "q1")
        # Barrier
        dag.add_edge(q0_list[1], barrier, "q0")
        dag.add_edge(q1_list[1], barrier, "q1")
        dag.add_edge(barrier, q0_list[2], "q0")
        dag.add_edge(barrier, q1_list[2], "q1")
        # CZ
        dag.add_edge(q0_list[2], cz_gate, "q0")
        dag.add_edge(q1_list[2], cz_gate, "q1")
        dag.add_edge(cz_gate, q0_list[3], "q0")
        dag.add_edge(cz_gate, q1_list[3], "q1")

        def filter_function(node):
            if node in ["cx", "cz"]:
                return True
            elif node == "barrier":
                return False
            else:
                return None

        def color_function(node):
            if "q" in node:
                return int(node[1:])
            else:
                return -1

        self.assertEqual(
            [["cx"], ["cz"]],
            retworkx.collect_bicolor_runs(dag, filter_function, color_function),
        )
