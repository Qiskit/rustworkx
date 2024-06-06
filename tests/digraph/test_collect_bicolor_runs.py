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


class TestCollectBicolorRuns(unittest.TestCase):
    def test_cycle(self):
        dag = rustworkx.PyDiGraph()
        dag.extend_from_edge_list([(0, 1), (1, 2), (2, 0)])
        with self.assertRaises(rustworkx.DAGHasCycle):
            rustworkx.collect_bicolor_runs(dag, lambda _: True, lambda _: None)

    def test_filter_function_inner_exception(self):
        dag = rustworkx.PyDiGraph()
        dag.add_node("a")
        dag.add_child(0, "b", None)
        dag.add_child(1, "c", None)

        def fail_function(node):
            raise IndexError("Things fail from time to time")

        with self.assertRaises(IndexError):
            rustworkx.collect_bicolor_runs(dag, fail_function, lambda _: None)

        with self.assertRaises(IndexError):
            rustworkx.collect_bicolor_runs(dag, lambda _: True, fail_function)

    def test_empty(self):
        dag = rustworkx.PyDAG()
        self.assertEqual(
            [],
            rustworkx.collect_bicolor_runs(dag, lambda _: True, lambda _: None),
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
            │          ┌─────────────┐         │
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
        dag = rustworkx.PyDAG()
        q0_list = []
        q1_list = []
        for _ in range(2):
            q0_list.append(dag.add_node("q0"))
            q1_list.append(dag.add_node("q1"))

        cx_gate = dag.add_node("cx")
        cz_gate = dag.add_node("cz")

        dag.add_edge(q0_list[0], cx_gate, "q0")
        dag.add_edge(q1_list[0], cx_gate, "q1")
        dag.add_edge(cx_gate, cz_gate, "q0")
        dag.add_edge(cx_gate, cz_gate, "q1")
        dag.add_edge(cz_gate, q0_list[1], "q0")
        dag.add_edge(cz_gate, q1_list[1], "q1")

        def filter_function(node):
            if node in ["cx", "cz"]:
                return True
            else:
                return None

        def color_function(node):
            print("node name:", node)
            if "q" in node:
                return int(node[1:])
            else:
                return None

        self.assertEqual(
            [["cx", "cz"]],
            rustworkx.collect_bicolor_runs(dag, filter_function, color_function),
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
            │                           ┌─────────────┐
            │                           │             │
            │                           │    q1       │
            │                           │             │
            |                           └──────┬──────┘
            │          ┌─────────────┐         │
        q0  │          │             │         │  q1
            │          │             │         │
            └─────────►│     cx      │◄────────┘
            ┌──────────┤             ├─────────┐
            │          │             │         │
        q0  │          └─────────────┘         │  q1
            │                                  │
            │          ┌─────────────┐         │
            │          │             │         │
            └─────────►│      cz     │◄────────┘
             ┌─────────┤             ├─────────┐
             │         └─────────────┘         │
         q0  │                                 │ q1
             │                                 │
         ┌───▼─────────┐                ┌──────▼──────┐
         │             │                │             │
         │    q0       │                │    y        │
         │             │                │             │
         └─────────────┘                └─────────────┘
                                            | q1
                                            │
                                        ┌───▼─────────┐
                                        │             │
                                        │    q1       │
                                        │             │
                                        └─────────────┘

        Expected: [[h, cx, cz, y]]
        """
        dag = rustworkx.PyDAG()
        q0_list = []
        q1_list = []
        for _ in range(2):
            q0_list.append(dag.add_node("q0"))
            q1_list.append(dag.add_node("q1"))

        h_gate = dag.add_node("h")
        cx_gate = dag.add_node("cx")
        cz_gate = dag.add_node("cz")
        y_gate = dag.add_node("y")

        dag.add_edge(q0_list[0], h_gate, "q0")
        dag.add_edge(h_gate, cx_gate, "q0")
        dag.add_edge(q1_list[0], cx_gate, "q1")
        dag.add_edge(cx_gate, cz_gate, "q0")
        dag.add_edge(cx_gate, cz_gate, "q1")
        dag.add_edge(cz_gate, q0_list[1], "q0")
        dag.add_edge(cz_gate, y_gate, "q1")
        dag.add_edge(y_gate, q1_list[1], "q1")

        def filter_function(node):
            if node in ["cx", "cz", "h", "y"]:
                return True
            else:
                return None

        def color_function(node):
            if "q" in node:
                return int(node[1:])
            else:
                return None

        self.assertEqual(
            [["h", "cx", "cz", "y"]],
            rustworkx.collect_bicolor_runs(dag, filter_function, color_function),
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
            │          ┌─────────────┐         │
            │          │             │         │
            └─────────►│  barrier    │◄────────┘
             ┌─────────┤             ├─────────┐
             │         └─────────────┘         │
         q0  │                                 │ q1
             │                                 │
             │         ┌─────────────┐         │
             │         │             │         │
             └────────►│     cz      │◄────────┘
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
        dag = rustworkx.PyDAG()
        q0_list = []
        q1_list = []
        for _ in range(2):
            q0_list.append(dag.add_node("q0"))
            q1_list.append(dag.add_node("q1"))

        cx_gate = dag.add_node("cx")
        barrier = dag.add_node("barrier")
        cz_gate = dag.add_node("cz")

        # CX
        dag.add_edge(q0_list[0], cx_gate, "q0")
        dag.add_edge(q1_list[0], cx_gate, "q1")
        # Barrier
        dag.add_edge(cx_gate, barrier, "q0")
        dag.add_edge(cx_gate, barrier, "q1")
        # CZ
        dag.add_edge(barrier, cz_gate, "q0")
        dag.add_edge(barrier, cz_gate, "q1")
        dag.add_edge(cz_gate, q0_list[1], "q0")
        dag.add_edge(cz_gate, q1_list[1], "q1")

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
                return None

        self.assertEqual(
            [["cx"], ["cz"]],
            rustworkx.collect_bicolor_runs(dag, filter_function, color_function),
        )

    def test_color_with_ignored_edge(self):
        """
        Input:
        ┌─────────────┐                 ┌─────────────┐
        │             │                 │             │
        │    q0       │                 │    c0       │
        │             │                 │             │
        └───┬─────────┘                 └──────┬──────┘
            │          ┌─────────────┐         │
        q0  │          │             │         │  c0
            └─────────►│     rx      │◄────────┘
            ┌──────────┤             ├─────────┐
        q0  │          └─────────────┘         │  c0
            │                                  │
            │          ┌─────────────┐         │
            │          │             │         │
            └─────────►│  barrier    │         │
             ┌─────────┤             │         │
             │         └─────────────┘         │
         q0  │                                 │ c0
             │                                 │
             │         ┌─────────────┐         │
             │         │             │         │
             └────────►│     rz      │◄────────┘
            ┌──────────┤             ├─────────┐
        q0  │          └─────────────┘         │  c0
            │                                  │
        ┌───▼─────────┐                 ┌──────▼──────┐
        │             │                 │             │
        │    q0       │                 │    c0       │
        │             │                 │             │
        └─────────────┘                 └─────────────┘

        Expected: []
        """
        dag = rustworkx.PyDAG()
        q0_list = []
        c0_list = []
        for _ in range(2):
            q0_list.append(dag.add_node("q0"))
            c0_list.append(dag.add_node("c0"))

        rx_gate = dag.add_node("rx")
        barrier = dag.add_node("barrier")
        rz_gate = dag.add_node("rz")

        # RX
        dag.add_edge(q0_list[0], rx_gate, "q0")
        dag.add_edge(c0_list[0], rx_gate, "c0")
        # Barrier
        dag.add_edge(rx_gate, barrier, "q0")
        # RZ
        dag.add_edge(barrier, rz_gate, "q0")
        dag.add_edge(rx_gate, rz_gate, "c0")
        dag.add_edge(rz_gate, q0_list[1], "q0")
        dag.add_edge(rz_gate, c0_list[1], "c0")

        def filter_function(node):
            if node == "barrier":
                return False
            else:
                return None

        def color_function(node):
            if "q" in node:
                return int(node[1:])
            else:
                return None

        self.assertEqual(
            [],
            rustworkx.collect_bicolor_runs(dag, filter_function, color_function),
        )
