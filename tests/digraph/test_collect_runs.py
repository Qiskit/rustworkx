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


class TestCollectRuns(unittest.TestCase):
    def test_dagcircuit_basic(self):
        dag = rustworkx.PyDAG()
        qr_0_in = dag.add_node("qr[0]")
        qr_0_out = dag.add_node("qr[0]")
        qr_1_in = dag.add_node("qr[1]")
        qr_1_out = dag.add_node("qr[1]")
        cr_0_in = dag.add_node("cr[0]")
        cr_0_out = dag.add_node("cr[0]")
        cr_1_in = dag.add_node("cr[1]")
        cr_1_out = dag.add_node("cr[1]")

        h_gate = dag.add_child(qr_0_in, "h", "qr[0]")
        x_gate = dag.add_child(h_gate, "x", "qr[0]")
        cx_gate = dag.add_child(x_gate, "cx", "qr[0]")
        dag.add_edge(qr_1_in, cx_gate, "qr[1]")

        measure_qr_1 = dag.add_child(cx_gate, "measure", "qr[1]")
        dag.add_edge(cr_1_in, measure_qr_1, "cr[1]")
        x_gate = dag.add_child(measure_qr_1, "x", "qr[1]")
        dag.add_edge(measure_qr_1, x_gate, "cr[1]")
        dag.add_edge(cr_0_in, x_gate, "cr[0]")

        measure_qr_0 = dag.add_child(cx_gate, "measure", "qr[0]")
        dag.add_edge(measure_qr_0, qr_0_out, "qr[0]")
        dag.add_edge(measure_qr_0, cr_0_out, "cr[0]")
        dag.add_edge(x_gate, measure_qr_0, "cr[0]")

        measure_qr_1_out = dag.add_child(x_gate, "measure", "cr[1]")
        dag.add_edge(x_gate, measure_qr_1_out, "qr[1]")
        dag.add_edge(measure_qr_1_out, qr_1_out, "qr[1]")
        dag.add_edge(measure_qr_1_out, cr_1_out, "cr[1]")

        def filter_function(node):
            return node in ["h", "x"]

        res = rustworkx.collect_runs(dag, filter_function)
        expected = [["h", "x"], ["x"]]
        self.assertEqual(expected, res)

    def test_multiple_successor_edges(self):
        dag = rustworkx.PyDiGraph()
        q0, q1 = dag.add_nodes_from(["q0", "q1"])
        cx_1 = dag.add_child(q0, "cx", "q0")
        dag.add_edge(q1, cx_1, "q1")
        cx_2 = dag.add_child(cx_1, "cx", "q0")
        dag.add_edge(q1, cx_2, "q1")
        cx_3 = dag.add_child(cx_2, "cx", "q0")
        dag.add_edge(q1, cx_3, "q1")

        def filter_function(node):
            return node == "cx"

        res = rustworkx.collect_runs(dag, filter_function)
        self.assertEqual([["cx", "cx", "cx"]], res)

    def test_cycle(self):
        dag = rustworkx.PyDiGraph()
        dag.extend_from_edge_list([(0, 1), (1, 2), (2, 0)])
        with self.assertRaises(rustworkx.DAGHasCycle):
            rustworkx.collect_runs(dag, lambda _: True)

    def test_filter_function_inner_exception(self):
        dag = rustworkx.PyDiGraph()
        dag.add_node("a")
        dag.add_child(0, "b", None)

        def filter_function(node):
            raise IndexError("Things fail from time to time")

        with self.assertRaises(IndexError):
            rustworkx.collect_runs(dag, filter_function)

    def test_empty(self):
        dag = rustworkx.PyDAG()
        self.assertEqual([], rustworkx.collect_runs(dag, lambda _: True))

    def test_h_h_cx(self):
        dag = rustworkx.PyDiGraph()
        q0, q1 = dag.add_nodes_from(["q0", "q1"])
        h_1 = dag.add_child(q0, "h", "q0")
        h_2 = dag.add_child(q1, "h", "q1")
        cx_2 = dag.add_child(h_1, "cx", "q0")
        dag.add_edge(h_2, cx_2, "q1")

        def filter_function(node):
            return node in ["cx", "h"]

        res = rustworkx.collect_runs(dag, filter_function)
        self.assertEqual([["h", "cx"], ["h"]], res)

    def test_cx_h_h_cx(self):
        dag = rustworkx.PyDiGraph()
        q0, q1 = dag.add_nodes_from(["q0", "q1"])
        cx_1 = dag.add_child(q0, "cx", "q0")
        dag.add_edge(q1, cx_1, "q1")
        h_1 = dag.add_child(cx_1, "h", "q0")
        h_2 = dag.add_child(cx_1, "h", "q1")
        cx_2 = dag.add_child(h_1, "cx", "q0")
        dag.add_edge(h_2, cx_2, "q1")

        def filter_function(node):
            return node in ["cx", "h"]

        res = rustworkx.collect_runs(dag, filter_function)
        self.assertEqual([["cx"], ["h", "cx"], ["h"]], res)

    def test_cx_h_cx(self):
        dag = rustworkx.PyDiGraph()
        q0, q1 = dag.add_nodes_from(["q0", "q1"])
        cx_1 = dag.add_child(q0, "cx", "q0")
        dag.add_edge(q1, cx_1, "q1")
        h_1 = dag.add_child(cx_1, "h", "q0")
        cx_2 = dag.add_child(h_1, "cx", "q0")
        dag.add_edge(cx_1, cx_2, "q1")

        def filter_function(node):
            return node in ["cx", "h"]

        res = rustworkx.collect_runs(dag, filter_function)
        self.assertEqual([["cx"], ["h", "cx"]], res)
