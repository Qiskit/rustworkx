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


class TestLayers(unittest.TestCase):
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
        input_nodes = [qr_0_in, qr_1_in, cr_0_in, cr_1_in]

        h_gate = dag.add_child(qr_0_in, "h", "qr[0]")
        cx_gate = dag.add_child(h_gate, "cx", "qr[0]")
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

        res = rustworkx.layers(dag, input_nodes)
        expected = [
            ["qr[0]", "qr[1]", "cr[0]", "cr[1]"],
            ["h"],
            ["cx"],
            ["measure"],
            ["x"],
            ["measure", "measure"],
            ["cr[1]", "qr[1]", "cr[0]", "qr[0]"],
        ]
        self.assertEqual(expected, res)

    def test_first_layer_invalid_node(self):
        dag = rustworkx.PyDAG()
        with self.assertRaises(rustworkx.InvalidNode):
            rustworkx.layers(dag, [42])

    def test_dagcircuit_basic_index_output(self):
        dag = rustworkx.PyDAG()
        qr_0_in = dag.add_node("qr[0]")
        qr_0_out = dag.add_node("qr[0]")
        qr_1_in = dag.add_node("qr[1]")
        qr_1_out = dag.add_node("qr[1]")
        cr_0_in = dag.add_node("cr[0]")
        cr_0_out = dag.add_node("cr[0]")
        cr_1_in = dag.add_node("cr[1]")
        cr_1_out = dag.add_node("cr[1]")
        input_nodes = [qr_0_in, qr_1_in, cr_0_in, cr_1_in]

        h_gate = dag.add_child(qr_0_in, "h", "qr[0]")
        cx_gate = dag.add_child(h_gate, "cx", "qr[0]")
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

        res = rustworkx.layers(dag, input_nodes, index_output=True)
        expected = [
            [qr_0_in, qr_1_in, cr_0_in, cr_1_in],
            [h_gate],
            [cx_gate],
            [measure_qr_1],
            [x_gate],
            [measure_qr_1_out, measure_qr_0],
            [cr_1_out, qr_1_out, cr_0_out, qr_0_out],
        ]
        self.assertEqual(expected, res)

    def test_first_layer_invalid_node_index_output(self):
        dag = rustworkx.PyDAG()
        with self.assertRaises(rustworkx.InvalidNode):
            rustworkx.layers(dag, [42], index_output=True)
