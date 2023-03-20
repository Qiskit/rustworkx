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


class TestCollectMultiBlock(unittest.TestCase):
    def test_blocks_2q_blocks(self):
        graph = rustworkx.PyDiGraph()
        q0 = graph.add_node({"type": "in", "name": "q0", "groups": []})
        q1 = graph.add_node({"type": "in", "name": "q1", "groups": []})
        q2 = graph.add_node({"type": "in", "name": "q2", "groups": []})
        u1 = graph.add_node({"type": "op", "name": "u1", "groups": [0]})
        u2 = graph.add_node({"type": "op", "name": "u2", "groups": [1]})
        cx_1 = graph.add_node({"type": "op", "name": "cx", "groups": [2, 1]})
        cx_2 = graph.add_node({"type": "op", "name": "cx", "groups": [0, 1]})
        q0_out = graph.add_node({"type": "out", "name": "q0", "groups": []})
        q1_out = graph.add_node({"type": "out", "name": "q1", "groups": []})
        q2_out = graph.add_node({"type": "out", "name": "q2", "groups": []})
        graph.add_edges_from_no_data(
            [
                (q0, u1),
                (q1, u2),
                (q2, cx_1),
                (u2, cx_1),
                (u1, cx_2),
                (cx_1, cx_2),
                (cx_2, q2_out),
                (cx_1, q1_out),
                (cx_2, q0_out),
            ]
        )

        def group_fn(node):
            return set(node["groups"])

        def key_fn(node):
            if node["type"] == "in":
                return "a"
            if node["type"] != "op":
                return "d"
            return "b" + chr(ord("a") + len(node["groups"]))

        def filter_fn(node):
            if node["type"] != "op":
                return None
            return True

        blocks = rustworkx.collect_multi_blocks(graph, 2, key_fn, group_fn, filter_fn)
        self.assertEqual(blocks, [[4, 5], [3, 6]])

    def test_blocks_unprocessed(self):
        graph = rustworkx.PyDiGraph()
        q0 = graph.add_node({"type": "in", "name": "q0", "groups": []})
        q1 = graph.add_node({"type": "in", "name": "q1", "groups": []})
        q2 = graph.add_node({"type": "in", "name": "q2", "groups": []})
        c0 = graph.add_node({"type": "in", "name": "c0", "groups": []})
        cx_1 = graph.add_node({"type": "op", "name": "cx", "groups": [0, 1]})
        cx_2 = graph.add_node({"type": "op", "name": "cx", "groups": [1, 2]})
        measure = graph.add_node({"type": "op", "name": "measure", "groups": [0]})
        cx_3 = graph.add_node({"type": "op", "name": "cx", "groups": [1, 2]})
        x = graph.add_node({"type": "op", "name": "x", "groups": [1]})
        h = graph.add_node({"type": "op", "name": "h", "groups": [2]})
        q0_out = graph.add_node({"type": "out", "name": "q0", "groups": []})
        q1_out = graph.add_node({"type": "out", "name": "q1", "groups": []})
        q2_out = graph.add_node({"type": "out", "name": "q2", "groups": []})
        c0_out = graph.add_node({"type": "out", "name": "c0", "groups": []})
        graph.add_edges_from_no_data(
            [
                (q0, cx_1),
                (q1, cx_1),
                (cx_1, cx_2),
                (q2, cx_2),
                (cx_1, measure),
                (c0, measure),
                (cx_2, cx_3),
                (cx_2, cx_3),
                (cx_3, x),
                (cx_3, h),
                (measure, q0_out),
                (measure, c0_out),
                (x, q1_out),
                (h, q2_out),
            ]
        )

        def group_fn(node):
            return set(node["groups"])

        def key_fn(node):
            if node["type"] == "in":
                return "a"
            if node["type"] != "op":
                return "d"
            if node["name"] == "measure":
                return "d"
            return "b" + chr(ord("a") + len(node["groups"]))

        def filter_fn(node):
            if node["type"] != "op":
                return None
            if node["name"] == "measure":
                return False
            return True

        blocks = rustworkx.collect_multi_blocks(graph, 2, key_fn, group_fn, filter_fn)
        self.assertEqual(blocks, [[4], [5, 7, 8, 9]])

    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        block = rustworkx.collect_multi_blocks(graph, 1, lambda x: x, lambda x: x, lambda x: x)
        self.assertEqual(block, [])

    def test_larger_block(self):
        graph = rustworkx.PyDiGraph()
        q0 = graph.add_node({"type": "in", "name": "q0", "groups": []})
        q1 = graph.add_node({"type": "in", "name": "q1", "groups": []})
        q2 = graph.add_node({"type": "in", "name": "q2", "groups": []})
        q3 = graph.add_node({"type": "in", "name": "q3", "groups": []})
        q4 = graph.add_node({"type": "in", "name": "q4", "groups": []})
        cx_1 = graph.add_node({"type": "op", "name": "cx", "groups": [0, 1]})
        cx_2 = graph.add_node({"type": "op", "name": "cx", "groups": [1, 2]})
        cx_3 = graph.add_node({"type": "op", "name": "cx", "groups": [2, 3]})
        ccx = graph.add_node({"type": "op", "name": "ccx", "groups": [0, 1, 2]})
        cx_4 = graph.add_node({"type": "op", "name": "cx", "groups": [3, 4]})
        cx_5 = graph.add_node({"type": "op", "name": "cx", "groups": [3, 4]})
        q0_out = graph.add_node({"type": "out", "name": "q0", "groups": []})
        q1_out = graph.add_node({"type": "out", "name": "q1", "groups": []})
        q2_out = graph.add_node({"type": "out", "name": "q2", "groups": []})
        q3_out = graph.add_node({"type": "out", "name": "q3", "groups": []})
        q4_out = graph.add_node({"type": "out", "name": "q4", "groups": []})

        graph.add_edges_from(
            [
                (q0, cx_1, "q0"),
                (q1, cx_1, "q1"),
                (cx_1, cx_2, "q1"),
                (q2, cx_2, "q2"),
                (cx_2, cx_3, "q2"),
                (q3, cx_3, "q3"),
                (cx_1, ccx, "q0"),
                (cx_2, ccx, "q1"),
                (cx_3, ccx, "q2"),
                (cx_3, cx_4, "q3"),
                (q4, cx_4, "q4"),
                (cx_4, cx_5, "q3"),
                (cx_4, cx_5, "q4"),
                (ccx, q0_out, "q0"),
                (ccx, q1_out, "q1"),
                (ccx, q2_out, "q2"),
                (cx_5, q3_out, "q3"),
                (cx_5, q4_out, "q4"),
            ]
        )

        def group_fn(node):
            return set(node["groups"])

        def key_fn(node):
            if node["type"] == "in":
                return "a"
            if node["type"] != "op":
                return "d"
            if node["name"] == "measure":
                return "d"
            return "b" + chr(ord("a") + len(node["groups"]))

        def filter_fn(node):
            if node["type"] != "op":
                return None
            if node["name"] == "measure":
                return False
            return True

        blocks = rustworkx.collect_multi_blocks(graph, 4, key_fn, group_fn, filter_fn)
        self.assertEqual([[cx_1, cx_2, cx_3], [ccx], [cx_4, cx_5]], blocks)
