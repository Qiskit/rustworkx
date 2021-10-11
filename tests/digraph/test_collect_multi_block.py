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


class TestCollectMultiBlock(unittest.TestCase):
    def test_blocks_2q_blocks(self):
        graph = retworkx.PyDiGraph()
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

        blocks = retworkx.collect_multi_blocks(
            graph, 2, key_fn, group_fn, filter_fn
        )
        self.assertEqual(blocks, [[4, 5], [3, 6]])

    def test_blocks_unprocessed(self):
        graph = retworkx.PyDiGraph()
        q0 = graph.add_node({"type": "in", "name": "q0", "groups": []})
        q1 = graph.add_node({"type": "in", "name": "q1", "groups": []})
        q2 = graph.add_node({"type": "in", "name": "q2", "groups": []})
        c0 = graph.add_node({"type": "in", "name": "c0", "groups": []})
        cx_1 = graph.add_node({"type": "op", "name": "cx", "groups": [0, 1]})
        cx_2 = graph.add_node({"type": "op", "name": "cx", "groups": [1, 2]})
        measure = graph.add_node(
            {"type": "op", "name": "measure", "groups": [0]}
        )
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

        blocks = retworkx.collect_multi_blocks(
            graph, 2, key_fn, group_fn, filter_fn
        )
        self.assertEqual(blocks, [[4], [5, 7, 8, 9]])
