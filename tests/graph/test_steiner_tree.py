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

import pprint
import unittest

import retworkx


class TestMetricClosure(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyGraph()
        self.graph.add_node(None)
        self.graph.extend_from_weighted_edge_list(
            [
                (1, 2, 10),
                (2, 3, 10),
                (3, 4, 10),
                (4, 5, 10),
                (5, 6, 10),
                (2, 7, 1),
                (7, 5, 1),
            ]
        )
        self.graph.remove_node(0)

    def test_metric_closure(self):
        closure_graph = retworkx.metric_closure(self.graph, weight_fn=float)
        expected_edges = [
            (1, 2, (10.0, [1, 2])),
            (1, 3, (20.0, [1, 2, 3])),
            (1, 4, (22.0, [1, 2, 7, 5, 4])),
            (1, 5, (12.0, [1, 2, 7, 5])),
            (1, 6, (22.0, [1, 2, 7, 5, 6])),
            (1, 7, (11.0, [1, 2, 7])),
            (2, 3, (10.0, [2, 3])),
            (2, 4, (12.0, [2, 7, 5, 4])),
            (2, 5, (2.0, [2, 7, 5])),
            (2, 6, (12, [2, 7, 5, 6])),
            (2, 7, (1.0, [2, 7])),
            (3, 4, (10.0, [3, 4])),
            (3, 5, (12.0, [3, 2, 7, 5])),
            (3, 6, (22.0, [3, 2, 7, 5, 6])),
            (3, 7, (11.0, [3, 2, 7])),
            (4, 5, (10.0, [4, 5])),
            (4, 6, (20.0, [4, 5, 6])),
            (4, 7, (11.0, [4, 5, 7])),
            (5, 6, (10.0, [5, 6])),
            (5, 7, (1.0, [5, 7])),
            (6, 7, (11.0, [6, 5, 7])),
        ]
        edges = list(closure_graph.weighted_edge_list())
        for edge in expected_edges:
            found = False
            if edge in edges:
                found = True
            if not found:

                if (
                    edge[1],
                    edge[0],
                    (edge[2][0], list(reversed(edge[2][1]))),
                ) in edges:
                    found = True
            if not found:
                self.fail(
                    f"edge: {edge} nor it's reverse not found in metric "
                    f"closure output:\n{pprint.pformat(edges)}"
                )

    def test_not_connected_metric_closure(self):
        self.graph.add_node(None)
        with self.assertRaises(ValueError):
            retworkx.metric_closure(self.graph, weight_fn=float)

    def test_partially_connected_metric_closure(self):
        graph = retworkx.PyGraph()
        graph.add_node(None)
        graph.extend_from_weighted_edge_list(
            [
                (1, 2, 10),
                (2, 3, 10),
                (3, 4, 10),
                (4, 5, 10),
                (5, 6, 10),
                (2, 7, 1),
                (7, 5, 1),
            ]
        )
        graph.extend_from_weighted_edge_list(
            [
                (0, 8, 20),
                (0, 9, 20),
                (0, 10, 20),
                (8, 10, 10),
                (9, 10, 5),
            ]
        )
        with self.assertRaises(ValueError):
            retworkx.metric_closure(graph, weight_fn=float)

    def test_metric_closure_empty_graph(self):
        graph = retworkx.PyGraph()
        closure = retworkx.metric_closure(graph, weight_fn=float)
        self.assertEqual([], closure.weighted_edge_list())
