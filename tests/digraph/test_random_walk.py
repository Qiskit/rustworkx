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

import rustworkx as rx


class TestRandomWalk(unittest.TestCase):
    def test_invalid_node_error(self):
        graph = rx.PyDiGraph()
        with self.assertRaises(IndexError):
            rx.generate_random_path_digraph(graph, 0, 10, None)

    def test_zero_degree_early_stop(self):
        graph = rx.PyDiGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, None)
        res = rx.generate_random_path_digraph(graph, 0, 10, None)
        self.assertEqual(res, [0, 1])

    def test_alternating_path(self):
        graph = rx.PyDiGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, None)
        graph.add_edge(1, 0, None)

        self.assertEqual(rx.generate_random_path_digraph(graph, 0, 3, None), [0, 1, 0, 1])
