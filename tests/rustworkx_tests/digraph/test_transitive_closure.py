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


class TestTransitivity(unittest.TestCase):
    def test_path_graph(self):
        graph = rx.generators.directed_path_graph(4)
        transitive_closure = rx.transitive_closure_dag(graph)
        expected_edge_list = [(0, 1), (1, 2), (2, 3), (1, 3), (0, 3), (0, 2)]
        self.assertEqual(transitive_closure.edge_list(), expected_edge_list)

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            rx.transitive_closure_dag(rx.PyGraph())

    def test_cycle_error(self):
        graph = rx.PyDiGraph()
        graph.extend_from_edge_list([(0, 1), (1, 0)])
        with self.assertRaises(rx.DAGHasCycle):
            rx.transitive_closure_dag(graph)
