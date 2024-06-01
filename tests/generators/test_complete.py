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


class TestCompleteGraph(unittest.TestCase):
    def test_complete_graph(self):
        for m in [0, 1, 3, 5]:
            graph = rustworkx.generators.complete_graph(m)
            self.assertEqual(len(graph), m)
            self.assertEqual(len(graph.edges()), m * (m - 1) / 2)

    def test_complete_directed_graph(self):
        for m in [0, 1, 3, 5]:
            graph = rustworkx.generators.directed_complete_graph(m)
            self.assertEqual(len(graph), m)
            self.assertEqual(len(graph.edges()), m * (m - 1))
            self.assertIsInstance(graph, rustworkx.PyDiGraph)
