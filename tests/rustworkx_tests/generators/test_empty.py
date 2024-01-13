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


class TestEmptyGraph(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.generators.empty_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)

    def test_empty_directed_graph(self):
        graph = rustworkx.generators.directed_empty_graph(20)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 0)
        self.assertIsInstance(graph, rustworkx.PyDiGraph)
