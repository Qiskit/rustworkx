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


class TestIsolates(unittest.TestCase):
    def test_isolates(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(4))
        graph.add_edge(0, 1, None)
        res = rustworkx.isolates(graph)
        self.assertEqual(res, [2, 3])

    def test_isolates_with_holes(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(range(4))
        graph.add_edge(0, 1, None)
        graph.remove_node(2)
        res = rustworkx.isolates(graph)
        self.assertEqual(res, [3])

    def test_isolates_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.isolates(graph)
        self.assertEqual(res, [])

    def test_isolates_outgoing_star(self):
        graph = rustworkx.generators.directed_star_graph(5)
        res = rustworkx.isolates(graph)
        self.assertEqual(res, [])

    def test_isolates_incoming_star(self):
        graph = rustworkx.generators.directed_star_graph(5, inward=True)
        res = rustworkx.isolates(graph)
        self.assertEqual(res, [])
