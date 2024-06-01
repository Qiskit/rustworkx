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


class TestTransitivity(unittest.TestCase):
    def test_transitivity_directed(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(5)))
        graph.add_edges_from_no_data([(0, 1), (0, 2), (0, 3), (1, 2)])
        res = rustworkx.transitivity(graph)
        self.assertEqual(res, 3 / 10)

    def test_transitivity_triangle_directed(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(3)))
        graph.add_edges_from_no_data([(0, 1), (0, 2), (1, 2)])
        res = rustworkx.transitivity(graph)
        self.assertEqual(res, 0.5)

    def test_transitivity_fulltriangle_directed(self):
        graph = rustworkx.PyDiGraph()
        graph.add_nodes_from(list(range(3)))
        graph.add_edges_from_no_data([(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)])
        res = rustworkx.transitivity(graph)
        self.assertEqual(res, 1.0)

    def test_transitivity_empty_directed(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.transitivity(graph)
        self.assertEqual(res, 0.0)
