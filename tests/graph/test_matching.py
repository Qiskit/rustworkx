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


class TestMatching(unittest.TestCase):
    def test_valid(self):
        graph = rustworkx.generators.path_graph(4)
        matching = {(0, 1), (2, 3)}
        self.assertTrue(rustworkx.is_maximal_matching(graph, matching))

    def test_not_matching(self):
        graph = rustworkx.generators.path_graph(4)
        matching = {(0, 1), (1, 2), (2, 3)}
        self.assertFalse(rustworkx.is_maximal_matching(graph, matching))

    def test_not_maximal(self):
        graph = rustworkx.generators.path_graph(4)
        matching = {(0, 1)}
        self.assertFalse(rustworkx.is_maximal_matching(graph, matching))

    def test_is_matching_empty(self):
        graph = rustworkx.generators.path_graph(4)
        matching = set()
        self.assertTrue(rustworkx.is_matching(graph, matching))

    def test_is_matching_single_edge(self):
        graph = rustworkx.generators.path_graph(4)
        matching = {(1, 2)}
        self.assertTrue(rustworkx.is_matching(graph, matching))

    def test_is_matching_valid(self):
        graph = rustworkx.generators.path_graph(4)
        matching = {(0, 1), (2, 3)}
        self.assertTrue(rustworkx.is_matching(graph, matching))

    def test_is_matching_invalid(self):
        graph = rustworkx.generators.path_graph(4)
        matching = {(0, 1), (1, 2), (2, 3)}
        self.assertFalse(rustworkx.is_matching(graph, matching))

    def test_is_matching_invalid_edge(self):
        graph = rustworkx.generators.path_graph(4)
        matching = {(0, 3), (1, 2)}
        self.assertFalse(rustworkx.is_matching(graph, matching))
