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


class TestHeavyHexGraph(unittest.TestCase):

    def test_heavy_hex_graph_5(self):
        d = 5
        graph = retworkx.generators.heavy_hex_graph(d)
        self.assertEqual(len(graph), (5 * d * d - 2 * d - 1) / 2)
        self.assertEqual(len(graph.edges()),
                         2 * d * (d - 1) + (d + 1) * (d - 1))

    def test_heavy_hex_graph_3(self):
        d = 3
        graph = retworkx.generators.heavy_hex_graph(d)
        self.assertEqual(len(graph), (5 * d * d - 2 * d - 1) / 2)
        self.assertEqual(len(graph.edges()),
                         2 * d * (d - 1) + (d + 1) * (d - 1))

    def test_heavy_hex_graph_even_d(self):
        with self.assertRaises(IndexError):
            retworkx.generators.heavy_hex_graph(2)
