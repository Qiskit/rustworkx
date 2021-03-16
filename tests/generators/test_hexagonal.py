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


class TestHexagonalGraph(unittest.TestCase):

    def test_hexagonal_graph_2_2(self):
        graph = retworkx.generators.hexagonal_graph(2,2)
        self.assertEqual(len(graph), 16)
        self.assertEqual(len(graph.edges()), 19)

    def test_hexagonal_graph_3_2(self):
        graph = retworkx.generators.hexagonal_graph(3,2)
        self.assertEqual(len(graph), 22)
        self.assertEqual(len(graph.edges()), 27)

    def test_hexagonal_graph_2_4(self):
        graph = retworkx.generators.hexagonal_graph(2,4)
        self.assertEqual(len(graph), 28)
        self.assertEqual(len(graph.edges()), 35)

    def test_hexagonal_no_num(self):
        with self.assertRaises(IndexError):
            retworkx.generators.hexagonal_graph()