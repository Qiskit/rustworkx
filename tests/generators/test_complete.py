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


class TestCompleteGraph(unittest.TestCase):

    def test_complete_graph(self):
        for n in range(2, 10):
            graph = retworkx.generators.complete_graph(n)
            self.assertEqual(len(graph), n)
            self.assertEqual(len(graph.edges()), n*(n-1)//2)

    def test_complete_graph_weights(self):
        graph = retworkx.generators.complete_graph(20, weights=list(range(20)))
        self.assertEqual(len(graph), 20)
        self.assertEqual([x for x in range(20)], graph.nodes())
        self.assertEqual(len(graph.edges()), 190)

    def test_hexagonal_no_weights_num(self):
        with self.assertRaises(IndexError):
            retworkx.generators.complete_graph()