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


class TestSimpleCycles(unittest.TestCase):

    def test_simple_cycles(self):
        edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
        graph = retworkx.PyDiGraph()
        graph.extend_from_edge_list(edges)
        expected = [[0], [0, 1, 2], [0, 2], [1, 2], [2]]
        res = retworkx.simple_cycles(graph)
        print(res)
        self.assertEqual(len(res), len(expected))
        for cycle in res:
            self.assertIn(sorted(cycle), expected)

    def test_mesh_graph(self):
        # Test taken from Table 2 in the Johnson Algorithm paper
        # which shows the number of cycles in a complete graph of
        # 2 to 9 nodes and the time to calculate it on a s370/168
        # The table in question is a benchmark comparing the runtime
        # to tarjan's algorithm, but it gives us a good test with
        # a known value (networkX does this too)
        num_circuits = [1, 5, 20, 84, 409, 2365, 16064]
        num_circuits = [5]
        for n, c in zip([3], num_circuits):
            with self.subTest(n=n):
                graph = retworkx.generators.directed_mesh_graph(n)
                print(graph.edge_list())
                res = retworkx.simple_cycles(graph)
                print(res)
                self.assertEqual(len(res), c)
