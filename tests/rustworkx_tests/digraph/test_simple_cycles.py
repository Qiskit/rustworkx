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


class TestSimpleCycles(unittest.TestCase):
    def test_simple_cycles(self):
        edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
        graph = rustworkx.PyDiGraph()
        graph.extend_from_edge_list(edges)
        expected = [[0], [0, 1, 2], [0, 2], [1, 2], [2]]
        res = list(rustworkx.simple_cycles(graph))
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
        for n, c in zip(range(2, 9), num_circuits):
            with self.subTest(n=n):
                graph = rustworkx.generators.directed_mesh_graph(n)
                res = list(rustworkx.simple_cycles(graph))
                self.assertEqual(len(res), c)

    def test_empty_graph(self):
        self.assertEqual(
            list(rustworkx.simple_cycles(rustworkx.PyDiGraph())),
            [],
        )

    def test_figure_1(self):
        # This graph tests figured 1 from the Johnson's algorithm paper
        for k in range(3, 10):
            with self.subTest(k=k):
                graph = rustworkx.PyDiGraph()
                edge_list = []
                for n in range(2, k + 2):
                    edge_list.append((1, n))
                    edge_list.append((n, k + 2))
                edge_list.append((2 * k + 1, 1))
                for n in range(k + 2, 2 * k + 2):
                    edge_list.append((n, 2 * k + 2))
                    edge_list.append((n, n + 1))
                edge_list.append((2 * k + 3, k + 2))
                for n in range(2 * k + 3, 3 * k + 3):
                    edge_list.append((2 * k + 2, n))
                    edge_list.append((n, 3 * k + 3))
                edge_list.append((3 * k + 3, 2 * k + 2))
                graph.extend_from_edge_list(edge_list)
                cycles = list(rustworkx.simple_cycles(graph))
                self.assertEqual(len(cycles), 3 * k)
