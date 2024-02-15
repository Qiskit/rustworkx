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

class TestBisimulation(unittest.TestCase):
    def test_is_maximum_bisimulation(self):
        for graph, solution in zip(self.graphs, self.reference_solution):
            res = rustworkx.digraph_maximum_bisimulation(graph)
            res_format = [tuple(element) for element in res]
            for element in res_format:
                self.assertTrue(all([any([el in sol for sol in solution]) for el in element]))

    def test_failure(self):
        with self.assertRaises(TypeError):
            rustworkx.digraph_maximum_bisimulation(rustworkx.PyGraph())

    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.digraph_maximum_bisimulation(graph)
        self.assertEqual(res, [])

    def setUp(self):
        graphs = []
        reference_solution = []

        graphs.append(rustworkx.PyDiGraph())
        graphs[0].add_nodes_from(list(range(5)))
        graphs[0].add_edges_from_no_data([(0, 1), (0, 2), (0, 3), (1, 2)])

        reference_solution.append({(1,), (2,3,4), (0,)})


        graphs.append(rustworkx.PyDiGraph())
        graphs[1].add_nodes_from(list(range(5)))
        graphs[1].add_edges_from([(0, 1, "C"), (0, 2, "D"), (0, 3, "B"), (1, 2, "G")])

        reference_solution.append({(1,), (2,3,4), (0,)})

        graphs.append(rustworkx.PyDiGraph())
        graphs[2].add_nodes_from(list(range(4)))
        graphs[2].add_edges_from_no_data([(0, 0), (1, 1), (2, 2), (3, 3)])

        reference_solution.append({(0, 1, 2, 3)})

        graphs.append(rustworkx.PyDiGraph())
        graphs[3].add_nodes_from(list(range(8)))
        graphs[3].add_edges_from_no_data([(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)])

        reference_solution.append({(0, 4), (3, 7), (2, 6), (1, 5)})

        graphs.append(rustworkx.PyDiGraph())
        graphs[4].add_nodes_from(list(range(12)))
        graphs[4].add_edges_from_no_data([(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11)])

        reference_solution.append({(0, 4, 8), (3, 7, 11), (2, 6, 10), (1, 5, 9)})

        graphs.append(rustworkx.PyDiGraph())
        graphs[5].add_nodes_from(list(range(12)))
        graphs[5].add_edges_from_no_data([(0, 1), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (11, 5)])

        reference_solution.append({(8,), (3, 7), (2, 6), (1, 5), (0,), (4, 11), (10,), (9,)})

        self.graphs = graphs
        self.reference_solution = reference_solution
