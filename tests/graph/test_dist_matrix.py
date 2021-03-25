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

import numpy

import retworkx


class TestDistanceMatrix(unittest.TestCase):

    def test_graph_distance_matrix(self):
        graph = retworkx.PyGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data(
            [(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = retworkx.graph_distance_matrix(graph)
        expected = numpy.array([[0., 1., 2., 3., 3., 2., 1.],
                                [1., 0., 1., 2., 3., 3., 2.],
                                [2., 1., 0., 1., 2., 3., 3.],
                                [3., 2., 1., 0., 1., 2., 3.],
                                [3., 3., 2., 1., 0., 1., 2.],
                                [2., 3., 3., 2., 1., 0., 1.],
                                [1., 2., 3., 3., 2., 1., 0.]])
        self.assertTrue(numpy.array_equal(dist, expected))

    def test_graph_distance_matrix_parallel(self):
        graph = retworkx.PyGraph()
        graph.add_nodes_from(list(range(7)))
        graph.add_edges_from_no_data(
            [(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        dist = retworkx.graph_distance_matrix(graph, parallel_threshold=5)
        expected = numpy.array([[0., 1., 2., 3., 3., 2., 1.],
                                [1., 0., 1., 2., 3., 3., 2.],
                                [2., 1., 0., 1., 2., 3., 3.],
                                [3., 2., 1., 0., 1., 2., 3.],
                                [3., 3., 2., 1., 0., 1., 2.],
                                [2., 3., 3., 2., 1., 0., 1.],
                                [1., 2., 3., 3., 2., 1., 0.]])
        self.assertTrue(numpy.array_equal(dist, expected))
        