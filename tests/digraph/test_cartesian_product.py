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


class TestCartesianProduct(unittest.TestCase):
    def test_null_cartesian_null(self):
        graph_1 = retworkx.PyDiGraph()
        graph_2 = retworkx.PyDiGraph()

        graph_product = retworkx.digraph_cartesian_product(graph_1, graph_2)
        self.assertEqual(len(graph_product.nodes()), 0)
        self.assertEqual(len(graph_product.edge_list()), 0)

    def test_directed_path_2_cartesian_path_2(self):
        graph_1 = retworkx.generators.directed_path_graph(2)
        graph_2 = retworkx.generators.directed_path_graph(2)

        graph_product = retworkx.digraph_cartesian_product(graph_1, graph_2)
        self.assertEqual(len(graph_product.nodes()), 4)
        self.assertEqual(len(graph_product.edge_list()), 4)

    def test_directed_path_2_cartesian_path_3(self):
        graph_1 = retworkx.generators.directed_path_graph(2)
        graph_2 = retworkx.generators.directed_path_graph(3)

        graph_product = retworkx.digraph_cartesian_product(graph_1, graph_2)
        self.assertEqual(len(graph_product.nodes()), 6)
        self.assertEqual(len(graph_product.edge_list()), 7)
