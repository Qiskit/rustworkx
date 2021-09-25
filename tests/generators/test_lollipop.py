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


class TestLollipopGraph(unittest.TestCase):
    def test_lollipop_graph(self):
        graph = retworkx.generators.lollipop_graph(17, 3)
        self.assertEqual(len(graph), 20)
        self.assertEqual(len(graph.edges()), 139)

    def test_lollipop_graph_weights(self):
        graph = retworkx.generators.lollipop_graph(
            mesh_weights=list(range(17)), path_weights=list(range(17, 20))
        )
        self.assertEqual(len(graph), 20)
        self.assertEqual(list(range(20)), graph.nodes())
        self.assertEqual(len(graph.edges()), 139)

    def test_mesh_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            retworkx.generators.lollipop_graph()
