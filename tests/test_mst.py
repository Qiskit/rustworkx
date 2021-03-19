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


class TestMinimumSpanningTree(unittest.TestCase):
    def setUp(self):
        self.graph = retworkx.PyGraph()
        self.a = self.graph.add_node("A")
        self.b = self.graph.add_node("B")
        self.c = self.graph.add_node("C")
        self.d = self.graph.add_node("D")
        self.e = self.graph.add_node("E")
        self.f = self.graph.add_node("F")
        edge_list = [
            (self.a, self.b, 7),
            (self.c, self.a, 9),
            (self.a, self.d, 14),
            (self.b, self.c, 10),
            (self.d, self.c, 2),
            (self.d, self.e, 9),
            (self.b, self.f, 15),
            (self.c, self.f, 11),
            (self.e, self.f, 6),
        ]
        self.graph.add_edges_from(edge_list)

    def test_kruskal(self):
        path = retworkx.minimum_spanning_tree_edges(self.graph)
        print("Unit Test {}".format(path))
        # self.assertEqual(1, path)

if __name__ == "__main__":
    unittest.main()