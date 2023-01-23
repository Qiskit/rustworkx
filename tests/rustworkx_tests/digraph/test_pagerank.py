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

class TestPageRank(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyDiGraph()
        self.a = self.graph.add_node("A")
        self.b = self.graph.add_node("B")
        self.c = self.graph.add_node("C")
        self.d = self.graph.add_node("D")
        self.d = self.graph.add_node("E")
        self.d = self.graph.add_node("F")
        edge_list = [
            (self.a, self.b, 1),
            (self.a, self.c, 1),
            (self.c, self.a, 2),
            (self.c, self.b, 2),
            (self.c, self.e, 1),
            (self.d, self.e, 1),
            (self.d, self.f, 2),
            (self.e, self.d, 1),
            (self.e, self.f, 2),
            (self.f, self.d, 1),
        ]
        self.graph.add_edges_from(edge_list)
    
    def test_pagerank(self):
        ranks = rustworkx.pagerank(self.graph)
        print(ranks)