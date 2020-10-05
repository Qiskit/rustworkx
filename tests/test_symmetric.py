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


class TestSymmetric(unittest.TestCase):
    def test_single_neighbor(self):
        digraph = retworkx.PyDiGraph()
        node_a = digraph.add_node('a')
        digraph.add_child(node_a, 'b', {'a': 1})
        digraph.add_child(node_a, 'c', {'a': 2})
        self.assertFalse(digraph.is_symmetric())

    def test_bidirectional_ring(self):
        digraph = retworkx.PyDiGraph()
        edge_list = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 0),
            (0, 3)
        ]
        digraph.extend_from_edge_list(edge_list)
        self.assertTrue(digraph.is_symmetric())
