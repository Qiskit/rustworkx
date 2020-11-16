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


class TestCustomReturnTypeComparisons(unittest.TestCase):

    def setUp(self):
        self.dag = retworkx.PyDAG()
        node_a = self.dag.add_node('a')
        self.dag.add_child(node_a, 'b', "Edgy")

    def test__eq__match(self):
        self.assertTrue(self.dag.node_indexes() == [0, 1])

    def test__eq__not_match(self):
        self.assertFalse(self.dag.node_indexes() == [1, 2])

    def test__eq__different_length(self):
        self.assertFalse(self.dag.node_indexes() == [0, 1, 2, 3])

    def test__eq__invalid_type(self):
        with self.assertRaises(TypeError):
            self.dag.node_indexes() == ['a', None]

    def test__ne__match(self):
        self.assertFalse(self.dag.node_indexes() != [0, 1])

    def test__ne__not_match(self):
        self.assertTrue(self.dag.node_indexes() != [1, 2])

    def test__ne__different_length(self):
        self.assertTrue(self.dag.node_indexes() != [0, 1, 2, 3])

    def test__ne__invalid_type(self):
        with self.assertRaises(TypeError):
            self.dag.node_indexes() != ['a', None]

    def test__gt__not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.dag.node_indexes() > [2, 1]
