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

class TestSubstituteNodes(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dag = retworkx.PyDAG()
        self.node_a = self.dag.add_node("a")
        self.node_b = self.dag.add_child(self.node_a, "b", "b")
        self.node_c = self.dag.add_child(self.node_b, "c", "c")

        self.node_m = self.dag.add_node("m")

    def test_cycle_check_enable_local(self):
        # Disable at class level.
        self.dag.check_cycle = False

        # Check removal is not allowed with explicit force_check_cycle=True.
        self.assertRaises(
            retworkx.DAGWouldCycle,
            self.dag.substitute_nodes_with_node,
            {self.node_a, self.node_c},
            self.node_m,
            force_check_cycle=True
        )

    def test_cycle_check_disable_local(self):
        # Enable at class level.
        self.dag.check_cycle = True

        # Check removal is still not allowed for force_check_cycle=False,
        # since disabling cycle checking just for this method is not allowed.
        self.assertRaises(
            retworkx.DAGWouldCycle,
            self.dag.substitute_nodes_with_node,
            {self.node_a, self.node_c},
            self.node_m,
            force_check_cycle=False
        )

    def test_cycle_check_inherit_class_enable(self):
        # Enable at class level.
        self.dag.check_cycle = True

        # Check removal is not allowed.
        self.assertRaises(
            retworkx.DAGWouldCycle,
            self.dag.substitute_nodes_with_node,
            {self.node_a, self.node_c},
            self.node_m
        )
    
    def test_cycle_check_inherit_class_disable(self):
        # Disable at class level.
        self.dag.check_cycle = False

        # Check removal is allowed.
        self.dag.substitute_nodes_with_node({self.node_a, self.node_c}, self.node_m)
        self.assertSetEqual(set(self.dag.node_indexes()), {self.node_b, self.node_m})
