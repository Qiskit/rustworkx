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


class TestSubstituteNodesCheckCycle(unittest.TestCase):
    def setUp(self) -> None:
        """┌─┐
         ┌─┤a├─┐
         │ └─┘ │
        ┌▼┐    │
        │b│    │
        └┬┘    │
         │    ┌▼┐
         └────┤c│
              └─┘
        """
        super().setUp()
        self.dag = retworkx.PyDAG()
        self.node_a = self.dag.add_node("a")
        self.node_b = self.dag.add_child(self.node_a, "b", "b")
        self.node_c = self.dag.add_child(self.node_b, "c", "c")

        self.new_weight = "m"

    def do_substitution(self, **kwargs):
        """┌─┐                         ┌─┐
         ┌─┤a├─┐             ┌─────────┤m│
         │ └─┘ │             │         └▲┘
        ┌▼┐    │            ┌▼┐         │
        │b│    │     ───►   │b├─────────┘
        └┬┘    │            └─┘
         │    ┌▼┐
         └────┤c│
              └─┘
        """
        self.dag.substitute_nodes_with_node(
            [self.node_a, self.node_c], self.new_weight, **kwargs
        )

    def test_cycle_check_enable_local(self):
        # Disable at class level.
        self.dag.check_cycle = False

        # Check removal is not allowed with explicit check_cycle=True.
        self.assertRaises(
            retworkx.DAGWouldCycle, self.do_substitution, check_cycle=True
        )

    def test_cycle_check_disable_local(self):
        # Enable at class level.
        self.dag.check_cycle = True

        # Check removal is allowed for check_cycle=False
        self.do_substitution(check_cycle=False)
        self.assertSetEqual(
            set(self.dag.nodes()), {"b", "m"}
        )

    def test_cycle_check_inherit_class_enable(self):
        # Enable at class level.
        self.dag.check_cycle = True

        # Check removal is not allowed.
        self.assertRaises(retworkx.DAGWouldCycle, self.do_substitution)

    def test_cycle_check_inherit_class_disable(self):
        # Disable at class level.
        self.dag.check_cycle = False

        # Check removal is allowed.
        self.do_substitution()
        self.assertSetEqual(
            set(self.dag.nodes()), {"b", "m"}
        )

class TestSubstituteNodes(unittest.TestCase):
    def test_empty_nodes(self):
        """Replacing empty nodes is functionally equivalent to add_node."""
        self.dag = retworkx.PyDAG()
        self.dag.substitute_nodes_with_node([], "m")

        self.assertSetEqual(set(self.dag.nodes()), {"m"})

    def test_unknown_nodes(self):
        """
        Replacing all unknown nodes is functionally equivalent to add_node,
        since unknown nodes should be ignored.
        """
        self.dag = retworkx.PyDAG()
        self.dag.substitute_nodes_with_node([0, 1, 2], "m")

        self.assertSetEqual(set(self.dag.nodes()), {"m"})