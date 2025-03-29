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

import operator as op
import unittest

import rustworkx


class TestContractNodesCheckCycleSwitch(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dag = rustworkx.PyDAG()
        self.node_a = self.dag.add_node("a")
        self.node_b = self.dag.add_child(self.node_a, "b", "b")
        self.node_c = self.dag.add_child(self.node_b, "c", "c")

        self.new_weight = "m"

    def contract(self, **kwargs):
        """
           ┌─┐                         ┌─┐
         ┌─┤a│               ┌─────────┤m│
         │ └─┘               │         └▲┘
        ┌▼┐                 ┌▼┐         │
        │b│          ───►   │b├─────────┘
        └┬┘                 └─┘
         │  ┌─┐
         └─►┤c│
            └─┘
        """
        self.dag.contract_nodes([self.node_a, self.node_c], self.new_weight, **kwargs)

    def test_cycle_check_enable_local(self):
        # Disable at class level.
        self.dag.check_cycle = False

        # Check removal is not allowed with explicit check_cycle=True.
        self.assertRaises(rustworkx.DAGWouldCycle, self.contract, check_cycle=True)

    def test_cycle_check_disable_local(self):
        # Enable at class level.
        self.dag.check_cycle = True

        # Check removal is allowed for check_cycle=False
        self.contract(check_cycle=False)
        self.assertEqual(set(self.dag.nodes()), {"b", "m"})

    def test_cycle_check_inherit_class_enable(self):
        # Enable at class level.
        self.dag.check_cycle = True

        # Check removal is not allowed.
        self.assertRaises(rustworkx.DAGWouldCycle, self.contract)

    def test_cycle_check_inherit_class_disable(self):
        # Disable at class level.
        self.dag.check_cycle = False

        # Check removal is allowed.
        self.contract()
        self.assertEqual(set(self.dag.nodes()), {"b", "m"})


class TestContractNodes(unittest.TestCase):
    def test_empty_nodes(self):
        """Replacing empty nodes is functionally equivalent to add_node."""
        dag = rustworkx.PyDAG()
        dag.contract_nodes([], "m")

        self.assertEqual(set(dag.nodes()), {"m"})

    def test_unknown_nodes(self):
        """
        Replacing all unknown nodes is functionally equivalent to add_node,
        since unknown nodes should be ignored.
        """
        dag = rustworkx.PyDAG()
        dag.contract_nodes([0, 1, 2], "m")

        self.assertEqual(set(dag.nodes()), {"m"})

    def test_cycle_path_len_gt_1(self):
        """
            ┌─┐              ┌─┐
         ┌4─┤a├─1┐           │m├──1───┐
         │  └─┘  │           └▲┘      │
        ┌▼┐     ┌▼┐           │      ┌▼┐
        │d│     │b│   ───►    │      │b│
        └▲┘     └┬┘           │      └┬┘
         │  ┌─┐  2            │  ┌─┐  2
         └3─┤c│◄─┘            └3─┤c│◄─┘
            └─┘                  └─┘
        """
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", 1)
        node_c = dag.add_child(node_b, "c", 2)
        node_d = dag.add_child(node_c, "c", 3)
        dag.add_edge(node_a, node_d, 4)

        with self.assertRaises(rustworkx.DAGWouldCycle):
            dag.contract_nodes([node_a, node_d], "m", check_cycle=True)

        # Proceed, ignoring cycles.
        node_m = dag.contract_nodes([node_a, node_d], "m")

        self.assertEqual([node_b, node_c, node_m], dag.node_indexes())
        self.assertEqual(
            {(node_b, node_c), (node_c, node_m), (node_m, node_b)},
            set(dag.edge_list()),
        )

    def test_multiple_paths_would_cycle(self):
        """
            ┌─┐     ┌─┐                  ┌─┐     ┌─┐
         ┌3─┤c│     │e├─5┐            ┌──┤c│     │e├──┐
         │  └▲┘     └▲┘  │            │  └▲┘     └▲┘  │
        ┌▼┐  2  ┌─┐  4  ┌▼┐           │   2  ┌─┐  4   │
        │d│  └──┤b├──┘  │f│   ───►    │   └──┤b├──┘   │
        └─┘     └▲┘     └─┘           3      └▲┘      5
                 1                    │       1       │
                ┌┴┐                   │      ┌┴┐      │
                │a│                   └─────►│m│◄─────┘
                └─┘                          └─┘
        """
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_b = dag.add_child(node_a, "b", 1)
        node_c = dag.add_child(node_b, "c", 2)
        node_d = dag.add_child(node_c, "d", 3)
        node_e = dag.add_child(node_b, "e", 4)
        node_f = dag.add_child(node_e, "f", 5)

        with self.assertRaises(rustworkx.DAGWouldCycle):
            dag.contract_nodes([node_a, node_d, node_f], "m", check_cycle=True)

        # Proceed, ignoring cycles.
        node_m = dag.contract_nodes([node_a, node_d, node_f], "m")

        self.assertEqual([node_b, node_c, node_e, node_m], dag.node_indexes())
        self.assertEqual(
            {
                (node_b, node_c),
                (node_c, node_m),
                (node_e, node_m),
                (node_b, node_e),
                (node_m, node_b),
            },
            set(dag.edge_list()),
        )

    def test_replace_node_no_neighbors(self):
        dag = rustworkx.PyDAG()
        node_a = dag.add_node("a")
        node_m = dag.contract_nodes([node_a], "m", check_cycle=True)
        self.assertEqual([node_m], dag.node_indexes())
        self.assertEqual(set(), set(dag.edge_list()))

    def test_keep_edges_multigraph(self):
        """
           ┌─┐            ┌─┐
         ┌─┤a│◄┐        ┌─┤a│◄┐
         │ └─┘ │        │ └─┘ │
         1     2   ──►  1     2
        ┌▼┐   ┌┴┐       │ ┌─┐ │
        │b│   │c│       └►│m├─┘
        └─┘   └─┘         └─┘
        """
        dag = rustworkx.PyDiGraph()
        node_a = dag.add_node("a")
        node_b = dag.add_node("b")
        node_c = dag.add_node("c")

        dag.add_edge(node_a, node_b, 1)
        dag.add_edge(node_c, node_a, 2)

        with self.assertRaises(rustworkx.DAGWouldCycle):
            dag.contract_nodes([node_b, node_c], "m", check_cycle=True)

        # Proceed, ignoring cycles.
        node_m = dag.contract_nodes([node_b, node_c], "m")
        self.assertEqual([node_a, node_m], dag.node_indexes())
        self.assertEqual(
            {(node_a, node_m, 1), (node_m, node_a, 2)},
            set(dag.weighted_edge_list()),
        )


class TestContractNodesSimpleGraph(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dag = rustworkx.PyDAG(multigraph=False)
        self.node_a = self.dag.add_node("a")
        self.node_b = self.dag.add_child(self.node_a, "b", 1)
        self.node_c = self.dag.add_child(self.node_a, "c", 2)
        self.node_d = self.dag.add_child(self.node_a, "d", 3)
        self.node_e = self.dag.add_node("e")
        self.dag.add_edge(self.node_b, self.node_e, 4)
        self.dag.add_edge(self.node_c, self.node_e, 5)
        self.dag.add_edge(self.node_d, self.node_e, 6)

    def test_collapse_parallel_edges_no_combo_fn(self):
        """
        Parallel edges are collapsed arbitrarily when weight_combo_fn is None.
            ┌─┐               ┌─┐
            │a│               │a│
         ┌──┴┬┴──┐            └┬┘
         1   2   3        1 or 2 or 3
        ┌▼┐ ┌▼┐ ┌▼┐           ┌▼┐
        │b│ │c│ │d│   ──►     │m│
        └┬┘ └┬┘ └┬┘           └┬┘
         4   5   6        4 or 5 or 6
         └──►▼◄──┘            ┌▼┐
            │e│               │e│
            └─┘               └─┘
        """
        self.dag.contract_nodes([self.node_b, self.node_c, self.node_d], "m")

        self.assertEqual(set(self.dag.nodes()), {"a", "e", "m"})
        self.assertEqual(len(self.dag.edges()), 2)

        # Should have one incoming edge, one outgoing
        self.assertTrue(any(e in self.dag.edges() for e in {1, 2, 3}))
        self.assertTrue(any(e in self.dag.edges() for e in {4, 5, 6}))

    def test_collapse_parallel_edges(self):
        """
        Parallel edges are collapsed using weight_combo_fn.
            ┌─┐               ┌─┐
            │a│               │a│
         ┌──┴┬┴──┐            └┬┘
         1   2   3             6
        ┌▼┐ ┌▼┐ ┌▼┐           ┌▼┐
        │b│ │c│ │d│   ──►     │m│
        └┬┘ └┬┘ └┬┘           └┬┘
         4   5   6             15
         └──►▼◄──┘            ┌▼┐
            │e│               │e│
            └─┘               └─┘
        """
        self.dag.contract_nodes(
            [self.node_b, self.node_c, self.node_d],
            "m",
            weight_combo_fn=op.add,
        )

        self.assertEqual(set(self.dag.nodes()), {"a", "e", "m"})
        self.assertEqual(len(self.dag.edges()), 2)

        # Should have one incoming edge, one outgoing
        self.assertEqual(set(self.dag.edges()), {6, 15})

    def test_replace_all_nodes(self):
        self.dag.contract_nodes(self.dag.node_indexes(), "m")
        self.assertEqual(set(self.dag.nodes()), {"m"})
        self.assertFalse(self.dag.edges())
