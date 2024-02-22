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


class TestTopologicalSorter(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyDiGraph()
        self.graph.extend_from_edge_list(
            [
                (0, 2),
                (1, 2),
                (2, 3),
                (2, 4),
                (3, 5),
            ]
        )

    def test_topo_sort(self):
        sorter = rustworkx.TopologicalSorter(self.graph)
        nodes = sorter.get_ready()
        self.assertEqual(nodes, [0, 1])
        sorter.done(nodes)
        nodes = sorter.get_ready()
        self.assertEqual(nodes, [2])
        sorter.done(nodes)
        nodes = sorter.get_ready()
        self.assertEqual(nodes, [4, 3])
        sorter.done(nodes)
        nodes = sorter.get_ready()
        self.assertEqual(nodes, [5])
        sorter.done(nodes)
        nodes = sorter.get_ready()
        self.assertEqual(nodes, [])

    def test_topo_sort_do_not_emit_if_node_has_undone_preds(self):
        sorter = rustworkx.TopologicalSorter(self.graph)
        nodes = sorter.get_ready()
        self.assertEqual(nodes, [0, 1])
        sorter.done([0])
        nodes = sorter.get_ready()
        self.assertEqual(nodes, [])

    def test_topo_sort_raises_if_node_not_ready(self):
        sorter = rustworkx.TopologicalSorter(self.graph)
        with self.assertRaises(ValueError):
            sorter.done([0])

    def test_topo_sort_raises_if_node_already_done(self):
        sorter = rustworkx.TopologicalSorter(self.graph)
        sorter.get_ready()
        sorter.done([0])
        with self.assertRaises(ValueError):
            sorter.done([0])

    def test_topo_sort_raises_if_graph_has_cycle(self):
        graph = rustworkx.generators.directed_cycle_graph(5)
        with self.assertRaises(rustworkx.DAGHasCycle):
            _ = rustworkx.TopologicalSorter(graph)

    def test_topo_sort_progress_if_graph_has_cycle_and_cycle_check_disabled(
        self,
    ):
        graph = rustworkx.generators.directed_cycle_graph(5)
        starting_node = graph.add_node("starting node")
        graph.add_edge(starting_node, 0, "starting edge")

        sorter = rustworkx.TopologicalSorter(graph, check_cycle=False)
        nodes = sorter.get_ready()
        self.assertEqual(nodes, [starting_node])
        sorter.done(nodes)
        self.assertFalse(sorter.is_active())

    def test_reverse_order(self):
        sorter = rustworkx.TopologicalSorter(self.graph, reverse=True)
        self.assertEqual(set(sorter.get_ready()), {4, 5})
        sorter.done([5])
        self.assertEqual(set(sorter.get_ready()), {3})
        sorter.done([3, 4])
        self.assertEqual(set(sorter.get_ready()), {2})
        sorter.done([2])
        self.assertEqual(set(sorter.get_ready()), {0, 1})
        sorter.done([0, 1])
        self.assertFalse(sorter.is_active())

    def test_initial(self):
        dag = rustworkx.PyDiGraph()
        dag.add_nodes_from(range(9))
        dag.add_edges_from_no_data(
            [
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 4),
                (3, 4),
                (4, 5),
                (5, 6),
                (4, 7),
                (6, 8),
                (7, 8),
            ]
        )

        # Last three nodes, nothing reachable except nodes that will be returned.
        sorter = rustworkx.TopologicalSorter(dag, initial=[6, 7])
        self.assertEqual(set(sorter.get_ready()), {6, 7})
        sorter.done([6, 7])
        self.assertEqual(set(sorter.get_ready()), {8})
        sorter.done([8])
        self.assertFalse(sorter.is_active())

        # Setting `initial` to the set of root nodes should return the same as not setting it.
        initial_sorter = rustworkx.TopologicalSorter(dag, initial=[0])
        base_sorter = rustworkx.TopologicalSorter(dag)
        bases = []
        initials = []
        while base_ready := base_sorter.get_ready():
            bases.append(base_ready)
            initials.append(initial_sorter.get_ready())
            base_sorter.done(bases[-1])
            initial_sorter.done(initials[-1])
        self.assertEqual(bases, initials)
        self.assertFalse(initial_sorter.is_active())

        # Node 8 is reachable from 7, but isn't dominated by it, so shouldn't be returned.
        sorter = rustworkx.TopologicalSorter(dag, initial=[7])
        self.assertEqual(set(sorter.get_ready()), {7})
        sorter.done([7])
        self.assertFalse(sorter.is_active())

    def test_initial_reverse(self):
        dag = rustworkx.PyDiGraph()
        dag.add_nodes_from(range(9))
        dag.add_edges_from_no_data(
            [
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 4),
                (3, 4),
                (4, 5),
                (5, 6),
                (4, 7),
                (6, 8),
                (7, 8),
            ]
        )

        # Last three nodes, nothing reachable except nodes that will be returned.
        sorter = rustworkx.TopologicalSorter(dag, reverse=True, initial=[1, 2])
        self.assertEqual(set(sorter.get_ready()), {1, 2})
        sorter.done([1, 2])
        self.assertEqual(set(sorter.get_ready()), {0})
        sorter.done([0])
        self.assertFalse(sorter.is_active())

        # Setting `initial` to the set of root nodes should return the same as not setting it.
        initial_sorter = rustworkx.TopologicalSorter(dag, reverse=True, initial=[8])
        base_sorter = rustworkx.TopologicalSorter(dag, reverse=True)
        bases = []
        initials = []
        while base_ready := base_sorter.get_ready():
            bases.append(base_ready)
            initials.append(initial_sorter.get_ready())
            base_sorter.done(bases[-1])
            initial_sorter.done(initials[-1])
        self.assertEqual(bases, initials)
        self.assertFalse(initial_sorter.is_active())

        # Node 0 is reachable from 1, but isn't dominated by it, so shouldn't be returned.
        sorter = rustworkx.TopologicalSorter(dag, reverse=True, initial=[1])
        self.assertEqual(set(sorter.get_ready()), {1})
        sorter.done([1])
        self.assertFalse(sorter.is_active())

    def test_initial_natural_zero(self):
        dag = rustworkx.PyDiGraph()
        dag.add_nodes_from(range(5))
        # There's no edges in this graph, so a natural topological ordering allows everything in the
        # first pass.  If `initial` is given, though, the loose zero-degree nodes are not dominated
        # by the givens, so should not be returned.
        forwards = rustworkx.TopologicalSorter(dag, initial=[0, 3])
        self.assertEqual(set(forwards.get_ready()), {0, 3})
        forwards.done([0, 3])
        self.assertFalse(forwards.is_active())

        backwards = rustworkx.TopologicalSorter(dag, reverse=True, initial=[0, 3])
        self.assertEqual(set(backwards.get_ready()), {0, 3})
        backwards.done([0, 3])
        self.assertFalse(backwards.is_active())

    def test_initial_invalid(self):
        dag = rustworkx.generators.directed_path_graph(5)
        forwards = rustworkx.TopologicalSorter(dag, initial=[0, 1])
        with self.assertRaisesRegex(ValueError, "initial node is reachable from another"):
            while ready := forwards.get_ready():
                forwards.done(ready)
        backwards = rustworkx.TopologicalSorter(dag, reverse=True, initial=[3, 4])
        with self.assertRaisesRegex(ValueError, "initial node is reachable from another"):
            while ready := backwards.get_ready():
                backwards.done(ready)
