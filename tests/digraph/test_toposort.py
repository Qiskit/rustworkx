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
