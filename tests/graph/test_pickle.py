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

import pickle
import unittest

import rustworkx as rx


class TestPickleGraph(unittest.TestCase):
    def test_noweight_graph(self):
        g = rx.PyGraph()
        for i in range(4):
            g.add_node(None)
        g.add_edges_from_no_data([(0, 1), (1, 2), (3, 0), (3, 1)])
        g.remove_node(0)

        gprime = pickle.loads(pickle.dumps(g))
        self.assertEqual([1, 2, 3], gprime.node_indices())
        self.assertEqual([None, None, None], gprime.nodes())
        self.assertEqual({1: (1, 2, None), 3: (3, 1, None)}, dict(gprime.edge_index_map()))

    def test_weight_graph(self):
        g = rx.PyGraph(node_count_hint=4, edge_count_hint=4)
        g.add_nodes_from(["A", "B", "C", "D"])
        g.add_edges_from([(0, 1, "A -> B"), (1, 2, "B -> C"), (3, 0, "D -> A"), (3, 1, "D -> B")])
        g.remove_node(0)

        gprime = pickle.loads(pickle.dumps(g))
        self.assertEqual([1, 2, 3], gprime.node_indices())
        self.assertEqual(["B", "C", "D"], gprime.nodes())
        self.assertEqual({1: (1, 2, "B -> C"), 3: (3, 1, "D -> B")}, dict(gprime.edge_index_map()))

    def test_contracted_nodes_pickle(self):
        """Test pickle/unpickle of graphs with contracted nodes (issue #1503)"""
        g = rx.PyGraph()
        g.add_node("A")  # Node 0
        g.add_node("B")  # Node 1
        g.add_node("C")  # Node 2

        # Contract nodes 0 and 1 into a new node
        contracted_idx = g.contract_nodes([0, 1], "AB")
        g.add_edge(2, contracted_idx, "C -> AB")

        # Verify initial state
        self.assertEqual([2, contracted_idx], g.node_indices())
        self.assertEqual([(2, contracted_idx)], g.edge_list())

        # Test pickle/unpickle
        gprime = pickle.loads(pickle.dumps(g))

        # Verify the unpickled graph matches
        self.assertEqual(g.node_indices(), gprime.node_indices())
        self.assertEqual(g.edge_list(), gprime.edge_list())
        self.assertEqual(g.nodes(), gprime.nodes())

    def test_contracted_nodes_with_weights_pickle(self):
        """Test pickle/unpickle of graphs with contracted nodes and edge weights"""
        g = rx.PyGraph()
        g.add_nodes_from(["Node0", "Node1", "Node2", "Node3"])
        g.add_edges_from([(0, 2, "edge_0_2"), (1, 3, "edge_1_3")])

        # Contract multiple nodes
        contracted_idx = g.contract_nodes([0, 1], "Contracted_0_1")
        g.add_edge(contracted_idx, 2, "contracted_to_2")
        g.add_edge(3, contracted_idx, "3_to_contracted")

        # Test pickle/unpickle
        gprime = pickle.loads(pickle.dumps(g))

        # Verify complete graph state is preserved
        self.assertEqual(g.node_indices(), gprime.node_indices())
        self.assertEqual(g.edge_list(), gprime.edge_list())
        self.assertEqual(g.nodes(), gprime.nodes())
        self.assertEqual(dict(g.edge_index_map()), dict(gprime.edge_index_map()))
