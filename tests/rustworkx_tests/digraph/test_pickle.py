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


class TestPickleDiGraph(unittest.TestCase):
    def test_noweight_graph(self):
        g = rx.PyDAG()
        for i in range(4):
            g.add_node(None)
        g.add_edges_from_no_data([(0, 1), (1, 2), (3, 0), (3, 1)])
        g.remove_node(0)

        gprime = pickle.loads(pickle.dumps(g))
        self.assertEqual([1, 2, 3], gprime.node_indices())
        self.assertEqual([None, None, None], gprime.nodes())
        self.assertEqual({1: (1, 2, None), 3: (3, 1, None)}, dict(gprime.edge_index_map()))

    def test_weight_graph(self):
        g = rx.PyDAG()
        g.add_nodes_from(["A", "B", "C", "D"])
        g.add_edges_from([(0, 1, "A -> B"), (1, 2, "B -> C"), (3, 0, "D -> A"), (3, 1, "D -> B")])
        g.remove_node(0)

        gprime = pickle.loads(pickle.dumps(g))
        self.assertEqual([1, 2, 3], gprime.node_indices())
        self.assertEqual(["B", "C", "D"], gprime.nodes())
        self.assertEqual({1: (1, 2, "B -> C"), 3: (3, 1, "D -> B")}, dict(gprime.edge_index_map()))
