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

import rustworkx as rx
import networkx as nx


class TestKarate(unittest.TestCase):
    def test_isomorphic_to_networkx(self):
        graph = rx.generators.karate_club_graph()
        nx_graph = rx.networkx_converter(nx.karate_club_graph(), keep_attributes=True)

        def node_matcher(a, b):
            if isinstance(a, dict):
                a, b, = (
                    b,
                    a,
                )
            return a == b["club"]

        def edge_matcher(a, b):
            if isinstance(a, dict):
                a, b, = (
                    b,
                    a,
                )
            return a == b["weight"]

        self.assertTrue(
            rx.is_isomorphic(graph, nx_graph, node_matcher=node_matcher, edge_matcher=edge_matcher)
        )
