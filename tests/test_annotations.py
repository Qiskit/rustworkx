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

import rustworkx as rx
import types
import unittest


class TestAnnotationSubscriptions(unittest.TestCase):
    def test_digraph(self):
        graph: rx.PyDiGraph[int, int] = rx.PyDiGraph()
        self.assertIsInstance(
            graph.__class_getitem__((int, int)),
            types.GenericAlias,
        )

    def test_graph(self):
        graph: rx.PyGraph[int, int] = rx.PyGraph()
        self.assertIsInstance(
            graph.__class_getitem__((int, int)),
            types.GenericAlias,
        )
    
    def test_dag(self):
        graph: rx.PyDAG[int, int] = rx.PyDAG()
        self.assertIsInstance(
            graph.__class_getitem__((int, int)),
            types.GenericAlias,
        )