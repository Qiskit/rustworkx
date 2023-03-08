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


class TestConfigurationModel(unittest.TestCase):
    def test_undirected_configuration_model(self):
        graph = rustworkx.undirected_configuration_model([0, 1, 1, 1, 1, 0], seed=42)
        self.assertEqual(len(graph), 6)
        self.assertEqual(len(graph.edges()), 2)

    def test_undirected_configuration_model_empty(self):
        graph = rustworkx.undirected_configuration_model([], seed=42)
        self.assertEqual(len(graph), 0)
        self.assertEqual(len(graph.edges()), 0)

    def test_undirected_configuration_model_weights(self):
        graph = rustworkx.undirected_configuration_model(
            [1, 2, 3, 4], weights=list(range(4)), seed=42
        )
        self.assertEqual(len(graph), 4)
        self.assertEqual([x for x in range(4)], graph.nodes())
        self.assertEqual(len(graph.edges()), 5)

    def test_undirected_configuration_model_length_mismatch(self):
        with self.assertRaises(ValueError):
            rustworkx.undirected_configuration_model([1, 2, 3, 4], weights=list(range(3)))

    def test_undirected_configuration_model_odd_sum(self):
        with self.assertRaises(ValueError):
            rustworkx.undirected_configuration_model([1, 2, 3, 5])
