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


class TestFilter(unittest.TestCase):
    def test_filter_nodes(self):
        def my_filter_function1(node):
            return node == "cat"

        def my_filter_function2(node):
            return node == "lizard"

        def my_filter_function3(node):
            return node == "human"

        graph = rx.PyGraph()
        graph.add_node("cat")
        graph.add_node("cat")
        graph.add_node("dog")
        graph.add_node("lizard")
        graph.add_node("cat")
        cat_indices = graph.filter_nodes(my_filter_function1)
        lizard_indices = graph.filter_nodes(my_filter_function2)
        human_indices = graph.filter_nodes(my_filter_function3)
        self.assertEqual(list(cat_indices), [0, 1, 4])
        self.assertEqual(list(lizard_indices), [3])
        self.assertEqual(list(human_indices), [])

    def test_filter_edges(self):
        def my_filter_function1(edge):
            return edge == "friends"

        def my_filter_function2(edge):
            return edge == "enemies"

        def my_filter_function3(node):
            return node == "frenemies"

        graph = rx.PyGraph()
        graph.add_node("cat")
        graph.add_node("cat")
        graph.add_node("dog")
        graph.add_node("lizard")
        graph.add_node("cat")
        graph.add_edge(0, 2, "friends")
        graph.add_edge(0, 1, "friends")
        graph.add_edge(0, 3, "enemies")
        friends_indices = graph.filter_edges(my_filter_function1)
        enemies_indices = graph.filter_edges(my_filter_function2)
        frenemies_indices = graph.filter_edges(my_filter_function3)
        self.assertEqual(list(friends_indices), [0, 1])
        self.assertEqual(list(enemies_indices), [2])
        self.assertEqual(list(frenemies_indices), [])

    def test_filter_errors(self):
        def my_filter_function1(node):
            raise TypeError("error!")

        graph = rx.PyGraph()
        graph.add_node("cat")
        graph.add_node("cat")
        graph.add_node("dog")
        graph.add_edge(0, 1, "friends")
        graph.add_edge(1, 2, "enemies")
        with self.assertRaises(TypeError):
            graph.filter_nodes(my_filter_function1)
        with self.assertRaises(TypeError):
            graph.filter_edges(my_filter_function1)
