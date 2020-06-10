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


class TestDAGAllSimplePaths(unittest.TestCase):
    def test_all_simple_paths(self):
        dag = retworkx.PyDAG()
        for i in range(6):
            dag.add_node(i)
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 2),
            (3, 4),
            (4, 2),
            (4, 5),
            (5, 2),
            (5, 3)]
        for edge in edges:
            dag.add_edge(edge[0], edge[1], None)
        paths = retworkx.dag_all_simple_paths(dag, 0, 5)
        expected = [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 4, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 4, 5],
            [0, 2, 3, 4, 5],
            [0, 2, 4, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 4, 5]]
        for i in expected:
            self.assertIn(i, paths)

    def test_all_simple_paths_with_cutoff(self):
        dag = retworkx.PyDAG()
        for i in range(6):
            dag.add_node(i)
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 2),
            (3, 4),
            (4, 2),
            (4, 5),
            (5, 2),
            (5, 3)]
        for edge in edges:
            dag.add_edge(edge[0], edge[1], None)
        paths = retworkx.dag_all_simple_paths(dag, 0, 5, cutoff=4)
        expected = [
            [0, 2, 4, 5],
            [0, 3, 4, 5]]
        self.assertEqual(len(expected), len(paths))
        for i in expected:
            self.assertIn(i, paths)

    def test_all_simple_path_no_path(self):
        dag = retworkx.PyDAG()
        dag.add_node(0)
        dag.add_node(1)
        self.assertEqual([], retworkx.dag_all_simple_paths(dag, 0, 5))

    def test_graph_dag_all_simple_paths(self):
        dag = retworkx.PyGraph()
        dag.add_node(0)
        dag.add_node(1)
        self.assertRaises(TypeError, retworkx.dag_all_simple_paths,
                          (dag, 0, 1))


class TestGraphAllSimplePaths(unittest.TestCase):
    def test_all_simple_paths(self):
        graph = retworkx.PyGraph()
        for i in range(6):
            graph.add_node(i)
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 2),
            (3, 4),
            (4, 2),
            (4, 5),
            (5, 2),
            (5, 3)]
        for edge in edges:
            graph.add_edge(edge[0], edge[1], None)
        paths = retworkx.graph_all_simple_paths(graph, 0, 5)
        expected = [
            [0, 3, 4, 5],
            [0, 3, 4, 2, 5],
            [0, 3, 4, 2, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 2, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 2, 5],
            [0, 3, 2, 4, 5],
            [0, 3, 1, 2, 4, 5],
            [0, 3, 1, 2, 5],
            [0, 3, 1, 2, 4, 5],
            [0, 2, 4, 5],
            [0, 2, 4, 3, 5],
            [0, 2, 3, 4, 5],
            [0, 2, 3, 5],
            [0, 2, 5],
            [0, 2, 4, 5],
            [0, 2, 4, 3, 5],
            [0, 2, 3, 4, 5],
            [0, 2, 3, 5],
            [0, 2, 1, 3, 4, 5],
            [0, 2, 1, 3, 5],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 2, 5],
            [0, 1, 3, 4, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 3, 2, 5],
            [0, 1, 3, 2, 4, 5],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 5],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 5]]
        self.assertEqual(len(expected), len(paths))
        for i in expected:
            self.assertIn(i, paths)

    def test_all_simple_paths_with_cutoff(self):
        graph = retworkx.PyGraph()
        for i in range(6):
            graph.add_node(i)
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 2),
            (3, 4),
            (4, 2),
            (4, 5),
            (5, 2),
            (5, 3)]
        for edge in edges:
            graph.add_edge(edge[0], edge[1], None)
        paths = retworkx.graph_all_simple_paths(graph, 0, 5, cutoff=4)
        expected = [
            [0, 3, 4, 5],
            [0, 3, 2, 5],
            [0, 3, 5],
            [0, 3, 2, 5],
            [0, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 2, 5],
            [0, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 1, 3, 5],
            [0, 1, 2, 5]]
        self.assertEqual(len(expected), len(paths))
        for i in expected:
            self.assertIn(i, paths)

    def test_all_simple_path_no_path(self):
        dag = retworkx.PyDAG()
        dag.add_node(0)
        dag.add_node(1)
        self.assertEqual([], retworkx.dag_all_simple_paths(dag, 0, 5))

    def test_dag_graph_all_simple_paths(self):
        dag = retworkx.PyDAG()
        dag.add_node(0)
        dag.add_node(1)
        self.assertRaises(TypeError, retworkx.graph_all_simple_paths,
                          (dag, 0, 1))
