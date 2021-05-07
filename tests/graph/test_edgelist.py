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

import os
import tempfile
import unittest

import retworkx


class TestEdgeList(unittest.TestCase):
    def test_empty_edge_list_graph(self):
        with tempfile.NamedTemporaryFile() as fd:
            graph = retworkx.PyGraph.read_edge_list(fd.name)
        self.assertEqual(graph.nodes(), [])

    def test_invalid_path_graph(self):
        path = os.path.join(tempfile.gettempdir(), "fake_file_name.txt")
        with self.assertRaises(FileNotFoundError):
            retworkx.PyGraph.read_edge_list(path)

    def test_simple_example_graph(self):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write("0 1\n")
            fd.write("1 2\n")
            fd.flush()
            graph = retworkx.PyGraph.read_edge_list(fd.name)
        self.assertEqual(graph.node_indexes(), [0, 1, 2])
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(1, 0))
        self.assertTrue(graph.has_edge(2, 1))
        self.assertFalse(graph.has_edge(0, 2))

    def test_blank_line_graph(self):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write("0 1\n")
            fd.write("\n")
            fd.write("1 2\n")
            fd.flush()
            graph = retworkx.PyGraph.read_edge_list(fd.name)
        self.assertEqual(graph.node_indexes(), [0, 1, 2])
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(1, 0))
        self.assertTrue(graph.has_edge(2, 1))
        self.assertFalse(graph.has_edge(0, 2))

    def test_comment_graph(self):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write("0 1\n")
            fd.write("1 2 # test comments\n")
            fd.write("#2 3\n")
            fd.flush()
            graph = retworkx.PyGraph.read_edge_list(fd.name, comment="#")
        self.assertEqual(graph.node_indexes(), [0, 1, 2])
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(1, 0))
        self.assertTrue(graph.has_edge(2, 1))
        self.assertFalse(graph.has_edge(0, 2))

    def test_comment_leading_space_graph(self):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write("0 1\n")
            fd.write("1 2 # test comments\n")
            fd.write("  #2 3\n")
            fd.flush()
            graph = retworkx.PyGraph.read_edge_list(fd.name, comment="#")
        self.assertEqual(graph.node_indexes(), [0, 1, 2])
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(1, 0))
        self.assertTrue(graph.has_edge(2, 1))
        self.assertFalse(graph.has_edge(0, 2))

    def test_weight_graph(self):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write("0 1 0\n")
            fd.write("1 2 1# test comments\n")
            fd.write("#2 3\n")
            fd.flush()
            graph = retworkx.PyGraph.read_edge_list(fd.name, comment="#")
        self.assertEqual(graph.node_indexes(), [0, 1, 2])
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(1, 0))
        self.assertTrue(graph.has_edge(2, 1))
        self.assertFalse(graph.has_edge(0, 2))
        self.assertEqual(graph.edges(), ["0", "1"])

    def test_delim_graph(self):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write("0,1,0\n")
            fd.write("1,2,1# test comments\n")
            fd.write("#2,3\n")
            fd.flush()
            graph = retworkx.PyGraph.read_edge_list(fd.name, comment="#", deliminator=",")
        self.assertEqual(graph.node_indexes(), [0, 1, 2])
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(1, 0))
        self.assertTrue(graph.has_edge(2, 1))
        self.assertFalse(graph.has_edge(0, 2))
        self.assertEqual(graph.edges(), ["0", "1"])
