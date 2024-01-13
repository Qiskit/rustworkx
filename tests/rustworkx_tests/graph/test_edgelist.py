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

import rustworkx


class TestEdgeList(unittest.TestCase):
    def test_empty_edge_list_graph(self):
        with tempfile.NamedTemporaryFile() as fd:
            graph = rustworkx.PyGraph.read_edge_list(fd.name)
        self.assertEqual(graph.nodes(), [])

    def test_invalid_path_graph(self):
        path = os.path.join(tempfile.gettempdir(), "fake_file_name.txt")
        with self.assertRaises(FileNotFoundError):
            rustworkx.PyGraph.read_edge_list(path)

    def test_simple_example_graph(self):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write("0 1\n")
            fd.write("1 2\n")
            fd.flush()
            graph = rustworkx.PyGraph.read_edge_list(fd.name)
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
            graph = rustworkx.PyGraph.read_edge_list(fd.name)
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
            graph = rustworkx.PyGraph.read_edge_list(fd.name, comment="#")
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
            graph = rustworkx.PyGraph.read_edge_list(fd.name, comment="#")
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
            graph = rustworkx.PyGraph.read_edge_list(fd.name, comment="#")
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
            graph = rustworkx.PyGraph.read_edge_list(fd.name, comment="#", deliminator=",")
        self.assertEqual(graph.node_indexes(), [0, 1, 2])
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(1, 0))
        self.assertTrue(graph.has_edge(2, 1))
        self.assertFalse(graph.has_edge(0, 2))
        self.assertEqual(graph.edges(), ["0", "1"])

    def test_labels_graph(self):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write("a|b|0// test a comment\n")
            fd.write("b|c|1\n")
            fd.write("//c|d\n")
            fd.flush()
            graph = rustworkx.PyGraph.read_edge_list(
                fd.name, comment="//", deliminator="|", labels=True
            )
        self.assertEqual(graph.node_indexes(), [0, 1, 2])
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertFalse(graph.has_edge(0, 2))
        self.assertEqual(graph.edges(), ["0", "1"])

    def test_labels_graph_target_existing(self):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write("a|b|0// test a comment\n")
            fd.write("b|c|1\n")
            fd.write("a|c\n")
            fd.flush()
            graph = rustworkx.PyGraph.read_edge_list(
                fd.name, comment="//", deliminator="|", labels=True
            )
        self.assertEqual(graph.node_indexes(), [0, 1, 2])
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(0, 2))
        self.assertEqual(graph.edges(), ["0", "1", None])

    def test_write_edge_list_empty_digraph(self):
        path = os.path.join(tempfile.gettempdir(), "empty.txt")
        graph = rustworkx.PyGraph()
        graph.write_edge_list(path)
        self.addCleanup(os.remove, path)
        with open(path) as edge_file:
            self.assertEqual("", edge_file.read())

    def test_write_edge_list_round_trip(self):
        path = os.path.join(tempfile.gettempdir(), "round_trip.txt")
        graph = rustworkx.generators.star_graph(5)
        count = iter(range(5))

        def weight_fn(edge):
            return str(next(count))

        graph.write_edge_list(path, weight_fn=weight_fn)
        self.addCleanup(os.remove, path)
        new_graph = rustworkx.PyGraph.read_edge_list(path)
        expected = [
            (0, 1, "0"),
            (0, 2, "1"),
            (0, 3, "2"),
            (0, 4, "3"),
        ]
        self.assertEqual(expected, new_graph.weighted_edge_list())

    def test_custom_delim(self):
        path = os.path.join(tempfile.gettempdir(), "custom_delim.txt")
        graph = rustworkx.generators.path_graph(5)
        graph.write_edge_list(path, deliminator=",")
        self.addCleanup(os.remove, path)
        expected = """0,1
1,2
2,3
3,4
"""
        with open(path) as edge_file:
            self.assertEqual(edge_file.read(), expected)

    def test_invalid_return_type_weight_fn(self):
        path = os.path.join(tempfile.gettempdir(), "fail.txt")
        graph = rustworkx.undirected_gnm_random_graph(5, 4)
        self.addCleanup(cleanup_file, path)
        with self.assertRaises(TypeError):
            graph.write_edge_list(path, weight_fn=lambda _: 4.5)

    def test_weight_fn_raises(self):
        path = os.path.join(tempfile.gettempdir(), "fail.txt")
        graph = rustworkx.undirected_gnm_random_graph(5, 4)

        def weight_fn(edge):
            raise KeyError

        self.addCleanup(cleanup_file, path)
        with self.assertRaises(KeyError):
            graph.write_edge_list(path, weight_fn=weight_fn)


def cleanup_file(path):
    try:
        os.remove(path)
    except Exception:
        pass
