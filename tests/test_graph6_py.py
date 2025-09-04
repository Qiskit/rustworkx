import tempfile
import rustworkx as rx
import unittest
import os


def test_graph6_roundtrip(tmp_path):
    # build a small graph with node/edge attrs
    g = rx.PyGraph()
    g.add_node({"label": "n0"})
    g.add_node({"label": "n1"})
    g.add_edge(0, 1, {"weight": 3})

    p = tmp_path / "g.g6"
    rx.graph_write_graph6_file(g, str(p))

    g2 = rx.read_graph6_file(str(p))
    assert isinstance(g2, rx.PyGraph)

    # check nodes and edges count
    assert g2.num_nodes() == 2
    assert g2.num_edges() == 1

    # check that node attrs 'label' were preserved in node data
    # Graph6 has no native attrs; our implementation stores None for attrs currently,
    # so assert node attrs exist or are None
    n0 = g2[0]
    assert n0 is None or ("label" in n0 and n0["label"] == "n0")

    # check edge exists
    assert list(g2.edge_list())


class TestGraph6(unittest.TestCase):
    def test_read_graph6_str_undirected(self):
        """Test reading an undirected graph from a graph6 string."""
        g6_str = "A_"
        graph = rx.read_graph6_str(g6_str)
        self.assertIsInstance(graph, rx.PyGraph)
        self.assertEqual(graph.num_nodes(), 2)
        self.assertEqual(graph.num_edges(), 1)
        self.assertTrue(graph.has_edge(0, 1))

    def test_read_graph6_str_directed(self):
        """Test reading a directed graph from a graph6 string."""
        g6_str = "&AG"
        graph = rx.read_graph6_str(g6_str)
        self.assertIsInstance(graph, rx.PyDiGraph)
        self.assertEqual(graph.num_nodes(), 2)
        self.assertEqual(graph.num_edges(), 1)
        self.assertTrue(graph.has_edge(1, 0))

    def test_write_graph6_from_pygraph(self):
        """Test writing a PyGraph to a graph6 string."""
        graph = rx.PyGraph()
        graph.add_nodes_from(range(2))
        graph.add_edge(0, 1, None)
        g6_str = rx.write_graph6_from_pygraph(graph)
        self.assertEqual(g6_str, "A_")

    def test_write_graph6_from_pydigraph(self):
        """Test writing a PyDiGraph to a graph6 string."""
        graph = rx.PyDiGraph()
        graph.add_nodes_from(range(2))
        graph.add_edge(1, 0, None)
        g6_str = rx.write_graph6_from_pydigraph(graph)
        self.assertEqual(g6_str, "&AG")

    def test_roundtrip_undirected(self):
        """Test roundtrip for an undirected graph."""
        graph = rx.generators.path_graph(4)
        g6_str = rx.write_graph6_from_pygraph(graph)
        new_graph = rx.read_graph6_str(g6_str)
        self.assertEqual(graph.num_nodes(), new_graph.num_nodes())
        self.assertEqual(graph.num_edges(), new_graph.num_edges())
        self.assertEqual(graph.edge_list(), new_graph.edge_list())

    def test_roundtrip_directed(self):
        """Test roundtrip for a directed graph."""
        graph = rx.generators.directed_path_graph(4)
        g6_str = rx.write_graph6_from_pydigraph(graph)
        new_graph = rx.read_graph6_str(g6_str)
        self.assertEqual(graph.num_nodes(), new_graph.num_nodes())
        self.assertEqual(graph.num_edges(), new_graph.num_edges())
        self.assertEqual(graph.edge_list(), new_graph.edge_list())

    def test_read_graph6_file(self):
        """Test reading a graph from a graph6 file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fd:
            fd.write("C~\\n")
            path = fd.name
        try:
            graph = rx.read_graph6_file(path)
            self.assertIsInstance(graph, rx.PyGraph)
            self.assertEqual(graph.num_nodes(), 4)
            self.assertEqual(graph.num_edges(), 6)  # K4
        finally:
            os.remove(path)

    def test_graph_write_graph6_file(self):
        """Test writing a PyGraph to a graph6 file."""
        graph = rx.generators.complete_graph(4)
        with tempfile.NamedTemporaryFile(delete=False) as fd:
            path = fd.name
        try:
            rx.graph_write_graph6_file(graph, path)
            with open(path, "r") as f:
                content = f.read()
            self.assertEqual(content, "C~")
        finally:
            os.remove(path)

    def test_digraph_write_graph6_file(self):
        """Test writing a PyDiGraph to a graph6 file."""
        graph = rx.PyDiGraph()
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1, None), (1, 2, None), (2, 0, None)])
        with tempfile.NamedTemporaryFile(delete=False) as fd:
            path = fd.name
        try:
            rx.digraph_write_graph6_file(graph, path)
            new_graph = rx.read_graph6_file(path)
            self.assertTrue(
                rx.is_isomorphic(graph, new_graph)
            )
        finally:
            os.remove(path)

    def test_invalid_graph6_string(self):
        """Test that an invalid graph6 string raises an error."""
        with self.assertRaises(Exception):
            rx.read_graph6_str("invalid_string")

    def test_empty_graph(self):
        """Test writing and reading an empty graph."""
        graph = rx.PyGraph()
        g6_str = rx.write_graph6_from_pygraph(graph)
        new_graph = rx.read_graph6_str(g6_str)
        self.assertEqual(new_graph.num_nodes(), 0)
        self.assertEqual(new_graph.num_edges(), 0)

    def test_graph_with_no_edges(self):
        """Test a graph with nodes but no edges."""
        graph = rx.PyGraph()
        graph.add_nodes_from(range(5))
        g6_str = rx.write_graph6_from_pygraph(graph)
        new_graph = rx.read_graph6_str(g6_str)
        self.assertEqual(new_graph.num_nodes(), 5)
        self.assertEqual(new_graph.num_edges(), 0)
