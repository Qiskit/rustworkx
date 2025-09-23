import tempfile
import rustworkx as rx
import unittest
import os
import gzip  # added for gzip file write tests


class TestGraph6(unittest.TestCase):
    def _build_two_node_graph(self):
        g = rx.PyGraph()
        g.add_nodes_from(range(2))
        g.add_edge(0, 1, None)
        return g
    def test_graph6_roundtrip(self):
        # build a small graph with node/edge attrs
        g = rx.PyGraph()
        g.add_node({"label": "n0"})
        g.add_node({"label": "n1"})
        g.add_edge(0, 1, {"weight": 3})

        # Use NamedTemporaryFile with context-managed cleanup
        with tempfile.NamedTemporaryFile() as fd:
            rx.graph_write_graph6(g, fd.name)
            g2 = rx.read_graph6(fd.name)
            self.assertIsInstance(g2, rx.PyGraph)
            self.assertEqual(g2.num_nodes(), 2)
            self.assertEqual(g2.num_edges(), 1)
            n0 = g2[0]
            self.assertTrue(n0 is None or ("label" in n0 and n0["label"] == "n0"))
            self.assertTrue(list(g2.edge_list()))

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

    def test_read_graph6(self):
        """Test reading a graph from a graph6 file."""
        with tempfile.NamedTemporaryFile(mode="w+") as fd:
            fd.write("C~\n")
            fd.flush()
            graph = rx.read_graph6(fd.name)
            self.assertIsInstance(graph, rx.PyGraph)
            self.assertEqual(graph.num_nodes(), 4)
            self.assertEqual(graph.num_edges(), 6)  # K4

    def test_graph_write_graph6(self):
        """Test writing a PyGraph to a graph6 file."""
        graph = rx.generators.complete_graph(4)
        with tempfile.NamedTemporaryFile() as fd:
            rx.graph_write_graph6(graph, fd.name)
            with open(fd.name, "r") as f:
                content = f.read()
            self.assertEqual(content, "C~")

    def test_digraph_write_graph6(self):
        """Test writing a PyDiGraph to a graph6 file."""
        graph = rx.PyDiGraph()
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1, None), (1, 2, None), (2, 0, None)])
        with tempfile.NamedTemporaryFile() as fd:
            rx.digraph_write_graph6(graph, fd.name)
            new_graph = rx.read_graph6(fd.name)
            self.assertTrue(rx.is_isomorphic(graph, new_graph))

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

    def test_write_plain_file(self):
        g = self._build_two_node_graph()
        expected = "A_"  # known graph6 for 2-node single edge
        with tempfile.NamedTemporaryFile(suffix=".g6") as fd:
            rx.graph_write_graph6(g, fd.name)
            with open(fd.name, "rt", encoding="ascii") as fh:
                content = fh.read().strip()
            self.assertEqual(expected, content)

    def test_write_gzip_file(self):
        g = self._build_two_node_graph()
        expected = "A_"
        with tempfile.NamedTemporaryFile(suffix=".g6.gz") as fd:
            rx.graph_write_graph6(g, fd.name)
            with gzip.open(fd.name, "rt", encoding="ascii") as fh:
                content = fh.read().strip()
            self.assertEqual(expected, content)


class TestGraph6FormatExtras(unittest.TestCase):
    def test_roundtrip_small_undirected(self):
        g = rx.PyGraph()
        g.add_nodes_from([None, None])
        g.add_edge(0, 1, None)
        s = rx.write_graph6_from_pygraph(g)
        new_g = rx.read_graph6_str(s)
        self.assertIsInstance(new_g, rx.PyGraph)
        self.assertEqual(new_g.num_nodes(), 2)
        self.assertEqual(new_g.num_edges(), 1)

    def test_write_and_read_triangle(self):
        g = rx.PyGraph()
        g.add_nodes_from([None, None, None])
        g.add_edges_from([(0, 1, None), (1, 2, None), (0, 2, None)])
        s = rx.write_graph6_from_pygraph(g)
        new_g = rx.read_graph6_str(s)
        self.assertIsInstance(new_g, rx.PyGraph)
        self.assertEqual(new_g.num_nodes(), 3)
        self.assertEqual(new_g.num_edges(), 3)

    def test_file_roundtrip_format(self):
        import tempfile, pathlib
        g = rx.PyGraph()
        g.add_nodes_from([None, None, None, None])
        g.add_edges_from([(0, 1, None), (2, 3, None)])
        s = rx.write_graph6_from_pygraph(g)
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / 'u.g6'
            rx.graph_write_graph6(g, str(p))
            g2 = rx.read_graph6(str(p))
        self.assertIsInstance(g2, rx.PyGraph)
        self.assertEqual(g2.num_nodes(), 4)
        self.assertEqual(g2.num_edges(), 2)
        self.assertEqual(rx.write_graph6_from_pygraph(g2), s)

    def test_invalid_string_format(self):
        with self.assertRaises(Exception):
            rx.read_graph6_str('invalid_string')


# ---- Size parse tests (merged from test_graph6_size_parse.py) ----

def _encode_medium(n: int) -> str:
    assert 63 <= n < (1 << 18)
    parts = [0, 0, 0]
    val = n
    for i in range(2, -1, -1):
        parts[i] = val & 0x3F
        val >>= 6
    return "~" + "".join(chr(p + 63) for p in parts)


def _encode_long(n: int) -> str:
    assert 0 <= n < (1 << 36)
    parts = [0] * 6
    val = n
    for i in range(5, -1, -1):
        parts[i] = val & 0x3F
        val >>= 6
    return "~~" + "".join(chr(p + 63) for p in parts)


class TestGraph6SizeParse(unittest.TestCase):
    def test_parse_short_boundary(self):
        n, consumed = rx.parse_graph6_size("}")
        self.assertEqual((n, consumed), (62, 1))

    def test_parse_medium_start(self):
        hdr = _encode_medium(63)
        n, consumed = rx.parse_graph6_size(hdr)
        self.assertEqual((n, consumed), (63, 4))

    def test_parse_long_start(self):
        n_val = 1 << 18
        hdr = _encode_long(n_val)
        n, consumed = rx.parse_graph6_size(hdr)
        self.assertEqual((n, consumed), (n_val, 8))

    def test_parse_directed_variants(self):
        n, consumed = rx.parse_graph6_size("&}", offset=1)
        self.assertEqual((n, consumed), (62, 1))
        hdr = "&" + _encode_medium(63)
        n2, consumed2 = rx.parse_graph6_size(hdr, offset=1)
        self.assertEqual((n2, consumed2), (63, 4))

    def test_non_canonical_medium_for_short(self):
        n = 62
        val = n
        parts = [0, 0, 0]
        for i in range(2, -1, -1):
            parts[i] = val & 0x3F
            val >>= 6
        bad_hdr = "~" + "".join(chr(p + 63) for p in parts)
        with self.assertRaises(rx.Graph6ParseError):
            rx.parse_graph6_size(bad_hdr)

    def test_overflow(self):
        overflow_val = 1 << 36
        parts = [0] * 6
        val = overflow_val
        for i in range(5, -1, -1):
            parts[i] = val & 0x3F
            val >>= 6
        hdr = "~~" + "".join(chr(p + 63) for p in parts)
        with self.assertRaises((rx.Graph6OverflowError, rx.Graph6ParseError)):
            rx.parse_graph6_size(hdr)
