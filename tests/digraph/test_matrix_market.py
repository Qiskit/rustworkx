import tempfile
import unittest
import rustworkx


class TestMatrixMarketDiGraph(unittest.TestCase):
    """Test reading and writing Matrix Market data for directed graphs."""

    def test_write_to_string(self):
        """Write a PyDiGraph to a Matrix Market string."""
        g = rustworkx.PyDiGraph()
        g.add_nodes_from([None, None, None])
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)

        mm_str = rustworkx.write_matrix_market_digraph(g, None)
        self.assertIsInstance(mm_str, str)
        self.assertIn("matrix", mm_str)
        # Directed edges are not duplicated
        self.assertIn("3 3 2", mm_str)

    def test_write_to_file(self):
        """Write PyDiGraph data to a Matrix Market file."""
        g = rustworkx.PyDiGraph()
        g.add_nodes_from([None, None, None])
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)

        with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp:
            rustworkx.write_matrix_market_digraph(g, tmp.name)
            tmp.seek(0)
            content = tmp.read().decode("utf-8")

        self.assertIn("matrix", content)
        self.assertIn("3 3 2", content)

    def test_read_from_file(self):
        """Read a Matrix Market file into a PyDiGraph."""
        content = """%%MatrixMarket matrix coordinate real general
3 3 2
1 2 1.0
2 3 1.0
"""

        with tempfile.NamedTemporaryFile(suffix=".mtx", mode="w+", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            g = rustworkx.read_matrix_market_file(tmp.name)

        self.assertIsInstance(g, rustworkx.PyDiGraph)
        self.assertEqual(len(g.nodes()), 3)
        self.assertEqual(len(g.edges()), 2)

    def test_read_from_string(self):
        """Read Matrix Market data directly from a string."""
        mm_str = """%%MatrixMarket matrix coordinate real general
3 3 2
1 2 1.0
2 3 1.0
"""

        g = rustworkx.read_matrix_market(mm_str)
        self.assertIsInstance(g, rustworkx.PyDiGraph)
        self.assertEqual(len(g.nodes()), 3)
        self.assertEqual(len(g.edges()), 2)

    def test_roundtrip_in_memory(self):
        """Roundtrip: write → read should reconstruct same directed graph."""
        g = rustworkx.PyDiGraph()
        g.add_nodes_from([None, None, None])
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)

        mm_str = rustworkx.write_matrix_market_digraph(g, None)
        g2 = rustworkx.read_matrix_market(mm_str)

        self.assertEqual(len(g2.nodes()), len(g.nodes()))
        # Directed edges are not duplicated
        self.assertEqual(len(g2.edges()), 2)

    def test_roundtrip_via_file(self):
        """Roundtrip through file should preserve directed structure."""
        g = rustworkx.PyDiGraph()
        g.add_nodes_from([None, None])
        g.add_edge(0, 1, 1.0)

        with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp:
            rustworkx.write_matrix_market_digraph(g, tmp.name)
            g2 = rustworkx.read_matrix_market_file(tmp.name)

        self.assertEqual(len(g2.nodes()), 2)
        self.assertEqual(len(g2.edges()), 1)
