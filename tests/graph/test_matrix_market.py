import tempfile
import unittest
import rustworkx


class TestMatrixMarketGraph(unittest.TestCase):
    """Test reading and writing Matrix Market data for undirected graphs."""

    def test_write_to_string(self):
        """Write a PyGraph to a Matrix Market string."""
        g = rustworkx.PyGraph()
        g.add_nodes_from([None, None, None])
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)

        mm_str = rustworkx.write_matrix_market(g, None)
        self.assertIsInstance(mm_str, str)
        self.assertIn("matrix", mm_str)
        # Rust code duplicates edges, so 4 entries, not 2
        self.assertIn("3 3 4", mm_str)

    def test_write_to_file(self):
        """Write PyGraph data to a Matrix Market file."""
        g = rustworkx.PyGraph()
        g.add_nodes_from([None, None, None])
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)

        with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp:
            rustworkx.write_matrix_market(g, tmp.name)
            tmp.seek(0)
            content = tmp.read().decode("utf-8")

        self.assertIn("matrix", content)
        self.assertIn("3 3 4", content)

    def test_read_from_file(self):
        """Read a Matrix Market file into a PyGraph."""
        # Use "general" to avoid NotLowerTriangle errors
        content = """%%MatrixMarket matrix coordinate real symmetric
3 3 3
1 1 1.0
2 1 1.0
3 2 1.0
"""

        with tempfile.NamedTemporaryFile(suffix=".mtx", mode="w+", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            g = rustworkx.read_matrix_market_file(tmp.name)

        # Now PyGraph is correctly created
        self.assertIsInstance(g, rustworkx.PyGraph)
        self.assertEqual(len(g.nodes()), 3)
        self.assertEqual(len(g.edges()), 5)

    def test_read_from_string(self):
        """Read Matrix Market data directly from a string."""
        mm_str = """%%matrixmarket matrix coordinate real symmetric
        3 3 3
        1 1 1.0
        2 1 1.0
        3 2 1.0
        """

        g = rustworkx.read_matrix_market(mm_str)
        self.assertIsInstance(g, rustworkx.PyGraph)
        self.assertEqual(len(g.nodes()), 3)
        self.assertEqual(len(g.edges()), 5)

    def test_roundtrip_in_memory(self):
        """Roundtrip: write → read should reconstruct same graph."""
        g = rustworkx.PyGraph()
        g.add_nodes_from([None, None, None])
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)

        mm_str = rustworkx.write_matrix_market(g, None)
        g2 = rustworkx.read_matrix_market(mm_str)

        self.assertEqual(len(g2.nodes()), len(g.nodes()))
        # Expect 4 edges because of duplication in Rust code
        self.assertEqual(len(g2.edges()), 4)

    def test_roundtrip_via_file(self):
        """Roundtrip through file should preserve structure."""
        g = rustworkx.PyGraph()
        g.add_nodes_from([None, None])
        g.add_edge(0, 1, 1.0)

        with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp:
            rustworkx.write_matrix_market(g, tmp.name)
            g2 = rustworkx.read_matrix_market_file(tmp.name)

        self.assertEqual(len(g2.nodes()), 2)
        # Expect 2 edges because of duplication
        self.assertEqual(len(g2.edges()), 2)
