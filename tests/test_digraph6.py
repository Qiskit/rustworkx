import unittest
import rustworkx as rx


class TestDigraph6Format(unittest.TestCase):
    def test_roundtrip_small_directed(self):
        g = rx.PyDiGraph()
        g.add_nodes_from([None, None])
        g.add_edge(0, 1, None)
        s = rx.write_graph6_from_pydigraph(g)
        new_g = rx.read_graph6_str(s)
        self.assertIsInstance(new_g, rx.PyDiGraph)
        self.assertEqual(new_g.num_nodes(), 2)
        self.assertEqual(new_g.num_edges(), 1)

    def test_asymmetric_two_edge(self):
        g = rx.PyDiGraph()
        g.add_nodes_from([None, None])
        g.add_edges_from([(0, 1, None), (1, 0, None)])
        s = rx.write_graph6_from_pydigraph(g)
        new_g = rx.read_graph6_str(s)
        self.assertIsInstance(new_g, rx.PyDiGraph)
        self.assertEqual(new_g.num_edges(), 2)

    def test_file_roundtrip_directed(self):
        import tempfile, pathlib
        g = rx.PyDiGraph()
        g.add_nodes_from([None, None, None])
        g.add_edges_from([(0, 1, None), (1, 2, None)])
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / 'd.d6'
            rx.digraph_write_graph6_file(g, str(p))
            g2 = rx.read_graph6_file(str(p))
        self.assertIsInstance(g2, rx.PyDiGraph)
        self.assertEqual(g2.num_nodes(), 3)
        self.assertEqual(g2.num_edges(), 2)

    def test_invalid_string(self):
        # Rust implementation may panic on malformed input; accept any
        # raised BaseException (including the pyo3 PanicException wrapper).
        with self.assertRaises(BaseException):
            rx.read_graph6_str('&invalid')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
