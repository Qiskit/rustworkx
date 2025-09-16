import unittest
import rustworkx
import rustworkx.sparse6 as rx_sparse6


class TestSparse6(unittest.TestCase):
    def test_header_only_raises(self):
        with self.assertRaises(rustworkx.Graph6Error):
            rx_sparse6.read_sparse6_str('>>sparse6<<:')

    def test_header_with_size_and_no_edges(self):
        # n = 1 encoded as '@' (value 1) after header colon
        g = rx_sparse6.read_sparse6_str('>>sparse6<<:@')
        self.assertEqual(g.num_nodes(), 1)
        self.assertEqual(g.num_edges(), 0)

    def test_empty_string_raises(self):
        with self.assertRaises(rustworkx.Graph6Error):
            rx_sparse6.read_sparse6_str('')

    def test_header_with_whitespace_raises(self):
        with self.assertRaises(rustworkx.Graph6Error):
            rx_sparse6.read_sparse6_str('>>sparse6<<:   ')

    def test_control_chars_in_payload(self):
        with self.assertRaises(rustworkx.Graph6Error):
            rx_sparse6.read_sparse6_str('>>sparse6<<:\x00\x01\x02')

    def test_roundtrip_small_graph(self):
        g = rustworkx.PyGraph()
        for _ in range(4):
            g.add_node(None)
        g.add_edge(0,1,None)
        g.add_edge(2,3,None)
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2 = rx_sparse6.read_sparse6_str(s)
        self.assertEqual(g2.num_nodes(), g.num_nodes())
        self.assertEqual(g2.num_edges(), g.num_edges())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
