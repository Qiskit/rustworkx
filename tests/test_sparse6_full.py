import unittest
import rustworkx as rx
import rustworkx.sparse6 as rx_sparse6


class TestSparse6Full(unittest.TestCase):
    def _make_graph(self, n, edges):
        g = rx.PyGraph()
        # ensure n nodes
        while g.num_nodes() < n:
            g.add_node(None)
        for u, v in edges:
            g.add_edge(u, v, None)
        return g

    def test_padding_special_case_examples(self):
        # Small k=1, n=2: single edge between 0 and 1
        g = self._make_graph(2, [(0, 1)])
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2 = rx_sparse6.read_sparse6_str(s)
        self.assertEqual(g.num_edges(), g2.num_edges())

        # k=2, n=4: build edges to trigger padding path
        edges = [(0, 1), (3, 2)]
        g = self._make_graph(4, edges)
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2 = rx_sparse6.read_sparse6_str(s)
        def canon(ep):
            u,v = ep
            return (u,v) if u < v else (v,u)
        edges_a = sorted([canon(g.get_edge_endpoints_by_index(e)) for e in g.edge_indices()])
        edges_b = sorted([canon(g2.get_edge_endpoints_by_index(e)) for e in g2.edge_indices()])
        self.assertEqual(edges_a, edges_b)

    def test_incremental_prefix_supported(self):
        # incremental sparse6 begins with ';' followed by same body as ':'
        g = self._make_graph(3, [(0, 1), (1, 2)])
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        # switch leading ':' to ';' to simulate incremental form
        if s.startswith(":"):
            s2 = ";" + s[1:]
        else:
            s2 = ";" + s
        g2 = rx_sparse6.read_sparse6_str(s2)
        self.assertEqual(g.num_edges(), g2.num_edges())

    def test_large_N_extended_forms(self):
        # 4-byte: n >= 63 and < 2^18
        n4 = 1000
        edges = [(0, 1), (10, 20), (500, 400)]
        g = self._make_graph(n4, edges)
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2 = rx_sparse6.read_sparse6_str(s)
        self.assertEqual(g.num_nodes(), g2.num_nodes())

        # 8-byte: test parsing of an 8-byte N(n) with no edges (avoid huge allocation)
        n8 = 1 << 18
        # encode n8 into 126,126 + 6 chars of 6-bit values
        parts = [126, 126]
        val = n8
        six_parts = []
        for i in range(6):
            six_parts.append((val >> (6 * (5 - i))) & 0x3F)
        parts.extend([p + 63 for p in six_parts])
        s = ":" + "".join(chr(p) for p in parts) + "\n"
        g2 = rx_sparse6.read_sparse6_str(s)
        self.assertEqual(g2.num_nodes(), n8)

    def test_roundtrip_random_small(self):
        g = self._make_graph(5, [(0, 1), (0, 2), (3, 4)])
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2 = rx_sparse6.read_sparse6_str(s)
        self.assertEqual(g.num_edges(), g2.num_edges())


if __name__ == '__main__':
    unittest.main()
