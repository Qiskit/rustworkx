import unittest
import rustworkx as rx
import rustworkx.sparse6 as rx_sparse6
import rustworkx

class TestSparse6Basic(unittest.TestCase):
    def test_header_only_raises(self):
        with self.assertRaises(rustworkx.Graph6Error):
            rx_sparse6.read_sparse6_str('>>sparse6<<:')
    def test_header_with_size_and_no_edges(self):
        g = rx_sparse6.read_sparse6_str('>>sparse6<<:@')
        self.assertEqual((g.num_nodes(), g.num_edges()), (1,0))
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
        g = rx.PyGraph()
        for _ in range(4): g.add_node(None)
        g.add_edge(0,1,None); g.add_edge(2,3,None)
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2 = rx_sparse6.read_sparse6_str(s)
        self.assertEqual((g2.num_nodes(), g2.num_edges()), (g.num_nodes(), g.num_edges()))

class TestSparse6Advanced(unittest.TestCase):
    def _make_graph(self, n, edges):
        g = rx.PyGraph()
        while g.num_nodes() < n: g.add_node(None)
        for u,v in edges: g.add_edge(u,v,None)
        return g
    def test_padding_special_case_examples(self):
        g = self._make_graph(2, [(0,1)])
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2 = rx_sparse6.read_sparse6_str(s)
        self.assertEqual(g.num_edges(), g2.num_edges())
        edges = [(0,1),(3,2)]
        g = self._make_graph(4, edges)
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2 = rx_sparse6.read_sparse6_str(s)
        def canon(ep): u,v=ep; return (u,v) if u < v else (v,u)
        edges_a = sorted(canon(g.get_edge_endpoints_by_index(e)) for e in g.edge_indices())
        edges_b = sorted(canon(g2.get_edge_endpoints_by_index(e)) for e in g2.edge_indices())
        self.assertEqual(edges_a, edges_b)
    def test_incremental_prefix_supported(self):
        g = self._make_graph(3, [(0,1),(1,2)])
        s = rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        s2 = (';' + s[1:]) if s.startswith(':') else (';' + s)
        g2 = rx_sparse6.read_sparse6_str(s2)
        self.assertEqual(g.num_edges(), g2.num_edges())
    def test_large_N_extended_forms(self):
        n4=1000
        edges=[(0,1),(10,20),(500,400)]
        g=self._make_graph(n4, edges)
        s=rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2=rx_sparse6.read_sparse6_str(s)
        self.assertEqual(g.num_nodes(), g2.num_nodes())
        n8=1<<18
        parts=[126,126]; val=n8; six_parts=[]
        for i in range(6): six_parts.append((val>>(6*(5-i))) & 0x3F)
        parts.extend(p+63 for p in six_parts)
        s=":"+"".join(chr(p) for p in parts)+"\n"
        g2=rx_sparse6.read_sparse6_str(s)
        self.assertEqual(g2.num_nodes(), n8)
    def test_roundtrip_random_small(self):
        g=self._make_graph(5,[(0,1),(0,2),(3,4)])
        s=rx_sparse6.write_sparse6_from_pygraph(g, header=False)
        g2=rx_sparse6.read_sparse6_str(s)
        self.assertEqual(g.num_edges(), g2.num_edges())

if __name__=='__main__':
    unittest.main()
