import unittest
import rustworkx as rx
import rustworkx.graph6 as rx_graph6


class TestGraph6Undirected(unittest.TestCase):
	def test_roundtrip_small_undirected(self):
		g = rx.PyGraph()
		g.add_nodes_from([None, None])
		g.add_edge(0, 1, None)
		s = rx_graph6.write_graph6_from_pygraph(g)
		# ensure roundtrip parses to an undirected PyGraph
		new_g = rx_graph6.read_graph6_str(s)
		self.assertIsInstance(new_g, rx.PyGraph)
		self.assertEqual(new_g.num_nodes(), 2)
		self.assertEqual(new_g.num_edges(), 1)
		self.assertEqual(new_g.num_nodes(), 2)
		self.assertEqual(new_g.num_edges(), 1)

	def test_write_and_read_triangle(self):
		g = rx.PyGraph()
		g.add_nodes_from([None, None, None])
		g.add_edges_from([(0, 1, None), (1, 2, None), (0, 2, None)])
		s = rx_graph6.write_graph6_from_pygraph(g)
		new_g = rx_graph6.read_graph6_str(s)
		self.assertIsInstance(new_g, rx.PyGraph)
		self.assertEqual(new_g.num_nodes(), 3)
		self.assertEqual(new_g.num_edges(), 3)

	def test_file_roundtrip(self):
		import tempfile, pathlib
		g = rx.PyGraph()
		g.add_nodes_from([None, None, None, None])
		g.add_edges_from([(0, 1, None), (2, 3, None)])
		s = rx_graph6.write_graph6_from_pygraph(g)
		with tempfile.TemporaryDirectory() as td:
			p = pathlib.Path(td) / 'u.g6'
			rx.graph_write_graph6_file(g, str(p))
			g2 = rx.read_graph6_file(str(p))
		self.assertIsInstance(g2, rx.PyGraph)
		self.assertEqual(g2.num_nodes(), 4)
		self.assertEqual(g2.num_edges(), 2)
		self.assertEqual(rx_graph6.write_graph6_from_pygraph(g2), s)

	def test_invalid_string(self):
		with self.assertRaises(Exception):
			rx_graph6.read_graph6_str('invalid_string')


if __name__ == '__main__':  # pragma: no cover
	unittest.main()
import unittest
print('FILE WRITE TEST: graph6_format loaded')
