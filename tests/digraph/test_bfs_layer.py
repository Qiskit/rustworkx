import unittest
import rustworkx


class TestDiGraphBfsLayers(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.generators.path_graph(6).to_directed()

    def test_simple_chain(self):
        layers = rustworkx.bfs_layers(self.graph, [3])
        self.assertEqual([sorted(layer) for layer in layers], [[3], [2, 4], [1, 5], [0]])

    def test_multiple_sources(self):
        layers = rustworkx.bfs_layers(self.graph, [0, 3])
        self.assertEqual(sorted(layers[0]), [0, 3])

    def test_disconnected_digraph(self):
        g = rustworkx.PyDiGraph()
        g.extend_from_edge_list([(0, 1), (2, 3)])
        layers = rustworkx.bfs_layers(g, [2])
        self.assertEqual(layers, [[2], [3]])

    def test_no_sources_defaults(self):
        layers = rustworkx.bfs_layers(self.graph, None)
        self.assertTrue(any(0 in layer for layer in layers))

    def test_invalid_source(self):
        with self.assertRaises(IndexError):
            rustworkx.bfs_layers(self.graph, [42])
