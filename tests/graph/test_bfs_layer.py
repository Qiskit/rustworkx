import unittest
import rustworkx


class TestGraphBfsLayers(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.generators.path_graph(5)

    def test_simple_path(self):
        layers = rustworkx.bfs_layers(self.graph, [0])
        self.assertEqual(layers, [[0], [1], [2], [3], [4]])

    def test_multiple_sources(self):
        layers = rustworkx.bfs_layers(self.graph, [0, 4])
        self.assertEqual(layers, [[0, 4], [1, 3], [2]])

    def test_disconnected_graph(self):
        g = rustworkx.PyGraph()
        g.extend_from_edge_list([(0, 1), (2, 3)])
        layers = rustworkx.bfs_layers(g, [0])
        self.assertEqual(layers, [[0], [1]])

    def test_no_sources_default_all_nodes(self):
        layers = rustworkx.bfs_layers(self.graph, None)
        self.assertTrue(all(isinstance(layer, list) for layer in layers))

    def test_invalid_source(self):
        with self.assertRaises(IndexError):
            rustworkx.bfs_layers(self.graph, [99])
