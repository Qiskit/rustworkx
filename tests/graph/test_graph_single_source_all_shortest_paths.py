import unittest
import rustworkx


class TestSingleSourceAllShortestPaths(unittest.TestCase):
    def setUp(self):
        # Set up a cycle graph with 4 nodes
        self.cycle = rustworkx.PyGraph()
        self.cycle_nodes = self.cycle.add_nodes_from([0, 1, 2, 3])
        self.cycle.add_edges_from(
            [
                (self.cycle_nodes[0], self.cycle_nodes[1], 1),
                (self.cycle_nodes[1], self.cycle_nodes[2], 1),
                (self.cycle_nodes[2], self.cycle_nodes[3], 1),
                (self.cycle_nodes[3], self.cycle_nodes[0], 1),
            ]
        )
        # Set up a 4x4 grid graph
        self.grid = rustworkx.generators.grid_graph(4, 4)
        # Set up a disconnected graph (cycle + isolated node)
        self.disconnected = rustworkx.PyGraph()
        self.disconnected_nodes = self.disconnected.add_nodes_from([0, 1, 2, 3, 4])
        self.disconnected.add_edges_from(
            [
                (self.disconnected_nodes[0], self.disconnected_nodes[1], 1),
                (self.disconnected_nodes[1], self.disconnected_nodes[2], 1),
                (self.disconnected_nodes[2], self.disconnected_nodes[3], 1),
                (self.disconnected_nodes[3], self.disconnected_nodes[0], 1),
            ]
        )

    def test_single_source_all_shortest_paths_cycle(self):
        paths = rustworkx.graph_single_source_all_shortest_paths(self.cycle, self.cycle_nodes[0])
        expected = {0: [[0]], 1: [[0, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 3]]}
        self.assertEqual(sorted(paths[2]), sorted(expected[2]))

    def test_single_source_all_shortest_paths_grid(self):
        paths = rustworkx.graph_single_source_all_shortest_paths(self.grid, 1)
        expected = [
            [1, 2, 3, 7, 11],
            [1, 2, 6, 7, 11],
            [1, 2, 6, 10, 11],
            [1, 5, 6, 7, 11],
            [1, 5, 6, 10, 11],
            [1, 5, 9, 10, 11],
        ]
        self.assertEqual(sorted(paths[11]), sorted(expected))

    def test_single_source_all_shortest_paths_weighted_cycle(self):
        paths = rustworkx.graph_single_source_all_shortest_paths(
            self.cycle, self.cycle_nodes[0], weight_fn=lambda x: float(x)
        )
        expected = {0: [[0]], 1: [[0, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 3]]}
        self.assertEqual(sorted(paths[2]), sorted(expected[2]))

    def test_single_source_all_shortest_paths_weighted_grid(self):
        paths = rustworkx.graph_single_source_all_shortest_paths(
            self.grid, 1, weight_fn=lambda x: float(x) if x is not None else 1.0
        )
        expected = [
            [1, 2, 3, 7, 11],
            [1, 2, 6, 7, 11],
            [1, 2, 6, 10, 11],
            [1, 5, 6, 7, 11],
            [1, 5, 6, 10, 11],
            [1, 5, 9, 10, 11],
        ]
        self.assertEqual(sorted(paths[11]), sorted(expected))

    def test_single_source_all_shortest_paths_disconnected_from_cycle(self):
        paths = rustworkx.graph_single_source_all_shortest_paths(
            self.disconnected, self.disconnected_nodes[0]
        )
        expected = [[0, 1, 2], [0, 3, 2]]
        self.assertEqual(sorted(paths[2]), sorted(expected))

    def test_single_source_all_shortest_paths_disconnected_from_isolated(self):
        paths = rustworkx.graph_single_source_all_shortest_paths(
            self.disconnected, self.disconnected_nodes[4]
        )
        expected = [[4]]
        self.assertEqual(paths[4], expected)

    def test_single_source_all_shortest_paths_with_invalid_weights(self):
        for invalid_weight in [float("nan"), -1]:
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises(ValueError):
                    rustworkx.graph_single_source_all_shortest_paths(
                        self.cycle, self.cycle_nodes[0], weight_fn=lambda _: invalid_weight
                    )

    def test_single_source_all_shortest_paths_with_digraph_input(self):
        g = rustworkx.PyDiGraph()
        g.add_nodes_from([0, 1])
        with self.assertRaises(TypeError):
            rustworkx.graph_single_source_all_shortest_paths(g, 0)
