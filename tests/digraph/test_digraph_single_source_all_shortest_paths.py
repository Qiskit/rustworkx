import unittest
import rustworkx


class TestDigraphSingleSourceAllShortestPaths(unittest.TestCase):
    def setUp(self):
        self.cycle = rustworkx.PyDiGraph()
        self.cycle_nodes = self.cycle.add_nodes_from([0, 1, 2, 3])
        self.cycle.add_edges_from(
            [
                (self.cycle_nodes[0], self.cycle_nodes[1], 1),
                (self.cycle_nodes[1], self.cycle_nodes[2], 1),
                (self.cycle_nodes[2], self.cycle_nodes[3], 1),
                (self.cycle_nodes[3], self.cycle_nodes[0], 1),
            ]
        )
        self.directed = rustworkx.PyDiGraph()
        self.directed_nodes = self.directed.add_nodes_from([0, 1, 2, 3])
        self.directed.add_edges_from(
            [
                (self.directed_nodes[0], self.directed_nodes[1], 1),
                (self.directed_nodes[0], self.directed_nodes[2], 1),
                (self.directed_nodes[1], self.directed_nodes[3], 1),
                (self.directed_nodes[2], self.directed_nodes[3], 1),
            ]
        )

    def test_single_source_all_shortest_paths_cycle(self):
        paths = rustworkx.digraph_single_source_all_shortest_paths(self.cycle, self.cycle_nodes[0])
        expected = {0: [[0]], 1: [[0, 1]], 2: [[0, 1, 2]], 3: [[0, 3]]}
        self.assertEqual(paths[2], expected[2])

    def test_single_source_all_shortest_paths_directed(self):
        paths = rustworkx.digraph_single_source_all_shortest_paths(
            self.directed, self.directed_nodes[0]
        )
        expected = {0: [[0]], 1: [[0, 1]], 2: [[0, 2]], 3: [[0, 1, 3], [0, 2, 3]]}
        self.assertEqual(sorted(paths[3]), sorted(expected[3]))

    def test_single_source_all_shortest_paths_as_undirected(self):
        paths = rustworkx.digraph_single_source_all_shortest_paths(
            self.directed, self.directed_nodes[0], as_undirected=True
        )
        expected = {0: [[0]], 1: [[0, 1]], 2: [[0, 2]], 3: [[0, 1, 3], [0, 2, 3]]}
        self.assertEqual(sorted(paths[3]), sorted(expected[3]))

    def test_single_source_all_shortest_paths_invalid_weights(self):
        for invalid_weight in [float("nan"), -1]:
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises(ValueError):
                    rustworkx.digraph_single_source_all_shortest_paths(
                        self.cycle, self.cycle_nodes[0], weight_fn=lambda _: invalid_weight
                    )
