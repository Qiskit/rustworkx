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
        for edge in self.grid.edge_list():
            self.grid.update_edge(edge[0], edge[1], 1.0)
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

    def test_single_source_all_shortest_paths_zero_weight(self):
        # Create a graph with zero-weighted edges
        graph = rustworkx.PyGraph()
        nodes = graph.add_nodes_from([0, 1, 2, 3])  # a=0, b=1, c=2, d=3
        graph.add_edge(nodes[0], nodes[1], 0.0)  # ab with weight 0.0
        graph.add_edge(nodes[1], nodes[2], 0.0)  # bc with weight 0.0
        graph.add_edge(nodes[2], nodes[0], 0.0)  # ca with weight 0.0
        graph.add_edge(nodes[2], nodes[3], 1.0)  # cd with weight 1.0

        source = nodes[0]

        # Compute shortest path lengths from source node using Dijkstra's algorithm
        shortest_lengths = rustworkx.dijkstra_shortest_path_lengths(graph, source, lambda x: x)

        # Compute the total weight of a path
        def path_weight(path):
            total = 0.0
            for i in range(len(path) - 1):
                edge_data = graph.get_edge_data(path[i], path[i + 1])
                total += edge_data
            return total

        # Compute expected shortest paths
        expected = {source: [[source]]}  # Trivial path from source to itself
        for target in graph.nodes():
            if target != source:
                # Get all simple paths from source to target
                paths = rustworkx.all_simple_paths(graph, source, target)
                # Filter paths to keep only those with total weight equal to the shortest path length
                expected_paths = [
                    path for path in paths if path_weight(path) == shortest_lengths[target]
                ]
                expected[target] = expected_paths

        # Compute all shortest paths using the function to test
        paths = rustworkx.graph_single_source_all_shortest_paths(graph, source)

        # Verify all paths match the expected output
        for node in expected:
            self.assertEqual(sorted(paths[node]), sorted(expected[node]))
