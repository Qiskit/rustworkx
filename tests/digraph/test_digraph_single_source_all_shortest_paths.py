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

    def test_single_source_all_shortest_paths_zero_weight(self):
        # Create a graph with zero-weighted edges
        graph = rustworkx.PyDiGraph()
        nodes = graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edge(nodes[0], nodes[1], 0.0)  # 0 -> 1 with weight 0
        graph.add_edge(nodes[0], nodes[2], 1.0)  # 0 -> 2 with weight 1
        graph.add_edge(nodes[1], nodes[3], 1.0)  # 1 -> 3 with weight 1
        graph.add_edge(nodes[2], nodes[3], 0.0)  # 2 -> 3 with weight 0

        source = nodes[0]

        # Compute shortest path lengths using Dijkstra's algorithm
        shortest_lengths = rustworkx.digraph_dijkstra_shortest_path_lengths(
            graph, source, lambda e: e
        )

        # Compute all shortest paths
        all_shortest_paths = rustworkx.digraph_single_source_all_shortest_paths(graph, source)

        # For each target node, compute all simple paths and filter by shortest path length
        for target in nodes:
            target_idx = target  # Target is already an integer index
            if target_idx == source:
                continue  # Skip source node

            # Get all simple paths from source to target
            all_paths = rustworkx.all_simple_paths(graph, source, target_idx)

            # Compute the total weight for each path
            def path_weight(path):
                weight = 0.0
                for i in range(len(path) - 1):
                    edge = graph.get_edge_data(path[i], path[i + 1])
                    weight += edge
                return weight

            # Filter paths with total weight equal to shortest path length
            expected_paths = [
                path for path in all_paths if path_weight(path) == shortest_lengths[target_idx]
            ]

            # Get the computed shortest paths, default to empty list if not found
            computed_paths = all_shortest_paths.get(target_idx, [])

            # Sort both lists for comparison
            expected_paths.sort()
            computed_paths.sort()

            # Assert equality
            self.assertEqual(computed_paths, expected_paths)
