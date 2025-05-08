# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import unittest

import rustworkx


class TestDijkstraSearch(unittest.TestCase):
    def setUp(self):
        self.graph = rustworkx.PyGraph()
        self.graph.extend_from_weighted_edge_list(
            [
                (0, 1, 1),
                (0, 2, 2),
                (1, 3, 10),
                (2, 1, 1),
                (2, 5, 1),
                (2, 6, 1),
                (5, 3, 1),
                (4, 7, 1),
            ]
        )

    def test_graph_dijkstra_tree_edges(self):
        class DijkstraTreeEdgesRecorder(rustworkx.visit.DijkstraVisitor):
            def __init__(self):
                self.edges = []
                self.parents = dict()

            def discover_vertex(self, v, _):
                u = self.parents.get(v, None)
                if u is not None:
                    self.edges.append((u, v))

            def edge_relaxed(self, edge):
                u, v, _ = edge
                self.parents[v] = u

        vis = DijkstraTreeEdgesRecorder()
        rustworkx.graph_dijkstra_search(self.graph, [0], float, vis)
        self.assertEqual(vis.edges, [(0, 1), (0, 2), (2, 6), (2, 5), (5, 3)])

    def test_graph_dijkstra_tree_edges_no_starting_point(self):
        class DijkstraTreeEdgesRecorder(rustworkx.visit.DijkstraVisitor):
            def __init__(self):
                self.edges = []
                self.parents = dict()

            def discover_vertex(self, v, _):
                u = self.parents.get(v, None)
                if u is not None:
                    self.edges.append((u, v))

            def edge_relaxed(self, edge):
                u, v, _ = edge
                self.parents[v] = u

        vis = DijkstraTreeEdgesRecorder()
        rustworkx.graph_dijkstra_search(self.graph, None, float, vis)
        self.assertEqual(vis.edges, [(0, 1), (0, 2), (2, 6), (2, 5), (5, 3), (4, 7)])

    def test_graph_dijkstra_goal_search_with_stop_search_exception(self):
        class GoalSearch(rustworkx.visit.DijkstraVisitor):

            goal = 3

            def __init__(self):
                self.parents = {}
                self.opt_goal_cost = None

            def discover_vertex(self, v, score):
                if v == self.goal:
                    self.opt_goal_cost = score
                    raise rustworkx.visit.StopSearch

            def edge_relaxed(self, edge):
                u, v, _ = edge
                self.parents[v] = u

            def reconstruct_path(self):
                v = self.goal
                path = [v]
                while v in self.parents:
                    v = self.parents[v]
                    path.append(v)

                path.reverse()
                return path

        vis = GoalSearch()
        rustworkx.graph_dijkstra_search(self.graph, [0], float, vis)
        self.assertEqual(vis.reconstruct_path(), [0, 2, 5, 3])
        self.assertEqual(vis.opt_goal_cost, 4.0)

    def test_graph_dijkstra_goal_search_with_custom_exception(self):
        class StopIfGoalFound(Exception):
            pass

        class GoalSearch(rustworkx.visit.DijkstraVisitor):

            goal = 3

            def __init__(self):
                self.parents = {}
                self.opt_goal_cost = None

            def discover_vertex(self, v, score):
                if v == self.goal:
                    self.opt_goal_cost = score
                    raise StopIfGoalFound

            def edge_relaxed(self, edge):
                u, v, _ = edge
                self.parents[v] = u

            def reconstruct_path(self):
                v = self.goal
                path = [v]
                while v in self.parents:
                    v = self.parents[v]
                    path.append(v)

                path.reverse()
                return path

        vis = GoalSearch()
        try:
            rustworkx.graph_dijkstra_search(self.graph, [0], float, vis)
        except StopIfGoalFound:
            pass
        self.assertEqual(vis.reconstruct_path(), [0, 2, 5, 3])
        self.assertEqual(vis.opt_goal_cost, 4.0)

    def test_graph_dijkstra_goal_search_with_prohibited_edges(self):
        class GoalSearch(rustworkx.visit.DijkstraVisitor):

            goal = 3
            prohibited = [(5, 3)]

            def __init__(self):
                self.parents = {}
                self.opt_goal_cost = None

            def discover_vertex(self, v, score):
                if v == self.goal:
                    self.opt_goal_cost = score
                    raise rustworkx.visit.StopSearch

            def examine_edge(self, edge):
                u, v, _ = edge
                if (u, v) in self.prohibited:
                    raise rustworkx.visit.PruneSearch

            def edge_relaxed(self, edge):
                u, v, _ = edge
                self.parents[v] = u

            def reconstruct_path(self):
                v = self.goal
                path = [v]
                while v in self.parents:
                    v = self.parents[v]
                    path.append(v)

                path.reverse()
                return path

        vis = GoalSearch()
        rustworkx.graph_dijkstra_search(self.graph, [0], float, vis)
        self.assertEqual(vis.reconstruct_path(), [0, 1, 3])
        self.assertEqual(vis.opt_goal_cost, 11.0)

    def test_graph_prune_edge_not_relaxed(self):
        class PruneEdgeNotRelaxed(rustworkx.visit.DijkstraVisitor):
            def edge_not_relaxed(self, _):
                raise rustworkx.visit.PruneSearch

        vis = PruneEdgeNotRelaxed()
        rustworkx.graph_dijkstra_search(self.graph, [0], float, vis)

    def test_invalid_source(self):
        graph = rustworkx.PyGraph()
        with self.assertRaises(IndexError):
            rustworkx.dijkstra_search(graph, [1], float, rustworkx.visit.DijkstraVisitor())
