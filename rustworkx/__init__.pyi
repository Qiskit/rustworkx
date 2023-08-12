# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/lib.rs

from .rustworkx import *
from typing import Generic, TypeVar

from .centrality import digraph_eigenvector_centrality as digraph_eigenvector_centrality
from .centrality import graph_eigenvector_centrality as graph_eigenvector_centrality
from .centrality import digraph_betweenness_centrality as digraph_betweenness_centrality
from .centrality import graph_betweenness_centrality as graph_betweenness_centrality
from .centrality import digraph_edge_betweenness_centrality as digraph_edge_betweenness_centrality
from .centrality import graph_edge_betweenness_centrality as graph_edge_betweenness_centrality
from .centrality import digraph_closeness_centrality as digraph_closeness_centrality
from .centrality import graph_closeness_centrality as graph_closeness_centrality
from .centrality import digraph_katz_centrality as digraph_katz_centrality
from .centrality import graph_katz_centrality as graph_katz_centrality

from .isomorphism import digraph_is_isomorphic as digraph_is_isomorphic
from .isomorphism import graph_is_isomorphic as graph_is_isomorphic
from .isomorphism import digraph_is_subgraph_isomorphic as digraph_is_subgraph_isomorphic
from .isomorphism import graph_is_subgraph_isomorphic as graph_is_subgraph_isomorphic

from .layout import digraph_bipartite_layout as digraph_bipartite_layout
from .layout import graph_bipartite_layout as graph_bipartite_layout
from .layout import digraph_circular_layout as digraph_circular_layout
from .layout import graph_circular_layout as graph_circular_layout
from .layout import digraph_random_layout as digraph_random_layout
from .layout import graph_random_layout as graph_random_layout
from .layout import graph_shell_layout as graph_shell_layout
from .layout import digraph_spiral_layout as digraph_spiral_layout
from .layout import graph_spiral_layout as graph_spiral_layout
from .layout import digraph_spring_layout as digraph_spring_layout
from .layout import graph_spring_layout as graph_spring_layout

from .link_analysis import hits as hits
from .link_analysis import pagerank as pagerank

from .shortest_path import (
    digraph_bellman_ford_shortest_paths as digraph_bellman_ford_shortest_paths,
)
from .shortest_path import graph_bellman_ford_shortest_paths as graph_bellman_ford_shortest_paths
from .shortest_path import (
    digraph_bellman_ford_shortest_path_lengths as digraph_bellman_ford_shortest_path_lengths,
)
from .shortest_path import (
    graph_bellman_ford_shortest_path_lengths as graph_bellman_ford_shortest_path_lengths,
)
from .shortest_path import digraph_dijkstra_shortest_paths as digraph_dijkstra_shortest_paths
from .shortest_path import graph_dijkstra_shortest_paths as graph_dijkstra_shortest_paths
from .shortest_path import (
    digraph_dijkstra_shortest_path_lengths as digraph_dijkstra_shortest_path_lengths,
)
from .shortest_path import (
    graph_dijkstra_shortest_path_lengths as graph_dijkstra_shortest_path_lengths,
)
from .shortest_path import (
    digraph_all_pairs_bellman_ford_path_lengths as digraph_all_pairs_bellman_ford_path_lengths,
)
from .shortest_path import (
    graph_all_pairs_bellman_ford_path_lengths as graph_all_pairs_bellman_ford_path_lengths,
)
from .shortest_path import (
    digraph_all_pairs_bellman_ford_shortest_paths as digraph_all_pairs_bellman_ford_shortest_paths,
)
from .shortest_path import (
    graph_all_pairs_bellman_ford_shortest_paths as graph_all_pairs_bellman_ford_shortest_paths,
)
from .shortest_path import (
    digraph_all_pairs_dijkstra_path_lengths as digraph_all_pairs_dijkstra_path_lengths,
)
from .shortest_path import (
    graph_all_pairs_dijkstra_path_lengths as graph_all_pairs_dijkstra_path_lengths,
)
from .shortest_path import (
    digraph_all_pairs_dijkstra_shortest_paths as digraph_all_pairs_dijkstra_shortest_paths,
)
from .shortest_path import (
    graph_all_pairs_dijkstra_shortest_paths as graph_all_pairs_dijkstra_shortest_paths,
)
from .shortest_path import digraph_astar_shortest_path as digraph_astar_shortest_path
from .shortest_path import graph_astar_shortest_path as graph_astar_shortest_path
from .shortest_path import digraph_k_shortest_path_lengths as digraph_k_shortest_path_lengths
from .shortest_path import graph_k_shortest_path_lengths as graph_k_shortest_path_lengths
from .shortest_path import digraph_has_path as digraph_has_path
from .shortest_path import graph_has_path as graph_has_path
from .shortest_path import (
    digraph_num_shortest_paths_unweighted as digraph_num_shortest_paths_unweighted,
)
from .shortest_path import (
    graph_num_shortest_paths_unweighted as graph_num_shortest_paths_unweighted,
)
from .shortest_path import (
    digraph_unweighted_average_shortest_path_length as digraph_unweighted_average_shortest_path_length,
)
from .shortest_path import digraph_distance_matrix as digraph_distance_matrix
from .shortest_path import graph_distance_matrix as graph_distance_matrix
from .shortest_path import digraph_floyd_warshall as digraph_floyd_warshall
from .shortest_path import graph_floyd_warshall as graph_floyd_warshall
from .shortest_path import digraph_floyd_warshall_numpy as digraph_floyd_warshall_numpy
from .shortest_path import graph_floyd_warshall_numpy as graph_floyd_warshall_numpy

from .traversal import digraph_bfs_search as digraph_bfs_search
from .traversal import graph_bfs_search as graph_bfs_search
from .traversal import digraph_dfs_search as digraph_dfs_search
from .traversal import graph_dfs_search as graph_dfs_search
from .traversal import digraph_dijkstra_search as digraph_dijkstra_search
from .traversal import graph_dijkstra_search as graph_dijkstra_search
from .traversal import digraph_dfs_edges as digraph_dfs_edges
from .traversal import graph_dfs_edges as graph_dfs_edges
from .traversal import ancestors as ancestors
from .traversal import bfs_predecessors as bfs_predecessors
from .traversal import bfs_successors as bfs_successors
from .traversal import descendants as descendants

from .tree import minimum_spanning_edges as minimum_spanning_edges
from .tree import minimum_spanning_tree as minimum_spanning_tree
from .tree import steiner_tree as steiner_tree

S = TypeVar("S")
T = TypeVar("T")

class PyDAG(Generic[S, T], PyDiGraph[S, T]): ...
