.. _api-functions-pydigraph:

API functions for PyDigraph
===========================

These functions are algorithm functions that are type specific for
:class:`~rustworkx.PyDiGraph` or :class:`~rustworkx.PyDAG` objects. Universal
functions from Retworkx API that work for both graph types internally call
the functions from the explicitly typed based on the data type.

.. autosummary::
   :toctree: ../apiref

   rustworkx.digraph_is_isomorphic
   rustworkx.digraph_is_subgraph_isomorphic
   rustworkx.digraph_vf2_mapping
   rustworkx.digraph_distance_matrix
   rustworkx.digraph_floyd_warshall
   rustworkx.digraph_floyd_warshall_numpy
   rustworkx.digraph_adjacency_matrix
   rustworkx.digraph_all_simple_paths
   rustworkx.digraph_all_pairs_all_simple_paths
   rustworkx.digraph_astar_shortest_path
   rustworkx.digraph_dijkstra_shortest_paths
   rustworkx.digraph_all_pairs_dijkstra_shortest_paths
   rustworkx.digraph_dijkstra_shortest_path_lengths
   rustworkx.digraph_all_pairs_dijkstra_path_lengths
   rustworkx.digraph_bellman_ford_shortest_path_lengths
   rustworkx.digraph_bellman_ford_shortest_path_lengths
   rustworkx.digraph_all_pairs_bellman_ford_shortest_paths
   rustworkx.digraph_all_pairs_bellman_ford_path_lengths
   rustworkx.digraph_k_shortest_path_lengths
   rustworkx.digraph_dfs_edges
   rustworkx.digraph_dfs_search
   rustworkx.digraph_find_cycle
   rustworkx.digraph_transitivity
   rustworkx.digraph_core_number
   rustworkx.digraph_complement
   rustworkx.digraph_union
   rustworkx.digraph_tensor_product
   rustworkx.digraph_cartesian_product
   rustworkx.digraph_random_layout
   rustworkx.digraph_bipartite_layout
   rustworkx.digraph_circular_layout
   rustworkx.digraph_shell_layout
   rustworkx.digraph_spiral_layout
   rustworkx.digraph_spring_layout
   rustworkx.digraph_num_shortest_paths_unweighted
   rustworkx.digraph_betweenness_centrality
   rustworkx.digraph_edge_betweenness_centrality
   rustworkx.digraph_closeness_centrality
   rustworkx.digraph_eigenvector_centrality
   rustworkx.digraph_katz_centrality
   rustworkx.digraph_unweighted_average_shortest_path_length
   rustworkx.digraph_bfs_search
   rustworkx.digraph_dijkstra_search
   rustworkx.digraph_node_link_json
   rustworkx.digraph_longest_simple_path
