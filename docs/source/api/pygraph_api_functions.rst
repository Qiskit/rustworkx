.. _api-functions-pygraph:

API functions for PyGraph
=========================

These functions are algorithm functions that are type specific for
:class:`~rustworkx.PyGraph` objects. Universal functions from Rustworkx API that
work for both graph types internally call the functions from the explicitly
typed API based on the data type.

.. autosummary::
   :toctree: ../apiref

   rustworkx.graph_is_isomorphic
   rustworkx.graph_is_subgraph_isomorphic
   rustworkx.graph_vf2_mapping
   rustworkx.graph_distance_matrix
   rustworkx.graph_floyd_warshall
   rustworkx.graph_floyd_warshall_numpy
   rustworkx.graph_adjacency_matrix
   rustworkx.graph_all_simple_paths
   rustworkx.graph_all_pairs_all_simple_paths
   rustworkx.graph_astar_shortest_path
   rustworkx.graph_dijkstra_shortest_paths
   rustworkx.graph_dijkstra_shortest_path_lengths
   rustworkx.graph_all_pairs_dijkstra_shortest_paths
   rustworkx.graph_k_shortest_path_lengths
   rustworkx.graph_all_pairs_dijkstra_path_lengths
   rustworkx.graph_bellman_ford_shortest_path_lengths
   rustworkx.graph_bellman_ford_shortest_path_lengths
   rustworkx.graph_all_pairs_bellman_ford_shortest_paths
   rustworkx.graph_all_pairs_bellman_ford_path_lengths
   rustworkx.graph_dfs_edges
   rustworkx.graph_dfs_search
   rustworkx.graph_transitivity
   rustworkx.graph_core_number
   rustworkx.graph_complement
   rustworkx.graph_union
   rustworkx.graph_tensor_product
   rustworkx.graph_token_swapper
   rustworkx.graph_cartesian_product
   rustworkx.graph_random_layout
   rustworkx.graph_bipartite_layout
   rustworkx.graph_circular_layout
   rustworkx.graph_shell_layout
   rustworkx.graph_spiral_layout
   rustworkx.graph_spring_layout
   rustworkx.graph_num_shortest_paths_unweighted
   rustworkx.graph_betweenness_centrality
   rustworkx.graph_edge_betweenness_centrality
   rustworkx.graph_closeness_centrality
   rustworkx.graph_eigenvector_centrality
   rustworkx.graph_katz_centrality
   rustworkx.graph_unweighted_average_shortest_path_length
   rustworkx.graph_bfs_search
   rustworkx.graph_dijkstra_search
   rustworkx.graph_node_link_json
   rustworkx.graph_longest_simple_path
