.. _rustworkx:

#######################
Rustworkx API Reference
#######################

Graph Classes
=============

.. autosummary::
   :toctree: apiref

    rustworkx.PyGraph
    rustworkx.PyDiGraph
    rustworkx.PyDAG

.. _algorithm_api:

Algorithm Functions
===================

.. _shortest-paths:

Shortest Paths
--------------

.. autosummary::
   :toctree: apiref

   rustworkx.dijkstra_shortest_paths
   rustworkx.dijkstra_shortest_path_lengths
   rustworkx.all_pairs_dijkstra_shortest_paths
   rustworkx.all_pairs_dijkstra_path_lengths
   rustworkx.bellman_ford_shortest_paths
   rustworkx.bellman_ford_shortest_path_lengths
   rustworkx.all_pairs_bellman_ford_shortest_paths
   rustworkx.all_pairs_bellman_ford_path_lengths
   rustworkx.negative_edge_cycle
   rustworkx.find_negative_cycle
   rustworkx.distance_matrix
   rustworkx.floyd_warshall
   rustworkx.floyd_warshall_numpy
   rustworkx.astar_shortest_path
   rustworkx.k_shortest_path_lengths
   rustworkx.num_shortest_paths_unweighted
   rustworkx.unweighted_average_shortest_path_length

.. _centrality:

Centrality
--------------

.. autosummary::
   :toctree: apiref

   rustworkx.betweenness_centrality
   rustworkx.eigenvector_centrality

.. _traversal:

Traversal
---------

.. autosummary::
   :toctree: apiref

   rustworkx.dfs_edges
   rustworkx.dfs_search
   rustworkx.bfs_successors
   rustworkx.bfs_search
   rustworkx.dijkstra_search
   rustworkx.topological_sort
   rustworkx.lexicographical_topological_sort
   rustworkx.descendants
   rustworkx.ancestors
   rustworkx.collect_runs
   rustworkx.collect_bicolor_runs
   rustworkx.visit.DFSVisitor
   rustworkx.visit.BFSVisitor
   rustworkx.visit.DijkstraVisitor
   rustworkx.TopologicalSorter

.. _dag-algorithms:

DAG Algorithms
--------------

.. autosummary::
   :toctree: apiref

   rustworkx.dag_longest_path
   rustworkx.dag_longest_path_length
   rustworkx.dag_weighted_longest_path
   rustworkx.dag_weighted_longest_path_length
   rustworkx.is_directed_acyclic_graph
   rustworkx.layers

.. _tree:

Tree
----

.. autosummary::
   :toctree: apiref

   rustworkx.minimum_spanning_edges
   rustworkx.minimum_spanning_tree
   rustworkx.steiner_tree

.. _isomorphism:

Isomorphism
-----------

.. autosummary::
   :toctree: apiref

   rustworkx.is_isomorphic
   rustworkx.is_subgraph_isomorphic
   rustworkx.is_isomorphic_node_match
   rustworkx.vf2_mapping

.. _matching:

Matching
--------

.. autosummary::
   :toctree: apiref

   rustworkx.max_weight_matching
   rustworkx.is_matching
   rustworkx.is_maximal_matching

.. _connectivity-cycle-finding:

Connectivity and Cycles
-----------------------

.. autosummary::
   :toctree: apiref

   rustworkx.number_connected_components
   rustworkx.connected_components
   rustworkx.node_connected_component
   rustworkx.is_connected
   rustworkx.strongly_connected_components
   rustworkx.number_weakly_connected_components
   rustworkx.weakly_connected_components
   rustworkx.is_weakly_connected
   rustworkx.cycle_basis
   rustworkx.simple_cycles
   rustworkx.digraph_find_cycle
   rustworkx.articulation_points
   rustworkx.biconnected_components
   rustworkx.chain_decomposition
   rustworkx.all_simple_paths
   rustworkx.all_pairs_all_simple_paths
   rustworkx.stoer_wagner_min_cut

.. _graph-ops:

Graph Operations
----------------

.. autosummary::
   :toctree: apiref

   rustworkx.complement
   rustworkx.union
   rustworkx.cartesian_product

.. _other-algorithms:

Other Algorithm Functions
-------------------------

.. autosummary::
   :toctree: apiref

   rustworkx.adjacency_matrix
   rustworkx.transitivity
   rustworkx.core_number
   rustworkx.graph_greedy_color
   rustworkx.metric_closure
   rustworkx.is_planar

.. _generator_funcs:

Generators
==========

.. autosummary::
   :toctree: apiref

    rustworkx.generators.cycle_graph
    rustworkx.generators.directed_cycle_graph
    rustworkx.generators.path_graph
    rustworkx.generators.directed_path_graph
    rustworkx.generators.star_graph
    rustworkx.generators.directed_star_graph
    rustworkx.generators.mesh_graph
    rustworkx.generators.directed_mesh_graph
    rustworkx.generators.grid_graph
    rustworkx.generators.directed_grid_graph
    rustworkx.generators.binomial_tree_graph
    rustworkx.generators.directed_binomial_tree_graph
    rustworkx.generators.hexagonal_lattice_graph
    rustworkx.generators.directed_hexagonal_lattice_graph
    rustworkx.generators.heavy_square_graph
    rustworkx.generators.directed_heavy_square_graph
    rustworkx.generators.heavy_hex_graph
    rustworkx.generators.directed_heavy_hex_graph
    rustworkx.generators.lollipop_graph
    rustworkx.generators.generalized_petersen_graph
    rustworkx.generators.barbell_graph
    rustworkx.generators.full_rary_tree

.. _random_generators:

Random Graph Generator Functions
================================

.. autosummary::
   :toctree: apiref

    rustworkx.directed_gnp_random_graph
    rustworkx.undirected_gnp_random_graph
    rustworkx.directed_gnm_random_graph
    rustworkx.undirected_gnm_random_graph
    rustworkx.random_geometric_graph

.. _layout-functions:

Layout Functions
================

.. autosummary::
   :toctree: apiref

   rustworkx.random_layout
   rustworkx.spring_layout
   rustworkx.bipartite_layout
   rustworkx.circular_layout
   rustworkx.shell_layout
   rustworkx.spiral_layout


.. _serialization:

Serialization
=============

.. autosummary::
   :toctree: apiref

   rustworkx.node_link_json
   rustworkx.read_graphml

.. _converters:

Converters
==========

.. autosummary::
   :toctree: apiref

   rustworkx.networkx_converter

.. _api-functions-pydigraph:

API functions for PyDigraph
===========================

These functions are algorithm functions that are type specific for
:class:`~rustworkx.PyDiGraph` or :class:`~rustworkx.PyDAG` objects. Universal
functions from Retworkx API that work for both graph types internally call
the functions from the explicitly typed based on the data type.

.. autosummary::
   :toctree: apiref

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
   rustworkx.digraph_eigenvector_centrality
   rustworkx.digraph_unweighted_average_shortest_path_length
   rustworkx.digraph_bfs_search
   rustworkx.digraph_dijkstra_search
   rustworkx.digraph_node_link_json

.. _api-functions-pygraph:

API functions for PyGraph
=========================

These functions are algorithm functions that are type specific for
:class:`~rustworkx.PyGraph` objects. Universal functions from Rustworkx API that
work for both graph types internally call the functions from the explicitly
typed API based on the data type.

.. autosummary::
   :toctree: apiref

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
   rustworkx.graph_cartesian_product
   rustworkx.graph_random_layout
   rustworkx.graph_bipartite_layout
   rustworkx.graph_circular_layout
   rustworkx.graph_shell_layout
   rustworkx.graph_spiral_layout
   rustworkx.graph_spring_layout
   rustworkx.graph_num_shortest_paths_unweighted
   rustworkx.graph_betweenness_centrality
   rustworkx.graph_eigenvector_centrality
   rustworkx.graph_unweighted_average_shortest_path_length
   rustworkx.graph_bfs_search
   rustworkx.graph_dijkstra_search
   rustworkx.graph_node_link_json

Exceptions
==========

.. autosummary::
   :toctree: apiref

   rustworkx.InvalidNode
   rustworkx.DAGWouldCycle
   rustworkx.NoEdgeBetweenNodes
   rustworkx.DAGHasCycle
   rustworkx.NegativeCycle
   rustworkx.NoSuitableNeighbors
   rustworkx.NoPathFound
   rustworkx.NullGraph
   rustworkx.visit.StopSearch
   rustworkx.visit.PruneSearch
   rustworkx.JSONSerializationError

Custom Return Types
===================

.. autosummary::
   :toctree: apiref

   rustworkx.BFSSuccessors
   rustworkx.NodeIndices
   rustworkx.EdgeIndices
   rustworkx.EdgeList
   rustworkx.WeightedEdgeList
   rustworkx.EdgeIndexMap
   rustworkx.PathMapping
   rustworkx.PathLengthMapping
   rustworkx.Pos2DMapping
   rustworkx.AllPairsPathMapping
   rustworkx.AllPairsPathLengthMapping
   rustworkx.CentralityMapping
   rustworkx.Chains
   rustworkx.NodeMap
   rustworkx.ProductNodeMap
   rustworkx.BiconnectedComponents
