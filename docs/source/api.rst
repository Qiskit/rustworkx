.. _retworkx:

######################
Retworkx API Reference
######################

Graph Classes
=============

.. autosummary::
   :toctree: apiref

    retworkx.PyGraph
    retworkx.PyDiGraph
    retworkx.PyDAG

.. _algorithm_api:

Algorithm Functions
===================

.. _shortest-paths:

Shortest Paths
--------------

.. autosummary::
   :toctree: apiref

   retworkx.dijkstra_shortest_paths
   retworkx.dijkstra_shortest_path_lengths
   retworkx.all_pairs_dijkstra_shortest_paths
   retworkx.all_pairs_dijkstra_path_lengths
   retworkx.bellman_ford_shortest_paths
   retworkx.bellman_ford_shortest_path_lengths
   retworkx.negative_edge_cycle
   retworkx.find_negative_cycle
   retworkx.distance_matrix
   retworkx.floyd_warshall
   retworkx.floyd_warshall_numpy
   retworkx.astar_shortest_path
   retworkx.k_shortest_path_lengths
   retworkx.num_shortest_paths_unweighted
   retworkx.unweighted_average_shortest_path_length

.. _centrality:

Centrality
--------------

.. autosummary::
   :toctree: apiref

   retworkx.betweenness_centrality

.. _traversal:

Traversal
---------

.. autosummary::
   :toctree: apiref

   retworkx.dfs_edges
   retworkx.dfs_search
   retworkx.bfs_successors
   retworkx.bfs_search
   retworkx.dijkstra_search
   retworkx.topological_sort
   retworkx.lexicographical_topological_sort
   retworkx.descendants
   retworkx.ancestors
   retworkx.collect_runs
   retworkx.collect_bicolor_runs
   retworkx.visit.DFSVisitor
   retworkx.visit.BFSVisitor
   retworkx.visit.DijkstraVisitor
   retworkx.TopologicalSorter

.. _dag-algorithms:

DAG Algorithms
--------------

.. autosummary::
   :toctree: apiref

   retworkx.dag_longest_path
   retworkx.dag_longest_path_length
   retworkx.dag_weighted_longest_path
   retworkx.dag_weighted_longest_path_length
   retworkx.is_directed_acyclic_graph
   retworkx.layers

.. _tree:

Tree
----

.. autosummary::
   :toctree: apiref

   retworkx.minimum_spanning_edges
   retworkx.minimum_spanning_tree
   retworkx.steiner_tree
   retworkx.bipartition_tree

.. _isomorphism:

Isomorphism
-----------

.. autosummary::
   :toctree: apiref

   retworkx.is_isomorphic
   retworkx.is_subgraph_isomorphic
   retworkx.is_isomorphic_node_match
   retworkx.vf2_mapping

.. _matching:

Matching
--------

.. autosummary::
   :toctree: apiref

   retworkx.max_weight_matching
   retworkx.is_matching
   retworkx.is_maximal_matching

.. _connectivity-cycle-finding:

Connectivity and Cycles
-----------------------

.. autosummary::
   :toctree: apiref

   retworkx.number_connected_components
   retworkx.connected_components
   retworkx.node_connected_component
   retworkx.is_connected
   retworkx.strongly_connected_components
   retworkx.number_weakly_connected_components
   retworkx.weakly_connected_components
   retworkx.is_weakly_connected
   retworkx.cycle_basis
   retworkx.digraph_find_cycle
   retworkx.articulation_points
   retworkx.biconnected_components
   retworkx.chain_decomposition
   retworkx.all_simple_paths
   retworkx.all_pairs_all_simple_paths

.. _graph-ops:

Graph Operations
----------------

.. autosummary::
   :toctree: apiref

   retworkx.complement
   retworkx.union
   retworkx.cartesian_product

.. _other-algorithms:

Other Algorithm Functions
-------------------------

.. autosummary::
   :toctree: apiref

   retworkx.adjacency_matrix
   retworkx.transitivity
   retworkx.core_number
   retworkx.graph_greedy_color
   retworkx.metric_closure
   retworkx.bipartition_graph_mst

.. _generator_funcs:

Generators
==========

.. autosummary::
   :toctree: apiref

    retworkx.generators.cycle_graph
    retworkx.generators.directed_cycle_graph
    retworkx.generators.path_graph
    retworkx.generators.directed_path_graph
    retworkx.generators.star_graph
    retworkx.generators.directed_star_graph
    retworkx.generators.mesh_graph
    retworkx.generators.directed_mesh_graph
    retworkx.generators.grid_graph
    retworkx.generators.directed_grid_graph
    retworkx.generators.binomial_tree_graph
    retworkx.generators.directed_binomial_tree_graph
    retworkx.generators.hexagonal_lattice_graph
    retworkx.generators.directed_hexagonal_lattice_graph
    retworkx.generators.heavy_square_graph
    retworkx.generators.directed_heavy_square_graph
    retworkx.generators.heavy_hex_graph
    retworkx.generators.directed_heavy_hex_graph
    retworkx.generators.lollipop_graph
    retworkx.generators.generalized_petersen_graph
    retworkx.generators.barbell_graph
    retworkx.generators.full_rary_tree

.. _random_generators:

Random Graph Generator Functions
================================

.. autosummary::
   :toctree: apiref

    retworkx.directed_gnp_random_graph
    retworkx.undirected_gnp_random_graph
    retworkx.directed_gnm_random_graph
    retworkx.undirected_gnm_random_graph
    retworkx.random_geometric_graph

.. _layout-functions:

Layout Functions
================

.. autosummary::
   :toctree: apiref

   retworkx.random_layout
   retworkx.spring_layout
   retworkx.bipartite_layout
   retworkx.circular_layout
   retworkx.shell_layout
   retworkx.spiral_layout


.. _graphml:

GraphML
==========

.. autosummary::
   :toctree: apiref

   retworkx.read_graphml

.. _converters:

Converters
==========

.. autosummary::
   :toctree: apiref

   retworkx.networkx_converter

.. _api-functions-pydigraph:

API functions for PyDigraph
===========================

These functions are algorithm functions that are type specific for
:class:`~retworkx.PyDiGraph` or :class:`~retworkx.PyDAG` objects. Universal
functions from Retworkx API that work for both graph types internally call
the functions from the explicitly typed based on the data type.

.. autosummary::
   :toctree: apiref

   retworkx.digraph_is_isomorphic
   retworkx.digraph_is_subgraph_isomorphic
   retworkx.digraph_vf2_mapping
   retworkx.digraph_distance_matrix
   retworkx.digraph_floyd_warshall
   retworkx.digraph_floyd_warshall_numpy
   retworkx.digraph_adjacency_matrix
   retworkx.digraph_all_simple_paths
   retworkx.digraph_all_pairs_all_simple_paths
   retworkx.digraph_astar_shortest_path
   retworkx.digraph_dijkstra_shortest_paths
   retworkx.digraph_all_pairs_dijkstra_shortest_paths
   retworkx.digraph_dijkstra_shortest_path_lengths
   retworkx.digraph_all_pairs_dijkstra_path_lengths
   retworkx.digraph_bellman_ford_shortest_path_lengths
   retworkx.digraph_bellman_ford_shortest_path_lengths
   retworkx.digraph_k_shortest_path_lengths
   retworkx.digraph_dfs_edges
   retworkx.digraph_dfs_search
   retworkx.digraph_find_cycle
   retworkx.digraph_transitivity
   retworkx.digraph_core_number
   retworkx.digraph_complement
   retworkx.digraph_union
   retworkx.digraph_tensor_product
   retworkx.digraph_cartesian_product
   retworkx.digraph_random_layout
   retworkx.digraph_bipartite_layout
   retworkx.digraph_circular_layout
   retworkx.digraph_shell_layout
   retworkx.digraph_spiral_layout
   retworkx.digraph_spring_layout
   retworkx.digraph_num_shortest_paths_unweighted
   retworkx.digraph_betweenness_centrality
   retworkx.digraph_unweighted_average_shortest_path_length
   retworkx.digraph_bfs_search
   retworkx.digraph_dijkstra_search

.. _api-functions-pygraph:

API functions for PyGraph
=========================

These functions are algorithm functions that are type specific for
:class:`~retworkx.PyGraph` objects. Universal functions from Retworkx API that
work for both graph types internally call the functions from the explicitly
typed API based on the data type.

.. autosummary::
   :toctree: apiref

   retworkx.graph_is_isomorphic
   retworkx.graph_is_subgraph_isomorphic
   retworkx.graph_vf2_mapping
   retworkx.graph_distance_matrix
   retworkx.graph_floyd_warshall
   retworkx.graph_floyd_warshall_numpy
   retworkx.graph_adjacency_matrix
   retworkx.graph_all_simple_paths
   retworkx.graph_all_pairs_all_simple_paths
   retworkx.graph_astar_shortest_path
   retworkx.graph_dijkstra_shortest_paths
   retworkx.graph_dijkstra_shortest_path_lengths
   retworkx.graph_all_pairs_dijkstra_shortest_paths
   retworkx.graph_k_shortest_path_lengths
   retworkx.graph_all_pairs_dijkstra_path_lengths
   retworkx.graph_bellman_ford_shortest_path_lengths
   retworkx.graph_bellman_ford_shortest_path_lengths
   retworkx.graph_dfs_edges
   retworkx.graph_dfs_search
   retworkx.graph_transitivity
   retworkx.graph_core_number
   retworkx.graph_complement
   retworkx.graph_union
   retworkx.graph_tensor_product
   retworkx.graph_cartesian_product
   retworkx.graph_random_layout
   retworkx.graph_bipartite_layout
   retworkx.graph_circular_layout
   retworkx.graph_shell_layout
   retworkx.graph_spiral_layout
   retworkx.graph_spring_layout
   retworkx.graph_num_shortest_paths_unweighted
   retworkx.graph_betweenness_centrality
   retworkx.graph_unweighted_average_shortest_path_length
   retworkx.graph_bfs_search
   retworkx.graph_dijkstra_search

Exceptions
==========

.. autosummary::
   :toctree: apiref

   retworkx.InvalidNode
   retworkx.DAGWouldCycle
   retworkx.NoEdgeBetweenNodes
   retworkx.DAGHasCycle
   retworkx.NegativeCycle
   retworkx.NoSuitableNeighbors
   retworkx.NoPathFound
   retworkx.NullGraph
   retworkx.visit.StopSearch
   retworkx.visit.PruneSearch

Custom Return Types
===================

.. autosummary::
   :toctree: apiref

   retworkx.BFSSuccessors
   retworkx.NodeIndices
   retworkx.EdgeIndices
   retworkx.EdgeList
   retworkx.WeightedEdgeList
   retworkx.EdgeIndexMap
   retworkx.PathMapping
   retworkx.PathLengthMapping
   retworkx.Pos2DMapping
   retworkx.AllPairsPathMapping
   retworkx.AllPairsPathLengthMapping
   retworkx.CentralityMapping
   retworkx.Chains
   retworkx.NodeMap
   retworkx.ProductNodeMap
   retworkx.BiconnectedComponents
