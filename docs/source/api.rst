.. _retworkx:

######################
Retworkx API Reference
######################

Graph Classes
=============

.. autosummary::
   :toctree: stubs

    retworkx.PyGraph
    retworkx.PyDiGraph
    retworkx.PyDAG

Algorithm Functions
===================

.. _shortest-paths:

Shortest Paths
--------------

.. autosummary::
   :toctree: stubs

   retworkx.dijkstra_shortest_paths
   retworkx.dijkstra_shortest_path_lengths
   retworkx.all_pairs_dijkstra_shortest_paths
   retworkx.all_pairs_dijkstra_path_lengths
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
   :toctree: stubs

   retworkx.betweenness_centrality

.. _traversal:

Traversal
---------

.. autosummary::
   :toctree: stubs

   retworkx.dfs_edges
   retworkx.bfs_successors
   retworkx.topological_sort
   retworkx.lexicographical_topological_sort
   retworkx.descendants
   retworkx.ancestors
   retworkx.collect_runs
   retworkx.collect_bicolor_runs

.. _dag-algorithms:

DAG Algorithms
--------------

.. autosummary::
   :toctree: stubs

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
   :toctree: stubs

   retworkx.minimum_spanning_edges
   retworkx.minimum_spanning_tree
   retworkx.steiner_tree

.. _isomorphism:

Isomorphism
-----------

.. autosummary::
   :toctree: stubs

   retworkx.is_isomorphic
   retworkx.is_subgraph_isomorphic
   retworkx.is_isomorphic_node_match
   retworkx.vf2_mapping

.. _matching:

Matching
--------

.. autosummary::
   :toctree: stubs

   retworkx.max_weight_matching
   retworkx.is_matching
   retworkx.is_maximal_matching

.. _connectivity-cycle-finding:

Connectivity and Cycles
-----------------------

.. autosummary::
   :toctree: stubs

   retworkx.strongly_connected_components
   retworkx.number_weakly_connected_components
   retworkx.weakly_connected_components
   retworkx.is_weakly_connected
   retworkx.cycle_basis
   retworkx.digraph_find_cycle
   retworkx.chain_decomposition

.. _other-algorithms:

Other Algorithm Functions
-------------------------

.. autosummary::
   :toctree: stubs

   retworkx.complement
   retworkx.adjacency_matrix
   retworkx.all_simple_paths
   retworkx.transitivity
   retworkx.core_number
   retworkx.graph_greedy_color
   retworkx.union
   retworkx.metric_closure

Generators
==========

.. autosummary::
   :toctree: stubs

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
    retworkx.generators.hexagonal_lattice_graph
    retworkx.generators.directed_hexagonal_lattice_graph
    retworkx.generators.heavy_square_graph
    retworkx.generators.directed_heavy_square_graph
    retworkx.generators.heavy_hex_graph
    retworkx.generators.directed_heavy_hex_graph

Random Circuit Functions
========================

.. autosummary::
   :toctree: stubs

    retworkx.directed_gnp_random_graph
    retworkx.undirected_gnp_random_graph
    retworkx.directed_gnm_random_graph
    retworkx.undirected_gnm_random_graph
    retworkx.random_geometric_graph

.. _layout-functions:

Layout Functions
================

.. autosummary::
   :toctree: stubs

   retworkx.random_layout
   retworkx.spring_layout
   retworkx.bipartite_layout
   retworkx.circular_layout
   retworkx.shell_layout
   retworkx.spiral_layout


.. _converters:

Converters
==========

.. autosummary::
   :toctree: stubs

   retworkx.networkx_converter

.. _api-functions-pydigraph:

API functions for PyDigraph
===========================

These functions are algorithm functions that are type specific for
:class:`~retworkx.PyDiGraph` or :class:`~retworkx.PyDAG` objects. Universal
functions from Retworkx API that work for both graph types internally call
the functions from the explicitly typed based on the data type.

.. autosummary::
   :toctree: stubs

   retworkx.digraph_is_isomorphic
   retworkx.digraph_is_subgraph_isomorphic
   retworkx.digraph_vf2_mapping
   retworkx.digraph_distance_matrix
   retworkx.digraph_floyd_warshall
   retworkx.digraph_floyd_warshall_numpy
   retworkx.digraph_adjacency_matrix
   retworkx.digraph_all_simple_paths
   retworkx.digraph_astar_shortest_path
   retworkx.digraph_dijkstra_shortest_paths
   retworkx.digraph_all_pairs_dijkstra_shortest_paths
   retworkx.digraph_dijkstra_shortest_path_lengths
   retworkx.digraph_all_pairs_dijkstra_path_lengths
   retworkx.digraph_k_shortest_path_lengths
   retworkx.digraph_dfs_edges
   retworkx.digraph_find_cycle
   retworkx.digraph_transitivity
   retworkx.digraph_core_number
   retworkx.digraph_complement
   retworkx.digraph_union
   retworkx.digraph_random_layout
   retworkx.digraph_bipartite_layout
   retworkx.digraph_circular_layout
   retworkx.digraph_shell_layout
   retworkx.digraph_spiral_layout
   retworkx.digraph_spring_layout
   retworkx.digraph_num_shortest_paths_unweighted
   retworkx.digraph_betweenness_centrality
   retworkx.digraph_unweighted_average_shortest_path_length

.. _api-functions-pygraph:

API functions for PyGraph
=========================

These functions are algorithm functions that are type specific for
:class:`~retworkx.PyGraph` objects. Universal functions from Retworkx API that
work for both graph types internally call the functions from the explicitly
typed API based on the data type.

.. autosummary::
   :toctree: stubs

   retworkx.graph_is_isomorphic
   retworkx.graph_is_subgraph_isomorphic
   retworkx.graph_vf2_mapping
   retworkx.graph_distance_matrix
   retworkx.graph_floyd_warshall
   retworkx.graph_floyd_warshall_numpy
   retworkx.graph_adjacency_matrix
   retworkx.graph_all_simple_paths
   retworkx.graph_astar_shortest_path
   retworkx.graph_dijkstra_shortest_paths
   retworkx.graph_dijkstra_shortest_path_lengths
   retworkx.graph_all_pairs_dijkstra_shortest_paths
   retworkx.graph_k_shortest_path_lengths
   retworkx.graph_all_pairs_dijkstra_path_lengths
   retworkx.graph_dfs_edges
   retworkx.graph_transitivity
   retworkx.graph_core_number
   retworkx.graph_complement
   retworkx.graph_union
   retworkx.graph_random_layout
   retworkx.graph_bipartite_layout
   retworkx.graph_circular_layout
   retworkx.graph_shell_layout
   retworkx.graph_spiral_layout
   retworkx.graph_spring_layout
   retworkx.graph_num_shortest_paths_unweighted
   retworkx.graph_betweenness_centrality
   retworkx.graph_unweighted_average_shortest_path_length

Exceptions
==========

.. autosummary::
   :toctree: stubs

   retworkx.InvalidNode
   retworkx.DAGWouldCycle
   retworkx.NoEdgeBetweenNodes
   retworkx.DAGHasCycle
   retworkx.NoSuitableNeighbors
   retworkx.NoPathFound
   retworkx.NullGraph

Custom Return Types
===================

.. autosummary::
   :toctree: stubs

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
