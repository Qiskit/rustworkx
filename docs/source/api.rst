.. _retworkx:

======================
Retworkx API Reference
======================

Graph Classes
-------------

.. autosummary::
   :toctree: stubs

    retworkx.PyGraph
    retworkx.PyDiGraph
    retworkx.PyDAG

Generators
----------

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

Random Circuit Functions
------------------------

.. autosummary::
   :toctree: stubs

    retworkx.directed_gnp_random_graph
    retworkx.undirected_gnp_random_graph
    retworkx.directed_gnm_random_graph
    retworkx.undirected_gnm_random_graph

Algorithm Functions
-------------------

Specific Graph Type Methods
'''''''''''''''''''''''''''

.. autosummary::
   :toctree: stubs

   retworkx.bfs_successors
   retworkx.dag_longest_path
   retworkx.dag_longest_path_length
   retworkx.number_weakly_connected_components
   retworkx.weakly_connected_components
   retworkx.is_weakly_connected
   retworkx.is_directed_acyclic_graph
   retworkx.digraph_is_isomorphic
   retworkx.graph_is_isomorphic
   retworkx.topological_sort
   retworkx.descendants
   retworkx.ancestors
   retworkx.lexicographical_topological_sort
   retworkx.graph_distance_matrix
   retworkx.digraph_distance_matrix
   retworkx.floyd_warshall
   retworkx.graph_floyd_warshall_numpy
   retworkx.digraph_floyd_warshall_numpy
   retworkx.collect_runs
   retworkx.layers
   retworkx.digraph_adjacency_matrix
   retworkx.graph_adjacency_matrix
   retworkx.graph_all_simple_paths
   retworkx.digraph_all_simple_paths
   retworkx.graph_astar_shortest_path
   retworkx.digraph_astar_shortest_path
   retworkx.graph_dijkstra_shortest_paths
   retworkx.digraph_dijkstra_shortest_paths
   retworkx.graph_dijkstra_shortest_path_lengths
   retworkx.digraph_dijkstra_shortest_path_lengths
   retworkx.graph_k_shortest_path_lengths
   retworkx.digraph_k_shortest_path_lengths
   retworkx.graph_greedy_color
   retworkx.cycle_basis
   retworkx.strongly_connected_components
   retworkx.graph_dfs_edges
   retworkx.digraph_dfs_edges
   retworkx.digraph_find_cycle
   retworkx.digraph_union
   retworkx.is_matching
   retworkx.is_maximal_matching
   retworkx.max_weight_matching
   retworkx.graph_transitivity
   retworkx.digraph_transitivity
   retworkx.graph_core_number
   retworkx.digraph_core_number

.. _universal-functions:

Universal Functions
'''''''''''''''''''

These functions are algorithm functions that wrap per graph object
type functions in the algorithms API but can be run with a
:class:`~retworkx.PyGraph`, :class:`~retworkx.PyDiGraph`, or
:class:`~retworkx.PyDAG` object.

.. autosummary::
   :toctree: stubs

   retworkx.distance_matrix
   retworkx.floyd_warshall_numpy
   retworkx.adjacency_matrix
   retworkx.all_simple_paths
   retworkx.astar_shortest_path
   retworkx.dijkstra_shortest_paths
   retworkx.dijkstra_shortest_path_lengths
   retworkx.k_shortest_path_lengths
   retworkx.dfs_edges
   retworkx.is_isomorphic
   retworkx.is_isomorphic_node_match
   retworkx.transitivity
   retworkx.core_number

Converters
----------

.. autosummary::
   :toctree: stubs

   retworkx.networkx_converter

Exceptions
----------

.. autosummary::
   :toctree: stubs

   retworkx.InvalidNode
   retworkx.DAGWouldCycle
   retworkx.NoEdgeBetweenNodes
   retworkx.DAGHasCycle
   retworkx.NoSuitableNeighbors
   retworkx.NoPathFound
   retworkx.NullGraph

Return Iterator Types
---------------------

.. autosummary::
   :toctree: stubs

   retworkx.BFSSuccessors
   retworkx.NodeIndices
   retworkx.EdgeList
   retworkx.WeightedEdgeList
