*************
Release Notes
*************

.. release-notes::

0.7.1
=====

This release includes a fix for an oversight in the previous 0.7.0 and
0.6.0 releases. Those releases both added custom return types
:class:`~rustworkx.BFSSuccessors`, :class:`~rustworkx.NodeIndices`,
:class:`~rustworkx.EdgeList`, and :class:`~rustworkx.WeightedEdgeList` that
implemented the Python sequence protocol which were used in place of
lists for certain functions and methods. However, none of those classes
had support for being pickled, which was causing compatibility issues
for users that were using the return in a context where it would be
pickled (for example as an argument to or return of a function called
with multiprocessing). This release has a single change over 0.7.0 which
is to add the missing support for pickling :class:`~rustworkx.BFSSuccessors`,
:class:`~rustworkx.NodeIndices`, :class:`~rustworkx.EdgeList`, and
:class:`~rustworkx.WeightedEdgeList` which fixes that issue.

0.7.0
=====

This release includes several new features and bug fixes.

This release also dropped support for Python 3.5. If you want to use
rustworkx with Python 3.5 that last version which supports Python 3.5
is 0.6.0.

New Features
------------

- New generator functions for two new generator types, mesh and grid
  were added to :mod:`rustworkx.generators` for generating all to all and grid
  graphs respectively.  These functions are:
  :func:`~rustworkx.generators.mesh_graph`,
  :func:`~rustworkx.generators.directed_mesh_graph`,
  :func:`~rustworkx.generators.grid_graph`, and
  :func:`~rustworkx.generators.directed_grid_graph`
- A new function, :func:`rustworkx.digraph_union`, for taking the union between
  two :class:`~rustworkx.PyDiGraph` objects has been added.
- A new :class:`~rustworkx.PyDiGraph` method
  :meth:`~rustworkx.PyDiGraph.merge_nodes` has been added. This method can be
  used to merge 2 nodes in a graph if they have the same weight/data payload.
- A new :class:`~rustworkx.PyDiGraph` method
  :meth:`~rustworkx.PyDiGraph.find_node_by_weight()` which can be used to lookup
  a node index by a given weight/data payload.
- A new return type :class:`~rustworkx.NodeIndices` has been added. This class
  is returned by functions and methods that return a list of node indices. It
  implements the Python sequence protocol and can be used as list.
- Two new return types :class:`~rustworkx.EdgeList` and
  :class:`~rustworkx.WeightedEdgeList`. These classes are returned from functions
  and methods that return a list of edge tuples and a list of edge tuples with
  weights. They both implement the Python sequence protocol and can be used as
  a list
- A new function :func:`~rustworkx.collect_runs` has been added. This function is
  used to find linear paths of nodes that match a given condition.

Upgrade Notes
-------------

- Support for running rustworkx on Python 3.5 has been dropped. The last
  release with support for Python 3.5 is 0.6.0.
- The :meth:`rustworkx.PyDiGraph.node_indexes`,
  :meth:`rustworkx.PyDiGraph.neighbors`,
  :meth:`rustworkx.PyDiGraph.successor_indices`,
  :meth:`rustworkx.PyDiGraph.predecessor_indices`,
  :meth:`rustworkx.PyDiGraph.add_nodes_from`,
  :meth:`rustworkx.PyGraph.node_indexes`,
  :meth:`rustworkx.PyGraph.add_nodes_from`, and
  :meth:`rustworkx.PyGraph.neighbors` methods and the
  :func:`~rustworkx.dag_longest_path`, :func:`~rustworkx.topological_sort`,
  :func:`~rustworkx.graph_astar_shortest_path`, and
  :func:`~rustworkx.digraph_astar_shortest_path`  functions now return a
  :class:`~rustworkx.NodeIndices` object instead of a list of integers. This
  should not require any changes unless explicit type checking for a list was
  used.
- The :meth:`rustworkx.PyDiGraph.edge_list`, and
  :meth:`rustworkx.PyGraph.edge_list` methods and
  :func:`~rustworkx.digraph_dfs_edges`, :func:`~rustworkx.graph_dfs_edges`,
  and :func:`~rustworkx.digraph_find_cycle` functions now return an
  :class:`~rustworkx.EdgeList` object instead of a list of integers. This should
  not require any changes unless explicit type checking for a list was used.
- The :meth:`rustworkx.PyDiGraph.weighted_edge_list`,
  :meth:`rustworkx.PyDiGraph.in_edges`, :meth:`rustworkx.PyDiGraph.out_edges`,
  and `rustworkx.PyGraph.weighted_edge_list` methods now return a
  :class:`~rustworkx.WeightedEdgeList` object instead of a list of integers.
  This should not require any changes unless explicit type checking for a list
  was used.

Fixes
-----
- :class:`~rustworkx.BFSSuccessors` objects now can be compared with ``==`` and
  ``!=`` to any other Python sequence type.
- The built and published sdist packages for rustworkx were previously
  not including the Cargo.lock file. This meant that the reproducible
  build versions of the rust dependencies were not passed through to
  source. This has been fixed so building from sdist will always use
  known working versions that we use for testing in CI.


0.6.0
=====

This release includes a number of new features and bug fixes. The main focus of
this release was to expand the rustworkx API functionality to include some
commonly needed functions that were missing.

This release is also the first release to provide full support for running with
Python 3.9. On previous releases Python 3.9 would likely work, but it would
require building rustworkx from source. Also this will likely be the final
release that supports Python 3.5.

New Features
------------

- Two new functions, :func:`~rustworkx.digraph_k_shortest_path_lengths` and
  :func:`~rustworkx.graph_k_shortest_path_lengths`, for finding the k shortest
  path lengths from a node in a :class:`~rustworkx.PyDiGraph` and
  :class:`~rustworkx.PyGraph`.
- A new method, :meth:`~rustworkx.PyDiGraph.is_symmetric`, to the
  :class:`~rustworkx.PyDiGraph` class. This method will check whether the graph
  is symmetric or not
- A new kwarg, ``as_undirected``, was added to the
  :func:`~rustworkx.digraph_floyd_warshall_numpy()` function. This can be used
  to treat the input :class:`~rustworkx.PyDiGraph` object as if it was
  undirected for the generated output matrix.
- A new function, :func:`~rustworkx.digraph_find_cycle()`, which will return the
  first cycle during a depth first search of a :class:`~rustworkx.PyDiGraph`
  object.
- Two new functions, :func:`~rustworkx.directed_gnm_random_graph()` and
  :func:`~rustworkx.undirected_gnm_random_graph()`, for generating random
  :math:`G(n, m)` graphs.
- A new method, :meth:`~rustworkx.PyDiGraph.remove_edges_from`, was added to
  :class:`~rustworkx.PyDiGraph` and :class:`~rustworkx.PyGraph`
  (:meth:`~rustworkx.PyGraph.removed_edges_from`). This can be used to remove
  multiple edges from a graph object in a single call.
- A new method, :meth:`~rustworkx.PyDiGraph.subgraph`, was added to
  :class:`~rustworkx.PyDiGraph` and :class:`~rustworkx.PyGraph`
  (:meth:`~rustworkx.PyGraph.subgraph`) which takes in a list of node indices
  and will return a new object of the same type representing a subgraph
  containing the node indices in that list.
- Support for running with Python 3.9
- A new method, :meth:`~rustworkx.PyDiGraph.to_undirected`, was added to
  :class:`~rustworkx.PyDiGraph`. This method will generate an undirected
  :class:`~rustworkx.PyGraph` object from the :class:`~rustworkx.PyDiGraph`
  object.
- A new kwarg, ``bidirectional``, was added to the directed generator functions
  :func:`~rustworkx.generators.directed_cycle_graph`,
  :func:`~rustworkx.generators.directed_path_graph`, and
  :func:`~rustworkx.generators.directed_star_graph`. When set to ``True`` the
  directed graphs generated by these functions will add edges in both directions.
- Added two new functions, :func:`~rustworkx.is_weakly_connected()` and
  :func:`~rustworkx.weakly_connected_components`, which will either check if a
  :class:`~rustworkx.PyDiGraph` object is weakly connected or return the list of
  the weakly connected components of an input :class:`~rustworkx.PyDiGraph`.
- The ``weight_fn`` kwarg for :func:`~rustworkx.graph_adjacency_matrix`,
  :func:`~rustworkx.digraph_adjacency_matrix`,
  :func:`~rustworkx.graph_floyd_warshall_numpy`, and
  :func:`~rustworkx.digraph_floyd_warshall_numpy` is now optional. Previously,
  it always had to be specified when calling these function. But, instead you
  can now rely on a default weight float (which defaults to ``1.0``) to be used
  for all the edges in the graph.
- Add a :meth:`~rustworkx.PyGraph.neighbors` method to
  :class:`~rustworkx.PyGraph` and :class:`~rustworkx.PyDiGraph`
  (:meth:`~rustworkx.PyDiGraph.neighbors`). This function will return the node
  indices of the neighbor nodes for a given input node.
- Two new methods, :meth:`~rustworkx.PyDiGraph.successor_indices` and
  :meth:`~rustworkx.PyDiGraph.predecessor_indices`, were added to
  :class:`~rustworkx.PyDiGraph`. These methods will return the node indices for
  the successor and predecessor nodes of a given input node.
- Two new functions, :func:`~rustworkx.graph_distance_matrix` and
  :func:`~rustworkx.digraph_distance_matrix`, were added for generating a
  distance matrix from an input :class:`~rustworkx.PyGraph` and
  :class:`~rustworkx.PyDiGraph`.
- Two new functions, :func:`~rustworkx.digraph_dijkstra_shortest_paths` and
  :func:`~rustworkx.graph_dijkstra_shortest_path`, were added for returning the
  shortest paths from a node in a :class:`~rustworkx.PyDiGraph` and a
  :class:`~rustworkx.PyGraph` object.
- Four new methods, :meth:`~rustworkx.PyDiGraph.insert_node_on_in_edges`,
  :meth:`~rustworkx.PyDiGraph.insert_node_on_out_edges`,
  :meth:`~rustworkx.PyDiGraph.insert_node_on_in_edges_multiple`, and
  :meth:`~rustworkx.PyDiGraph.insert_node_on_out_edges_multiple` were added to
  :class:`~rustworkx.PyDiGraph`. These functions are used to insert an existing
  node in between an reference node(s) and all it's predecessors or successors.
- Two new functions, :func:`~rustworkx.graph_dfs_edges` and
  :func:`~rustworkx.digraph_dfs_edges`, were added to get an edge list in depth
  first order from a :class:`~rustworkx.PyGraph` and
  :class:`~rustworkx.PyDiGraph`.

Upgrade Notes
-------------

- The numpy arrays returned by :func:`~rustworkx.graph_floyd_warshall_numpy`,
  :func:`~rustworkx.digraph_floyd_warshall_numpy`,
  :func:`~rustworkx.digraph_adjacency_matrix`, and
  :func:`~rustworkx.graph_adjacency_matrix` will now be in a contiguous C array
  memory layout. Previously, they would return arrays in a column-major fortran
  layout. This was change was made to make it easier to interface the arrays
  returned by these functions with other C Python extensions. There should be
  no change when interacting with the numpy arrays via numpy's API.
- The :func:`~rustworkx.bfs_successors` method now returns an object of a custom
  type :class:`~rustworkx.BFSSuccessors` instead of a list. The
  :class:`~rustworkx.BFSSuccessors` type implements the Python sequence protocol
  so it can be used in place like a list (except for where explicit type checking
  is used). This was done to defer the type conversion between Rust and Python
  since doing it all at once can be a performance bottleneck especially for
  large graphs. The :class:`~rustworkx.BFSSuccessors` class will only do the type
  conversion when an element is accessed.

Fixes
-----
- When pickling :class:`~rustworkx.PyDiGraph` objects the original node indices
  will be preserved across the pickle.
- The random :math:`G(n, p)` functions,
  :func:`~rustworkx.directed_gnp_random_graph` and
  :func:`~rustworkx.undirected_gnp_random_graph`, will now also handle exact 0 or
  1 probabilities. Previously it would fail in these cases. Fixes
  `#172 <https://github.com/Qiskit/rustworkx/issues/172>`__


0.5.0
=====

This release include a number of new features and bug fixes. The main
focus of the improvements of this release was to increase the ease of
interacting with graph objects. This includes adding support for generating dot
output which can be used with graphviz (or similar tools) for visualizing
graphs adding more methods to query the state of graph, adding a generator
module for easily creating graphs of certain shape, and implementing the
mapping protocol so you can directly interact with graph objects.

New Features
------------

- A new method, :meth:`~rustworkx.PyGraph.to_dot`, was added to
  :class:`~rustworkx.PyGraph` and :class:`~rustworkx.PyDiGraph`
  (:meth:`~rustworkx.PyDiGraph.to_dot`). It will generate a
  `dot format <https://graphviz.org/doc/info/lang.html>`__ representation of
  the object which can be used with `Graphivz <https://graphviz.org/>`__ (or
  similar tooling) to generate visualizations of graphs.
- Added a new function, :func:`~rustworkx.strongly_connected_components`, to get
  the list of strongly connected components of a :class:`~rustworkx.PyDiGraph`
  object.
- A new method, :meth:`~rustworkx.PyGraph.compose`, for composing another graph
  object of the same type into a graph was added to :class:`~rustworkx.PyGraph`
  and :class:`~rustworkx.PyDiGraph` (:meth:`~rustworkx.PyDiGraph.compose`).
- The :class:`~rustworkx.PyGraph` and :class:`~rustworkx.PyDigraph` classes now
  implement the Python mapping protocol for interacting with graph nodes. You
  can now access and interact with node data directly by using standard map
  access patterns in Python. For example, accessing a graph like ``graph[1]``
  will return the weight/data payload for the node at index 1.
- A new module, :mod:`rustworkx.generators`, has been added. Functions in this
  module can be used for quickly generating graphs of certain shape. To start
  it includes:

  - :func:`rustworkx.generators.cycle_graph`
  - :func:`rustworkx.generators.directed_cycle_graph`
  - :func:`rustworkx.generators.path_graph`
  - :func:`rustworkx.generators.directed_path_graph`
  - :func:`rustworkx.generators.star_graph`
  - :func:`rustworkx.generators.directed_star_graph`

- A new method, :meth:`~rustworkx.PyDiGraph.remove_node_retain_edges`, has been
  added to the :class:`~rustworkx.PyDiGraph` class. This method can be used to
  remove a node and add edges from its predecesors to its successors.
- Two new methods, :meth:`~rustworkx.PyGraph.edge_list` and
  :meth:`~rustworkx.PyGraph.weighted_edge_list`, for getting a list of tuples
  with the edge source and target (with or without edge weights) have been
  added to :class:`~rustworkx.PyGraph` and :class:`~rustworkx.PyDiGraph`
  (:meth:`~rustworkx.PyDiGraph.edge_list` and
  :meth:`~rustworkx.PyDiGraph.weighted_edge_list`)
- A new function, :func:`~rustworkx.cycle_basis`, for getting a list of cycles
  which form a basis for cycles of a :class:`~rustworkx.PyGraph` object.
- Two new functions, :func:`~rustworkx.graph_floyd_warshall_numpy` and
  :func:`~rustworkx.digraph_floyd_warshall_numpy`, were added for running the
  Floyd Warshall algorithm and returning all the shortest path lengths as a
  distance matrix.
- A new constructor method, :meth:`~rustworkx.PyGraph.read_edge_list`, has been
  added to :class:`~rustworkx.PyGraph` and :class:`~rustworkx.PyDigraph`
  (:meth:`~rustworkx.read_edge_list`). This method will take in a path to an
  edge list file and will read that file and generate a new object from the
  contents.
- Two new methods, :meth:`~rustworkx.PyGraph.extend_from_edge_list` and
  :meth:`~rustworkx.PyGraoh.extend_from_weighted_edge_list` has been added
  to :class:`~rustworkx.PyGraph` and :class:`~rustworkx.PyDiGraph`
  (:meth:`~rustworkx.PyDiGraph.extend_from_edge_list` and
  :meth:`~rustworkx.PyDiGraph.extend_from_weighted_edge_list`). This method
  takes in an edge list and will add both the edges and nodes (if a node index
  used doesn't exist yet) in the list to the graph.

Fixes
-----

- The limitation with the :func:`~rustworkx.is_isomorphic` and
  :func:`~rustworkx.is_isomorphic_node_match` functions that would cause
  segfaults when comparing graphs with node removals has been fixed. You can
  now run either function with any
  :class:`~rustworkx.PyDiGraph`/:class:`~rustworkx.PyDAG` objects, even if there
  are node removals. Fixes
  `#27 <https://github.com/Qiskit/rustworkx/issues/27>`__
- If an invalid node index was passed as part of the ``first_layer``
  argument to the :func:`~rustworkx.layers` function it would previously raise
  a ``PanicException`` that included a Rust backtrace and no other user
  actionable details which was caused by an unhandled error. This has been
  fixed so that an ``IndexError`` is raised and the problematic node index
  is included in the exception message.

0.4.0
=====

This release includes many new features and fixes, including improved
performance and better documentation. But, the biggest change for this
release is that this is the first release of rustworkx that supports
compilation with a stable released version of rust. This was made
possible thanks to all the hard work of the PyO3 maintainers and
contributors in the PyO3 0.11.0 release.

New Features
------------

- A new class for undirected graphs, :class:`~rustworkx.PyGraph`, was added.
- 2 new functions :func:`~rustworkx.graph_adjacency_matrix` and
  :func:`~rustworkx.digraph_adjacency_matrix` to get the adjacency matrix of a
  :class:`~rustworkx.PyGraph` and :class:`~rustworkx.PyDiGraph` object.
- A new :class:`~rustworkx.PyDiGraph` method,
  :meth:`~rustworkx.PyDiGraph.find_adjacent_node_by_edge`, was added. This is
  used to locate an adjacent node given a condition based on the edge between them.
- New methods, :meth:`~rustworkx.PyDiGraph.add_nodes_from`,
  :meth:`~rustworkx.PyDiGraph.add_edges_from`,
  :meth:`~rustworkx.PyDiGraph.add_edges_from_no_data`, and
  :meth:`~rustworkx.PyDiGraph.remove_nodes_from` were added to
  :class:`~rustworkx.PyDiGraph`. These methods allow for the addition (and
  removal) of multiple nodes or edges from a graph in a single call.
- A new function, :func:`~rustworkx.graph_greedy_color`, which is used to
  return a coloring map from a :class:`~rustworkx.PyGraph` object.
- 2 new functions, :func:`~rustworkx.graph_astar_shortest_path` and
  :func:`~rustworkx.digraph_astar_shortest_path`, to find the shortest path
  from a node to a specified goal using the A* search algorithm.
- 2 new functions, :func:`~rustworkx.graph_all_simple_paths` and
  :func:`~rustworkx.digraph_all_simple_paths`, to return a list of all the
  simple paths between 2 nodes in a :class:`~rustworkx.PyGraph` or a
  :class:`~rustworkx.PyDiGraph` object.
- 2 new functions, :func:`~rustworkx.directed_gnp_random_graph` and
  :func:`~rustworkx.undirected_gnp_random_graph`, to generate :math:`G_{np}`
  random :class:`~rustworkx.PyDiGraph` and :class:`~rustworkx.PyGraph` objects.
- 2 new functions, :func:`~rustworkx.graph_dijkstra_shortest_path_lengths` and
  :func:`~rustworkx.digraph_dijkstra_shortest_path_lengths`, were added for find
  the shortest path length between nodes in :class:`~rustworkx.PyGraph` or
  :class:`~rustworkx.PyDiGraph` object using Dijkstra's algorithm.

Upgrade Notes
-------------

- The :class:`~rustworkx.PyDAG` class was renamed :class:`~rustworkx.PyDiGraph`
  to better reflect it's functionality. For backwards compatibility
  :class:`~rustworkx.PyDAG` still exists as a Python subclass of
  :class:`~rustworkx.PyDiGraph`. No changes should be required for existing
  users.
- `numpy <https://numpy.org/>`__ is now a dependency of rustworkx. This is used
  for the adjacency matrix functions to return numpy arrays. The minimum
  version of numpy supported is 1.16.0.

Fixes
-----

- The rustworkx exception classes are now properly exported from the
  rustworkx module. In prior releases it was not possible to import the
  exception classes (normally to catch one being raised) requiring users
  to catch the base Exception class. This has been fixed so a
  specialized rustworkx exception class can be used.
