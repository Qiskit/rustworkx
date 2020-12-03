.. release-notes:: Release Notes

0.7.1
=====

This release includes a fix for an oversight in the previous 0.7.0 and
0.6.0 releases. Those releases both added custom return types
:class:`~retworkx.BFSSuccessors`, :class:`~retworkx.NodeIndices`,
:class:`~retworkx.EdgeList`, and :class:`~retworkx.WeightedEdgeList` that
implemented the Python sequence protocol which were used in place of
lists for certain functions and methods. However, none of those classes
had support for being pickled, which was causing compatibility issues
for users that were using the return in a context where it would be
pickled (for example as an argument to or return of a function called
with multiprocessing). This release has a single change over 0.7.0 which
is to add the missing support for pickling :class:`~retworkx.BFSSuccessors`,
:class:`~retworkx.NodeIndices`, :class:`~retworkx.EdgeList`, and
:class:`~retworkx.WeightedEdgeList` which fixes that issue.

0.7.0
=====

This release includes several new features and bug fixes.

This release also dropped support for Python 3.5. If you want to use
retworkx with Python 3.5 that last version which supports Python 3.5
is 0.6.0.

Added
-----
- New generator functions for two new generator types, mesh and grid
  were added to :mod:`retworkx.generators` for generating all to all and grid
  graphs respectively.  These functions are:
  :func:`~retworkx.generators.mesh_graph`,
  :func:`~retworkx.generators.directed_mesh_graph`,
  :func:`~retworkx.generators.grid_graph`, and
  :func:`~retworkx.generators.directed_grid_graph`
- A new function, :func:`retworkx.digraph_union`, for taking the union between
  two :class:`~retworkx.PyDiGraph` objects has been added.
- A new :class:`~retworkx.PyDiGraph` method
  :meth:`~retworkx.PyDiGraph.merge_nodes` has been added. This method can be
  used to merge 2 nodes in a graph if they have the same weight/data payload.
- A new :class:`~retworkx.PyDiGraph` method
  :meth:`~retworkx.PyDiGraph.find_node_by_weight()` which can be used to lookup
  a node index by a given weight/data payload.
- A new return type :class:`~retworkx.NodeIndices` has been added. This class
  is returned by functions and methods that return a list of node indices. It
  implements the Python sequence protocol and can be used as list.
- Two new return types :class:`~retworkx.EdgeList` and
  :class:`~retworkx.WeightedEdgeList`. These classes are returned from functions
  and methods that return a list of edge tuples and a list of edge tuples with
  weights. They both implement the Python sequence protocol and can be used as
  a list
- A new function :func:`~retworkx.collect_runs` has been added. This function is
  used to find linear paths of nodes that match a given condition.

Upgrade
-------
- Support for running retworkx on Python 3.5 has been dropped. The last
  release with support for Python 3.5 is 0.6.0.
- The :meth:`retworkx.PyDiGraph.node_indexes`,
  :meth:`retworkx.PyDiGraph.neighbors`,
  :meth:`retworkx.PyDiGraph.successor_indices`,
  :meth:`retworkx.PyDiGraph.predecessor_indices`,
  :meth:`retworkx.PyDiGraph.add_nodes_from`,
  :meth:`retworkx.PyGraph.node_indexes`,
  :meth:`retworkx.PyGraph.add_nodes_from`, and
  :meth:`retworkx.PyGraph.neighbors` methods and the
  :func:`~retworkx.dag_longest_path`, :func:`~retworkx.topological_sort`,
  :func:`~retworkx.graph_astar_shortest_path`, and
  :func:`~retworkx.digraph_astar_shortest_path`  functions now return a
  :class:`~retworkx.NodeIndices` object instead of a list of integers. This
  should not require any changes unless explicit type checking for a list was
  used.
- The :meth:`retworkx.PyDiGraph.edge_list`, and
  :meth:`retworkx.PyGraph.edge_list` methods and
  :func:`~retworkx.digraph_dfs_edges`, :func:`~retworkx.graph_dfs_edges`,
  and :func:`~retworkx.digraph_find_cycle` functions now return an
  :class:`~retworkx.EdgeList` object instead of a list of integers. This should
  not require any changes unless explicit type checking for a list was used.
- The :meth:`retworkx.PyDiGraph.weighted_edge_list`,
  :meth:`retworkx.PyDiGraph.in_edges`, :meth:`retworkx.PyDiGraph.out_edges`,
  and `retworkx.PyGraph.weighted_edge_list` methods now return a
  :class:`~retworkx.WeightedEdgeList` object instead of a list of integers.
  This should not require any changes unless explicit type checking for a list
  was used.

Fixes
-----
- :class:`~retworkx.BFSSuccessors` objects now can be compared with ``==`` and
  ``!=`` to any other Python sequence type.
- The built and published sdist packages for retworkx were previously
  not including the Cargo.lock file. This meant that the reproducible
  build versions of the rust dependencies were not passed through to
  source. This has been fixed so building from sdist will always use
  known working versions that we use for testing in CI.

0.6.0
=====

This release includes a number of new features and bug fixes. The main focus of
this release was to expand the retworkx API functionality to include some
commonly needed functions that were missing.

This release is also the first release to provide full support for running with
Python 3.9. On previous releases Python 3.9 would likely work, but it would
require building retworkx from source. Also this will likely be the final
release that supports Python 3.5.

Added
-----
- Two new functions, :func:`~retworkx.digraph_k_shortest_path_lengths` and
  :func:`~retworkx.graph_k_shortest_path_lengths`, for finding the k shortest
  path lengths from a node in a :class:`~retworkx.PyDiGraph` and
  :class:`~retworkx.PyGraph`.
- A new method, :meth:`~retworkx.PyDiGraph.is_symmetric`, to the
  :class:`~retworkx.PyDiGraph` class. This method will check whether the graph
  is symmetric or not
- A new kwarg, ``as_undirected``, was added to the
  :func:`~retworkx.digraph_floyd_warshall_numpy()` function. This can be used
  to treat the input :class:`~retworkx.PyDiGraph` object as if it was
  undirected for the generated output matrix.
- A new function, :func:`~retworkx.digraph_find_cycle()`, which will return the
  first cycle during a depth first search of a :class:`~retworkx.PyDiGraph`
  object.
- Two new functions, :func:`~retworkx.directed_gnm_random_graph()` and
  :func:`~retworkx.undirected_gnm_random_graph()`, for generating random
  :math:`G(n, m)` graphs.
- A new method, :meth:`~retworkx.PyDiGraph.remove_edges_from`, was added to
  :class:`~retworkx.PyDiGraph` and :class:`~retworkx.PyGraph`
  (:meth:`~retworkx.PyGraph.removed_edges_from`). This can be used to remove
  multiple edges from a graph object in a single call.
- A new method, :meth:`~retworkx.PyDiGraph.subgraph`, was added to
  :class:`~retworkx.PyDiGraph` and :class:`~retworkx.PyGraph`
  (:meth:`~retworkx.PyGraph.subgraph`) which takes in a list of node indices
  and will return a new object of the same type representing a subgraph
  containing the node indices in that list.
- Support for running with Python 3.9
- A new method, :meth:`~retworkx.PyDiGraph.to_undirected`, was added to
  :class:`~retworkx.PyDiGraph`. This method will generate an undirected
  :class:`~retworkx.PyGraph` object from the :class:`~retworkx.PyDiGraph`
  object.
- A new kwarg, ``bidirectional``, was added to the directed generator functions
  :func:`~retworkx.generators.directed_cycle_graph`,
  :func:`~retworkx.generators.directed_path_graph`, and
  :func:`~retworkx.generators.directed_star_graph`. When set to ``True`` the
  directed graphs generated by these functions will add edges in both directions.
- Added two new functions, :func:`~retworkx.is_weakly_connected()` and
  :func:`~retworkx.weakly_connected_components`, which will either check if a
  :class:`~retworkx.PyDiGraph` object is weakly connected or return the list of
  the weakly connected components of an input :class:`~retworkx.PyDiGraph`.
- The ``weight_fn`` kwarg for :func:`~retworkx.graph_adjacency_matrix`,
  :func:`~retworkx.digraph_adjacency_matrix`,
  :func:`~retworkx.graph_floyd_warshall_numpy`, and
  :func:`~retworkx.digraph_floyd_warshall_numpy` is now optional. Previously,
  it always had to be specified when calling these function. But, instead you
  can now rely on a default weight float (which defaults to ``1.0``) to be used
  for all the edges in the graph.
- Add a :meth:`~retworkx.PyGraph.neighbors` method to
  :class:`~retworkx.PyGraph` and :class:`~retworkx.PyDiGraph`
  (:meth:`~retworkx.PyDiGraph.neighbors`). This function will return the node
  indices of the neighbor nodes for a given input node.
- Two new methods, :meth:`~retworkx.PyDiGraph.successor_indices` and
  :meth:`~retworkx.PyDiGraph.predecessor_indices`, were added to
  :class:`~retworkx.PyDiGraph`. These methods will return the node indices for
  the successor and predecessor nodes of a given input node.
- Two new functions, :func:`~retworkx.graph_distance_matrix` and
  :func:`~retworkx.digraph_distance_matrix`, were added for generating a
  distance matrix from an input :class:`~retworkx.PyGraph` and
  :class:`~retworkx.PyDiGraph`.
- Two new functions, :func:`~retworkx.digraph_dijkstra_shortest_paths` and
  :func:`~retworkx.graph_dijkstra_shortest_path`, were added for returning the
  shortest paths from a node in a :class:`~retworkx.PyDiGraph` and a
  :class:`~retworkx.PyGraph` object.
- Four new methods, :meth:`~retworkx.PyDiGraph.insert_node_on_in_edges`,
  :meth:`~retworkx.PyDiGraph.insert_node_on_out_edges`,
  :meth:`~retworkx.PyDiGraph.insert_node_on_in_edges_multiple`, and
  :meth:`~retworkx.PyDiGraph.insert_node_on_out_edges_multiple` were added to
  :class:`~retworkx.PyDiGraph`. These functions are used to insert an existing
  node in between an reference node(s) and all it's predecessors or successors.
- Two new functions, :func:`~retworkx.graph_dfs_edges` and
  :func:`~retworkx.digraph_dfs_edges`, were added to get an edge list in depth
  first order from a :class:`~retworkx.PyGraph` and
  :class:`~retworkx.PyDiGraph`.

Upgrade
-------
- The numpy arrays returned by :func:`~retworkx.graph_floyd_warshall_numpy`,
  :func:`~retworkx.digraph_floyd_warshall_numpy`,
  :func:`~retworkx.digraph_adjacency_matrix`, and
  :func:`~retworkx.graph_adjacency_matrix` will now be in a contiguous C array
  memory layout. Previously, they would return arrays in a column-major fortran
  layout. This was change was made to make it easier to interface the arrays
  returned by these functions with other C Python extensions. There should be
  no change when interacting with the numpy arrays via numpy's API.
- The :func:`~retworkx.bfs_successors` method now returns an object of a custom
  type :class:`~retworkx.BFSSuccessors` instead of a list. The
  :class:`~retworkx.BFSSuccessors` type implements the Python sequence protocol
  so it can be used in place like a list (except for where explicit type checking
  is used). This was done to defer the type conversion between Rust and Python
  since doing it all at once can be a performance bottleneck especially for
  large graphs. The :class:`~retworkx.BFSSuccessors` class will only do the type
  conversion when an element is accessed.

Fixes
-----
- When pickling :class:`~retworkx.PyDiGraph` objects the original node indices
  will be preserved across the pickle.
- The random :math:`G(n, p)` functions,
  :func:`~retworkx.directed_gnp_random_graph` and
  :func:`~retworkx.undirected_gnp_random_graph`, will now also handle exact 0 or
  1 probabilities. Previously it would fail in these cases. Fixes
  `#172 <https://github.com/Qiskit/retworkx/issues/172>`__


0.5.0
=====

This release include a number of new features and bug fixes. The main
focus of the improvements of this release was to increase the ease of
interacting with graph objects. This includes adding support for generating dot
output which can be used with graphviz (or similar tools) for visualizing
graphs adding more methods to query the state of graph, adding a generator
module for easily creating graphs of certain shape, and implementing the
mapping protocol so you can directly interact with graph objects.

Added
-----
- A new method, `to_dot()`, was added to `PyGraph` and `PyDiGraph`. It will
  generate a dot format representation of the object which can be used
  with Graphivz (or similar tooling) to generate visualizations of the
  graph.
- Added a new function, `strongly_connected_components()`, to get the list
  of strongly connected components of a `PyDiGraph` object.
- A new method, `compose()`, for combing another graph object of the same
  type was added to `PyGraph` and `PyDiGraph`.
- The `PyGraph` and `PyDigraph` classes now implement the Python mapping
  protocol for interacting with graph nodes. You can now access and
  interact with node data directly by using standard map access patterns
  in python. For example, running `graph[1]` will return the data for the
  index at node 1.
- A new module, `retworkx.generators`, has been added. Functions in this
  module can be used for quickly generating graphs of certain shape.
- A new method, `remove_node_retain_edges()`, has been added to the
  `PyDiGraph` class. This method can be used to remove a node and add
  edges from its predecesors to its successors.
- Two new methods, `edge_list()` and `weighted_edge_list()`, for getting a
  list of tuples with the edge source and target (with or without edge
  weights) have been added to `PyGraph` and `PyDigraph`.
- A new function, `cycle_basis()`, for getting a list of cycles which form
  a basis for cycles of a `PyGraph` object.
- Two new functions, `graph_floyd_warshall_numpy()` and
  `digraph_floyd_warshall_numpy()`, were added for running the Floyd Warshall
  algorithm and returning all the shortest path lengths as a distance
  matrix.
- A new constructor method, `read_edge_list()`, has been added to `PyGraph` and
  `PyDigraph`. This method will take in a path to an edge list file and
  will read that file and generate a new object from the contents.
- A new method, `extend_from_edge_list()`, has been added to `PyGraph` and
  `PyDiGraph`. This method takes in an edge list and will add both the edges and
  nodes (if a node index used doesn't exist yet) in the list to the graph.

Fixes
-----

- The limitation with `is_isomorphic()` and `is_isomorphic_node_match()` functions
  that would cause segfaults when comparing graphs with node removals
  has been fixed. You can now run either function with any
  `PyDiGraph`/`PyDAG` objects, even if there are node removals. Fixes #27
- If an invalid node index was passed as part of the `first_layer`
  argument to the `layers()` function

0.4.0
=====

This release includes many new features and fixes, including improved
performance and better documentation. But, the biggest change for this
release is that this is the first release of retworkx that supports
compilation with a stable released version of rust. This was made
possible thanks to all the hard work of the PyO3 maintainers and
contributors in the PyO3 0.11.0 release.

Added
-----

- A new class for undirected graphs, `PyGraph`, was added.
- 2 new functions `graph_adjacency_matrix()` and
  `digraph_adjacency_matrix()` to get the adjacency matrix of a `PyGraph`
  and` PyDiGraph` object.
- A new `PyDiGraph` method, `find_adjacent_node_by_edge()`, was added. This is
  used to locate an adjacent node given a condition based on the edge between them.
- New methods, `add_nodes_from()`, `add_edges_from()`,
  `add_edges_from_no_data()`, and `remove_nodes_from()` were added to `PyDiGraph`
  (and `PyGraph`). These methods allow for the addition (and removal) of
  multiple nodes or edges from a graph in a single call.
- A new function, `graph_greedy_coloring()`, which is used to return a
  coloring map from a `PyGraph` object.
- 2 new functions, `graph_astar_shortest_path()` and `digraph_astar_shortest_path()`,
  to find the shortest path from a node to a specified goal using the A*
  search algorithm.
- 2 new functions, `graph_all_simple_paths()` and
  `digraph_all_simple_paths()`, to return a list of all the simple paths
  between 2 nodes in a `PyGraph` or `PyDiGraph` object.
- 2 new functions, `directed_gnp_random_graph()` and
  `undirected_gnp_random_graph()`, to generate Gnp random `PyDiGraph` and
  `PyGraph` objects.
- 2 new functions, `graph_dijkstra_shortest_path_lengths()` and
  `digraph_dijkstra_shortest_path_lengths()`, were added for find the shortest paths
  between nodes in `PyGraph` or `PyDiGraph` object using Dijkstra's
  algorithm.

Changed
-------

- The `PyDAG` class was renamed `PyDiGraph` to better reflect it's
  functionality. For backwards compatibility `PyDAG` still exists as a python
  subclass of `PyDiGraph`. No changes should be required for existing
  users.
- numpy is now a dependency of retworkx. This is used for the adjacency
  matrix functions to return numpy arrays. The minimum version of numpy
  supported is 1.16.0.

Fixes
-----

- The retworkx exception classes are now properly exported from the
  retworkx module. In prior releases it was not possible to import the
  exception classes (normally to catch one being raised) requiring users
  to catch the base Exception class. This has been fixed so a
  specialized retworkx exception class can be used.
