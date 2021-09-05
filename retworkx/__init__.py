# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import sys
import functools

from .retworkx import *

sys.modules["retworkx.generators"] = generators


class PyDAG(PyDiGraph):
    """A class for creating direct acyclic graphs.

    PyDAG is just an alias of the PyDiGraph class and behaves identically to
    the :class:`~retworkx.PyDiGraph` class and can be used interchangably
    with ``PyDiGraph``. It currently exists solely as a backwards
    compatibility alias for users of retworkx from prior to the
    0.4.0 release when there was no PyDiGraph class.

    The PyDAG class is used to create a directed graph. It can be a
    multigraph (have multiple edges between nodes). Each node and edge
    (although rarely used for edges) is indexed by an integer id. These ids
    are stable for the lifetime of the graph object and on node or edge
    deletions you can have holes in the list of indices for the graph.
    Node indices will be reused on additions after removal. For example:

    .. jupyter-execute::

        import retworkx

        graph = retworkx.PyDAG()
        graph.add_nodes_from(list(range(5)))
        graph.add_nodes_from(list(range(2)))
        graph.remove_node(2)
        print("After deletion:", graph.node_indexes())
        res_manual = graph.add_parent(6, None, None)
        print("After adding a new node:", graph.node_indexes())

    Additionally, each node and edge contains an arbitrary Python object as a
    weight/data payload.

    You can use the index for access to the data payload as in the
    following example:

    .. jupyter-execute::

        import retworkx

        graph = retworkx.PyDAG()
        data_payload = "An arbitrary Python object"
        node_index = graph.add_node(data_payload)
        print("Node Index: %s" % node_index)
        print(graph[node_index])

    The PyDAG class implements the Python mapping protocol for nodes so in
    addition to access you can also update the data payload with:

    .. jupyter-execute::

        import retworkx

        graph = retworkx.PyDAG()
        data_payload = "An arbitrary Python object"
        node_index = graph.add_node(data_payload)
        graph[node_index] = "New Payload"
        print("Node Index: %s" % node_index)
        print(graph[node_index])

    The PyDAG class has an option for real time cycle checking which can
    be used to ensure any edges added to the graph does not introduce a cycle.
    By default the real time cycle checking feature is disabled for
    performance, however you can enable it by setting the ``check_cycle``
    attribute to True. For example::

        import retworkx
        dag = retworkx.PyDAG()
        dag.check_cycle = True

    or at object creation::

        import retworkx
        dag = retworkx.PyDAG(check_cycle=True)

    With check_cycle set to true any calls to :meth:`PyDAG.add_edge` will
    ensure that no cycles are added, ensuring that the PyDAG class truly
    represents a directed acyclic graph. Do note that this cycle checking on
    :meth:`~PyDAG.add_edge`, :meth:`~PyDigraph.add_edges_from`,
    :meth:`~PyDAG.add_edges_from_no_data`,
    :meth:`~PyDAG.extend_from_edge_list`,  and
    :meth:`~PyDAG.extend_from_weighted_edge_list` comes with a performance
    penalty that grows as the graph does.  If you're adding a node and edge at
    the same time, leveraging :meth:`PyDAG.add_child` or
    :meth:`PyDAG.add_parent` will avoid this overhead.

    By default a ``PyDAG`` is a multigraph (meaning there can be parallel
    edges between nodes) however this can be disabled by setting the
    ``multigraph`` kwarg to ``False`` when calling the ``PyDAG`` constructor.
    For example::

        import retworkx
        dag = retworkx.PyDAG(multigraph=False)

    This can only be set at ``PyDiGraph`` initialization and not adjusted after
    creation. When :attr:`~retworkx.PyDiGraph.multigraph` is set to ``False``
    if a method call is made that would add a parallel edge it will instead
    update the existing edge's weight/data payload.

    :param bool check_cycle: When this is set to ``True`` the created
        ``PyDAG`` has runtime cycle detection enabled.
    :param bool multgraph: When this is set to ``False`` the created
        ``PyDAG`` object will not be a multigraph. When ``False`` if a method
        call is made that would add parallel edges the the weight/weight from
        that method call will be used to update the existing edge in place.
    """

    pass


@functools.singledispatch
def distance_matrix(
    graph, parallel_threshold=300, as_undirected=False, null_value=0.0
):
    """Get the distance matrix for a graph

    This differs from functions like :func:`~retworkx.floyd_warshall_numpy` in
    that the edge weight/data payload is not used and each edge is treated as a
    distance of 1.

    This function is also multithreaded and will run in parallel if the number
    of nodes in the graph is above the value of ``parallel_threshold`` (it
    defaults to 300). If the function will be running in parallel the env var
    ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.

    :param graph: The graph to get the distance matrix for, can be either a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.
    :param int parallel_threshold: The number of nodes to calculate the
        the distance matrix in parallel at. It defaults to 300, but this can
        be tuned
    :param bool as_undirected: If set to ``True`` the input directed graph
        will be treat as if each edge was bidirectional/undirected in the
        output distance matrix.
    :param float null_value: An optional float that will treated as a null
        value. This is the default value in the output matrix and it is used
        to indicate the absence of an edge between 2 nodes. By default this
        is ``0.0``.

    :returns: The distance matrix
    :rtype: numpy.ndarray
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@distance_matrix.register(PyDiGraph)
def _digraph_distance_matrix(
    graph, parallel_threshold=300, as_undirected=False, null_value=0.0
):
    return digraph_distance_matrix(
        graph,
        parallel_threshold=parallel_threshold,
        as_undirected=as_undirected,
        null_value=null_value,
    )


@distance_matrix.register(PyGraph)
def _graph_distance_matrix(graph, parallel_threshold=300, null_value=0.0):
    return graph_distance_matrix(
        graph, parallel_threshold=parallel_threshold, null_value=null_value
    )


@functools.singledispatch
def unweighted_average_shortest_path_length(graph, parallel_threshold=300):
    r"""Return the average shortest path length with unweighted edges.

    The average shortest path length is calculated as

    .. math::

        a =\sum_{s,t \in V} \frac{d(s, t)}{n(n-1)}

    where :math:`V` is the set of nodes in ``graph``, :math:`d(s, t)` is the
    shortest path length from :math:`s` to :math:`t`, and :math:`n` is the
    number of nodes in ``graph``. This also assumes that :math:`d(s, t) = 0`
    if :math:`t` cannot be reached from :math:`s`.

    This function is also multithreaded and will run in parallel if the number
    of nodes in the graph is above the value of ``parallel_threshold`` (it
    defaults to 300). If the function will be running in parallel the env var
    ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
    By default it will use all available CPUs if the environment variable is
    not specified.

    :param PyDiGraph graph: The graph to compute the average shortest path length
        for
    :param int parallel_threshold: The number of nodes to calculate the
        the distance matrix in parallel at. It defaults to 300, but this can
        be tuned to any number of nodes.
    :param bool as_undirected: If set to ``True`` the input directed graph
        will be treated as if each edge was bidirectional/undirected while
        finding the shortest paths. Default: ``False``.

    :returns: The average shortest path length. If the graph is empty this
        will return NaN and if there is a single node 0 will be returned.

    :rtype: float
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@unweighted_average_shortest_path_length.register(PyDiGraph)
def _digraph_unweighted_average_shortest_path_length(
    graph, parallel_threshold=300, as_undirected=False
):
    return digraph_unweighted_average_shortest_path_length(
        graph,
        parallel_threshold=parallel_threshold,
        as_undirected=as_undirected,
    )


@unweighted_average_shortest_path_length.register(PyGraph)
def _graph_unweighted_shortest_path_length(graph, parallel_threshold=300):
    return graph_unweighted_average_shortest_path_length(
        graph, parallel_threshold=parallel_threshold
    )


@functools.singledispatch
def adjacency_matrix(graph, weight_fn=None, default_weight=1.0, null_value=0.0):
    """Return the adjacency matrix for a graph object

    In the case where there are multiple edges between nodes the value in the
    output matrix will be the sum of the edges' weights.

    :param graph: The graph used to generate the adjacency matrix from. Can
        either be a :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param callable weight_fn: A callable object (function, lambda, etc) which
        will be passed the edge object and expected to return a ``float``. This
        tells retworkx/rust how to extract a numerical weight as a ``float``
        for edge object. Some simple examples are::

            adjacency_matrix(graph, weight_fn: lambda x: 1)

        to return a weight of 1 for all edges. Also::

            adjacency_matrix(graph, weight_fn: lambda x: float(x))

        to cast the edge object as a float as the weight. If this is not
        specified a default value (either ``default_weight`` or 1) will be used
        for all edges.
    :param float default_weight: If ``weight_fn`` is not used this can be
        optionally used to specify a default weight to use for all edges.
    :param float null_value: An optional float that will treated as a null
        value. This is the default value in the output matrix and it is used
        to indicate the absence of an edge between 2 nodes. By default this is
        ``0.0``.

     :return: The adjacency matrix for the input dag as a numpy array
     :rtype: numpy.ndarray
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@adjacency_matrix.register(PyDiGraph)
def _digraph_adjacency_matrix(
    graph, weight_fn=None, default_weight=1.0, null_value=0.0
):
    return digraph_adjacency_matrix(
        graph,
        weight_fn=weight_fn,
        default_weight=default_weight,
        null_value=null_value,
    )


@adjacency_matrix.register(PyGraph)
def _graph_adjacency_matrix(
    graph, weight_fn=None, default_weight=1.0, null_value=0.0
):
    return graph_adjacency_matrix(
        graph,
        weight_fn=weight_fn,
        default_weight=default_weight,
        null_value=null_value,
    )


@functools.singledispatch
def all_simple_paths(graph, from_, to, min_depth=None, cutoff=None):
    """Return all simple paths between 2 nodes in a PyGraph object

    A simple path is a path with no repeated nodes.

    :param graph: The graph to find the path in. Can either be a
        class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param int from_: The node index to find the paths from
    :param int to: The node index to find the paths to
    :param int min_depth: The minimum depth of the path to include in the
        output list of paths. By default all paths are included regardless of
        depth, setting to 0 will behave like the default.
    :param int cutoff: The maximum depth of path to include in the output list
        of paths. By default includes all paths regardless of depth, setting to
        0 will behave like default.

    :returns: A list of lists where each inner list is a path of node indices
    :rtype: list
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@all_simple_paths.register(PyDiGraph)
def _digraph_all_simple_paths(graph, from_, to, min_depth=None, cutoff=None):
    return digraph_all_simple_paths(
        graph, from_, to, min_depth=min_depth, cutoff=cutoff
    )


@all_simple_paths.register(PyGraph)
def _graph_all_simple_paths(graph, from_, to, min_depth=None, cutoff=None):
    return graph_all_simple_paths(
        graph, from_, to, min_depth=min_depth, cutoff=cutoff
    )


@functools.singledispatch
def floyd_warshall(
    graph,
    weight_fn=None,
    default_weight=1.0,
    parallel_threshold=300,
):
    """Find all-pairs shortest path lengths using Floyd's algorithm

    Floyd's algorithm is used for finding shortest paths in dense graphs
    or graphs with negative weights (where Dijkstra's algorithm fails).

    This function is multithreaded and will launch a pool with threads equal
    to the number of CPUs by default if the number of nodes in the graph is
    above the value of ``parallel_threshold`` (it defaults to 300).
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads if parallelization was enabled.

    :param graph: The graph to run Floyd's algorithm on. Can
        either be a :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param callable weight_fn: A callable object (function, lambda, etc) which
        will be passed the edge object and expected to return a ``float``. This
        tells retworkx/rust how to extract a numerical weight as a ``float``
        for edge object. Some simple examples are::

            floyd_warshall(graph, weight_fn= lambda x: 1)

        to return a weight of 1 for all edges. Also::

            floyd_warshall(graph, weight_fn=float)

        to cast the edge object as a float as the weight. If this is not
        specified a default value (either ``default_weight`` or 1) will be used
        for all edges.
    :param float default_weight: If ``weight_fn`` is not used this can be
        optionally used to specify a default weight to use for all edges.
    :param int parallel_threshold: The number of nodes to execute
        the algorithm in parallel at. It defaults to 300, but this can
        be tuned

    :return: A read-only dictionary of path lengths. The keys are the source
        node indices and the values are a dict of the target node and the
        length of the shortest path to that node. For example::

            {
                0: {0: 0.0, 1: 2.0, 2: 2.0},
                1: {1: 0.0, 2: 1.0},
                2: {0: 1.0, 2: 0.0},
            }

    :rtype: AllPairsPathLengthMapping
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@floyd_warshall.register(PyDiGraph)
def _digraph_floyd_warshall(
    graph,
    weight_fn=None,
    default_weight=1.0,
    parallel_threshold=300,
):
    return digraph_floyd_warshall(
        graph,
        weight_fn=weight_fn,
        default_weight=default_weight,
        parallel_threshold=parallel_threshold,
    )


@floyd_warshall.register(PyGraph)
def _graph_floyd_warshall(
    graph,
    weight_fn=None,
    default_weight=1.0,
    parallel_threshold=300,
):
    return graph_floyd_warshall(
        graph,
        weight_fn=weight_fn,
        default_weight=default_weight,
        parallel_threshold=parallel_threshold,
    )


@functools.singledispatch
def floyd_warshall_numpy(
    graph,
    weight_fn=None,
    default_weight=1.0,
    parallel_threshold=300,
):
    """Find all-pairs shortest path lengths using Floyd's algorithm

    Floyd's algorithm is used for finding shortest paths in dense graphs
    or graphs with negative weights (where Dijkstra's algorithm fails).

    This function is multithreaded and will launch a pool with threads equal
    to the number of CPUs by default if the number of nodes in the graph is
    above the value of ``parallel_threshold`` (it defaults to 300).
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads if parallelization was enabled.

    :param graph: The graph to run Floyd's algorithm on. Can
        either be a :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param callable weight_fn: A callable object (function, lambda, etc) which
        will be passed the edge object and expected to return a ``float``. This
        tells retworkx/rust how to extract a numerical weight as a ``float``
        for edge object. Some simple examples are::

            floyd_warshall_numpy(graph, weight_fn: lambda x: 1)

        to return a weight of 1 for all edges. Also::

            floyd_warshall_numpy(graph, weight_fn: lambda x: float(x))

        to cast the edge object as a float as the weight. If this is not
        specified a default value (either ``default_weight`` or 1) will be used
        for all edges.
    :param float default_weight: If ``weight_fn`` is not used this can be
        optionally used to specify a default weight to use for all edges.
    :param int parallel_threshold: The number of nodes to execute
        the algorithm in parallel at. It defaults to 300, but this can
        be tuned

    :returns: A matrix of shortest path distances between nodes. If there is no
        path between two nodes then the corresponding matrix entry will be
        ``np.inf``.
    :rtype: numpy.ndarray
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@floyd_warshall_numpy.register(PyDiGraph)
def _digraph_floyd_warshall_numpy(
    graph, weight_fn=None, default_weight=1.0, parallel_threshold=300
):
    return digraph_floyd_warshall_numpy(
        graph,
        weight_fn=weight_fn,
        default_weight=default_weight,
        parallel_threshold=parallel_threshold,
    )


@floyd_warshall_numpy.register(PyGraph)
def _graph_floyd_warshall_numpy(
    graph, weight_fn=None, default_weight=1.0, parallel_threshold=300
):
    return graph_floyd_warshall_numpy(
        graph,
        weight_fn=weight_fn,
        default_weight=default_weight,
        parallel_threshold=parallel_threshold,
    )


@functools.singledispatch
def astar_shortest_path(graph, node, goal_fn, edge_cost_fn, estimate_cost_fn):
    """Compute the A* shortest path for a graph

    :param graph: The input graph to use. Can
        either be a :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param int node: The node index to compute the path from
    :param goal_fn: A python callable that will take in 1 parameter, a node's
        data object and will return a boolean which will be True if it is the
        finish node.
    :param edge_cost_fn: A python callable that will take in 1 parameter, an
        edge's data object and will return a float that represents the cost
        of that edge. It must be non-negative.
    :param estimate_cost_fn: A python callable that will take in 1 parameter, a
        node's data object and will return a float which represents the
        estimated cost for the next node. The return must be non-negative. For
        the algorithm to find the actual shortest path, it should be
        admissible, meaning that it should never overestimate the actual cost
        to get to the nearest goal node.

    :returns: The computed shortest path between node and finish as a list
        of node indices.
    :rtype: NodeIndices
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@astar_shortest_path.register(PyDiGraph)
def _digraph_astar_shortest_path(
    graph, node, goal_fn, edge_cost_fn, estimate_cost_fn
):
    return digraph_astar_shortest_path(
        graph, node, goal_fn, edge_cost_fn, estimate_cost_fn
    )


@astar_shortest_path.register(PyGraph)
def _graph_astar_shortest_path(
    graph, node, goal_fn, edge_cost_fn, estimate_cost_fn
):
    return graph_astar_shortest_path(
        graph, node, goal_fn, edge_cost_fn, estimate_cost_fn
    )


@functools.singledispatch
def dijkstra_shortest_paths(
    graph,
    source,
    target=None,
    weight_fn=None,
    default_weight=1.0,
    as_undirected=False,
):
    """Find the shortest path from a node

    This function will generate the shortest path from a source node using
    Dijkstra's algorithm.

    :param graph: The input graph to use. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param int source: The node index to find paths from
    :param int target: An optional target to find a path to
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float
        which will be used to represent the weight/cost of the edge
    :param float default_weight: If ``weight_fn`` isn't specified this optional
        float value will be used for the weight/cost of each edge.
    :param bool as_undirected: If set to true the graph will be treated as
        undirected for finding the shortest path. This only works with a
        :class:`~retworkx.PyDiGraph` input for ``graph``

    :return: Dictionary of paths. The keys are destination node indices and
        the dict values are lists of node indices making the path.
    :rtype: dict
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@dijkstra_shortest_paths.register(PyDiGraph)
def _digraph_dijkstra_shortest_path(
    graph,
    source,
    target=None,
    weight_fn=None,
    default_weight=1.0,
    as_undirected=False,
):
    return digraph_dijkstra_shortest_paths(
        graph,
        source,
        target=target,
        weight_fn=weight_fn,
        default_weight=default_weight,
        as_undirected=as_undirected,
    )


@dijkstra_shortest_paths.register(PyGraph)
def _graph_dijkstra_shortest_path(
    graph, source, target=None, weight_fn=None, default_weight=1.0
):
    return graph_dijkstra_shortest_paths(
        graph,
        source,
        target=target,
        weight_fn=weight_fn,
        default_weight=default_weight,
    )


@functools.singledispatch
def all_pairs_dijkstra_shortest_paths(graph, edge_cost_fn):
    """Find the shortest path from all nodes

    This function will generate the shortest path from all nodes int the graph
    using Dijkstra's algorithm. This function is multithreaded and will run
    launch a thread pool with threads equal to the number of CPUs by default.
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads.

    :param graph: The input graph to use. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param edge_cost_fn: A callable object that acts as a weight function for
        an edge. It will accept a single positional argument, the edge's weight
        object and will return a float which will be used to represent the
        weight/cost of the edge

    :return: A read-only dictionary of paths. The keys are destination node
        indices and the values are a dict of target node indices and a list
        of node indices making the path. For example::

            {
                0: {1: [0, 1],  2: [0, 1, 2]},
                1: {2: [1, 2]},
                2: {0: [2, 0]},
            }

    :rtype: AllPairsPathMapping
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@all_pairs_dijkstra_shortest_paths.register(PyDiGraph)
def _digraph_all_pairsdijkstra_shortest_path(graph, edge_cost_fn):
    return digraph_all_pairs_dijkstra_shortest_paths(graph, edge_cost_fn)


@all_pairs_dijkstra_shortest_paths.register(PyGraph)
def _graph_all_pairs_dijkstra_shortest_path(graph, edge_cost_fn):
    return graph_all_pairs_dijkstra_shortest_paths(graph, edge_cost_fn)


@functools.singledispatch
def all_pairs_dijkstra_path_lengths(graph, edge_cost_fn):
    """Find the shortest path from a node

    This function will generate the shortest path from a source node using
    Dijkstra's algorithm. This function is multithreaded and will run
    launch a thread pool with threads equal to the number of CPUs by default.
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads.

    :param graph: The input graph to use. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param edge_cost_fn: A callable object that acts as a weight function for
        an edge. It will accept a single positional argument, the edge's weight
        object and will return a float which will be used to represent the
        weight/cost of the edge

    :return: A read-only dictionary of path lengths. The keys are the source
        node indices and the values are a dict of the target node and the
        length of the shortest path to that node. For example::

            {
                0: {1: 2.0, 2: 2.0},
                1: {2: 1.0},
                2: {0: 1.0},
            }

    :rtype: AllPairsPathLengthMapping
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@all_pairs_dijkstra_path_lengths.register(PyDiGraph)
def _digraph_all_pairs_dijkstra_path_lengths(graph, edge_cost_fn):
    return digraph_all_pairs_dijkstra_path_lengths(graph, edge_cost_fn)


@all_pairs_dijkstra_path_lengths.register(PyGraph)
def _graph_all_pairs_dijkstra_path_lengths(graph, edge_cost_fn):
    return graph_all_pairs_dijkstra_path_lengths(graph, edge_cost_fn)


@functools.singledispatch
def dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, goal=None):
    """Compute the lengths of the shortest paths for a graph object using
    Dijkstra's algorithm.

    :param graph: The input graph to use. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param int node: The node index to use as the source for finding the
        shortest paths from
    :param edge_cost_fn: A python callable that will take in 1 parameter, an
        edge's data object and will return a float that represents the
        cost/weight of that edge. It must be non-negative
    :param int goal: An optional node index to use as the end of the path.
        When specified the traversal will stop when the goal is reached and
        the output dictionary will only have a single entry with the length
        of the shortest path to the goal node.

    :returns: A dictionary of the shortest paths from the provided node where
        the key is the node index of the end of the path and the value is the
        cost/sum of the weights of path
    :rtype: dict
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@dijkstra_shortest_path_lengths.register(PyDiGraph)
def _digraph_dijkstra_shortest_path_lengths(
    graph, node, edge_cost_fn, goal=None
):
    return digraph_dijkstra_shortest_path_lengths(
        graph, node, edge_cost_fn, goal=goal
    )


@dijkstra_shortest_path_lengths.register(PyGraph)
def _graph_dijkstra_shortest_path_lengths(graph, node, edge_cost_fn, goal=None):
    return graph_dijkstra_shortest_path_lengths(
        graph, node, edge_cost_fn, goal=goal
    )


@functools.singledispatch
def k_shortest_path_lengths(graph, start, k, edge_cost, goal=None):
    """Compute the length of the kth shortest path

    Computes the lengths of the kth shortest path from ``start`` to every
    reachable node.

    Computes in :math:`O(k * (|E| + |V|*log(|V|)))` time (average).

    :param graph: The graph to find the shortest paths in. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param int start: The node index to find the shortest paths from
    :param int k: The kth shortest path to find the lengths of
    :param edge_cost: A python callable that will receive an edge payload and
        return a float for the cost of that eedge
    :param int goal: An optional goal node index, if specified the output
        dictionary

    :returns: A dict of lengths where the key is the destination node index and
        the value is the length of the path.
    :rtype: dict
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@k_shortest_path_lengths.register(PyDiGraph)
def _digraph_k_shortest_path_lengths(graph, start, k, edge_cost, goal=None):
    return digraph_k_shortest_path_lengths(
        graph, start, k, edge_cost, goal=goal
    )


@k_shortest_path_lengths.register(PyGraph)
def _graph_k_shortest_path_lengths(graph, start, k, edge_cost, goal=None):
    return graph_k_shortest_path_lengths(graph, start, k, edge_cost, goal=goal)


@functools.singledispatch
def dfs_edges(graph, source):
    """Get edge list in depth first order

    :param PyGraph graph: The graph to get the DFS edge list from
    :param int source: An optional node index to use as the starting node
        for the depth-first search. The edge list will only return edges in
        the components reachable from this index. If this is not specified
        then a source will be chosen arbitrarly and repeated until all
        components of the graph are searched.

    :returns: A list of edges as a tuple of the form ``(source, target)`` in
        depth-first order
    :rtype: EdgeList
        raise TypeError("Invalid Input Type %s for graph" % type(graph))
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@dfs_edges.register(PyDiGraph)
def _digraph_dfs_edges(graph, source):
    return digraph_dfs_edges(graph, source)


@dfs_edges.register(PyGraph)
def _graph_dfs_edges(graph, source):
    return graph_dfs_edges(graph, source)


@functools.singledispatch
def is_isomorphic(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=True,
    call_limit=None,
):
    """Determine if 2 graphs are isomorphic

    This checks if 2 graphs are isomorphic both structurally and also
    comparing the node and edge data using the provided matcher functions.
    The matcher functions take in 2 data objects and will compare them. A
    simple example that checks if they're just equal would be::

            graph_a = retworkx.PyGraph()
            graph_b = retworkx.PyGraph()
            retworkx.is_isomorphic(graph_a, graph_b,
                                lambda x, y: x == y)

    .. note::

        For better performance on large graphs, consider setting
        `id_order=False`.

    :param first: The first graph to compare. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.
    :param second: The second graph to compare. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.
        It should be the same type as the first graph.
    :param callable node_matcher: A python callable object that takes 2
        positional one for each node data object. If the return of this
        function evaluates to True then the nodes passed to it are viewed
        as matching.
    :param callable edge_matcher: A python callable object that takes 2
        positional one for each edge data object. If the return of this
        function evaluates to True then the edges passed to it are viewed
        as matching.
    :param bool id_order: If set to ``False`` this function will use a
        heuristic matching order based on [VF2]_ paper. Otherwise it will
        default to matching the nodes in order specified by their ids.
    :param int call_limit: An optional bound on the number of states that VF2
        algorithm visits while searching for a solution. If it exceeds this limit,
        the algorithm will stop and return ``False``.

    :returns: ``True`` if the 2 graphs are isomorphic, ``False`` if they are
        not.
    :rtype: bool

    .. [VF2] VF2++  An Improved Subgraph Isomorphism Algorithm
        by Alpár Jüttner and Péter Madarasi
    """
    raise TypeError("Invalid Input Type %s for graph" % type(first))


@is_isomorphic.register(PyDiGraph)
def _digraph_is_isomorphic(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=True,
    call_limit=None,
):
    return digraph_is_isomorphic(
        first, second, node_matcher, edge_matcher, id_order, call_limit
    )


@is_isomorphic.register(PyGraph)
def _graph_is_isomorphic(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=True,
    call_limit=None,
):
    return graph_is_isomorphic(
        first, second, node_matcher, edge_matcher, id_order, call_limit
    )


@functools.singledispatch
def is_isomorphic_node_match(first, second, matcher, id_order=True):
    """Determine if 2 graphs are isomorphic

    This checks if 2 graphs are isomorphic both structurally and also
    comparing the node data using the provided matcher function. The matcher
    function takes in 2 node data objects and will compare them. A simple
    example that checks if they're just equal would be::

        graph_a = retworkx.PyDAG()
        graph_b = retworkx.PyDAG()
        retworkx.is_isomorphic_node_match(graph_a, graph_b,
                                        lambda x, y: x == y)

    .. note::

        For better performance on large graphs, consider setting
        `id_order=False`.

    :param first: The first graph to compare. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.
    :param second: The second graph to compare. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.
        It should be the same type as the first graph.
    :param callable matcher: A python callable object that takes 2 positional
        one for each node data object. If the return of this
        function evaluates to True then the nodes passed to it are vieded
        as matching.
    :param bool id_order: If set to ``False`` this function will use a
        heuristic matching order based on [VF2]_ paper. Otherwise it will
        default to matching the nodes in order specified by their ids.

    :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
        not.
    :rtype: bool
    """
    raise TypeError("Invalid Input Type %s for graph" % type(first))


@is_isomorphic_node_match.register(PyDiGraph)
def _digraph_is_isomorphic_node_match(first, second, matcher, id_order=True):
    return digraph_is_isomorphic(first, second, matcher, id_order=id_order)


@is_isomorphic_node_match.register(PyGraph)
def _graph_is_isomorphic_node_match(first, second, matcher, id_order=True):
    return graph_is_isomorphic(first, second, matcher, id_order=id_order)


@functools.singledispatch
def is_subgraph_isomorphic(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=False,
    induced=True,
    call_limit=None,
):
    """Determine if 2 graphs are subgraph isomorphic

    This checks if 2 graphs are subgraph isomorphic both structurally and also
    comparing the node and edge data using the provided matcher functions.
    The matcher functions take in 2 data objects and will compare them.
    Since there is an ambiguity in the term 'subgraph', do note that we check
    for an node-induced subgraph if argument `induced` is set to `True`. If it is
    set to `False`, we check for a non induced subgraph, meaning the second graph
    can have fewer edges than the subgraph of the first. By default it's `True`. A
    simple example that checks if they're just equal would be::

            graph_a = retworkx.PyGraph()
            graph_b = retworkx.PyGraph()
            retworkx.is_subgraph_isomorphic(graph_a, graph_b,
                                            lambda x, y: x == y)


    :param first: The first graph to compare. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.
    :param second: The second graph to compare. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.
        It should be the same type as the first graph.
    :param callable node_matcher: A python callable object that takes 2
        positional one for each node data object. If the return of this
        function evaluates to True then the nodes passed to it are viewed
        as matching.
    :param callable edge_matcher: A python callable object that takes 2
        positional one for each edge data object. If the return of this
        function evaluates to True then the edges passed to it are viewed
        as matching.
    :param bool id_order: If set to ``True`` this function will match the nodes
        in order specified by their ids. Otherwise it will default to a heuristic
        matching order based on [VF2]_ paper.
    :param bool induced: If set to ``True`` this function will check the existence
        of a node-induced subgraph of first isomorphic to second graph.
        Default: ``True``.
    :param int call_limit: An optional bound on the number of states that VF2
        algorithm visits while searching for a solution. If it exceeds this limit,
        the algorithm will stop and return ``False``.

    :returns: ``True`` if there is a subgraph of `first` isomorphic to `second`
        , ``False`` if there is not.
    :rtype: bool
    """
    raise TypeError("Invalid Input Type %s for graph" % type(first))


@is_subgraph_isomorphic.register(PyDiGraph)
def _digraph_is_subgraph_isomorphic(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=False,
    induced=True,
    call_limit=None,
):
    return digraph_is_subgraph_isomorphic(
        first, second, node_matcher, edge_matcher, id_order, induced, call_limit
    )


@is_subgraph_isomorphic.register(PyGraph)
def _graph_is_subgraph_isomorphic(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=False,
    induced=True,
    call_limit=None,
):
    return graph_is_subgraph_isomorphic(
        first, second, node_matcher, edge_matcher, id_order, induced, call_limit
    )


@functools.singledispatch
def transitivity(graph):
    """Compute the transitivity of a graph.

    This function is multithreaded and will run
    launch a thread pool with threads equal to the number of CPUs by default.
    You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
    limit the thread pool to 4 threads.

    .. note::

        The function implicitly assumes that there are no parallel edges
        or self loops. It may produce incorrect/unexpected results if the
        input graph has self loops or parallel edges.

    :param graph: The graph to be used. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.

    :returns: Transitivity of the graph.
    :rtype: float
        raise TypeError("Invalid Input Type %s for graph" % type(graph))
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@transitivity.register(PyDiGraph)
def _digraph_transitivity(graph):
    return digraph_transitivity(graph)


@transitivity.register(PyGraph)
def _graph_transitivity(graph):
    return graph_transitivity(graph)


@functools.singledispatch
def core_number(graph):
    """Return the core number for each node in the graph.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    .. note::

        The function implicitly assumes that there are no parallel edges
        or self loops. It may produce incorrect/unexpected results if the
        input graph has self loops or parallel edges.

    :param graph: The graph to get core numbers. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`

    :returns: A dictionary keyed by node index to the core number
    :rtype: dict
        raise TypeError("Invalid Input Type %s for graph" % type(graph))
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@core_number.register(PyDiGraph)
def _digraph_core_number(graph):
    return digraph_core_number(graph)


@core_number.register(PyGraph)
def _graph_core_number(graph):
    return graph_core_number(graph)


@functools.singledispatch
def complement(graph):
    """Compute the complement of a graph.

    :param graph: The graph to be used, can be either a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.

    :returns: The complement of the graph.
    :rtype: :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`

    .. note::
        Parallel edges and self-loops are never created,
        even if the ``multigraph`` is set to ``True``
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@complement.register(PyDiGraph)
def _digraph_complement(graph):
    return digraph_complement(graph)


@complement.register(PyGraph)
def _graph_complement(graph):
    return graph_complement(graph)


@functools.singledispatch
def random_layout(graph, center=None, seed=None):
    """Generate a random layout

    :param PyGraph graph: The graph to generate the layout for
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position
    :param int seed: An optional seed to set for the random number generator.

    :returns: The random layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@random_layout.register(PyDiGraph)
def _digraph_random_layout(graph, center=None, seed=None):
    return digraph_random_layout(graph, center=center, seed=seed)


@random_layout.register(PyGraph)
def _graph_random_layout(graph, center=None, seed=None):
    return graph_random_layout(graph, center=center, seed=seed)


@functools.singledispatch
def spring_layout(
    graph,
    pos=None,
    fixed=None,
    k=None,
    repulsive_exponent=2,
    adaptive_cooling=True,
    num_iter=50,
    tol=1e-6,
    weight_fn=None,
    default_weight=1,
    scale=1,
    center=None,
    seed=None,
):
    """
    Position nodes using Fruchterman-Reingold force-directed algorithm.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an anti-gravity force.
    Simulation continues until the positions are close to an equilibrium.

    :param graph: Graph to be used. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.
    :param dict pos:
        Initial node positions as a dictionary with node ids as keys and values
        as a coordinate list. If ``None``, then use random initial positions.
        (``default=None``)
    :param set fixed: Nodes to keep fixed at initial position.
        Error raised if fixed specified and ``pos`` is not. (``default=None``)
    :param float  k:
        Optimal distance between nodes. If ``None`` the distance is set to
        :math:`\\frac{1}{\\sqrt{n}}` where :math:`n` is the number of nodes.
        Increase this value to move nodes farther apart. (``default=None``)
    :param int repulsive_exponent:
        Repulsive force exponent. (``default=2``)
    :param bool adaptive_cooling:
        Use an adaptive cooling scheme. If set to ``False``,
        a linear cooling scheme is used. (``default=True``)
    :param int num_iter:
        Maximum number of iterations. (``default=50``)
    :param float tol:
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.
        (``default = 1e-6``)
    :param weight_fn: An optional weight function for an edge. It will accept
        a single argument, the edge's weight object and will return a float
        which will be used to represent the weight of the edge.
    :param float (default=1) default_weight: If ``weight_fn`` isn't specified
        this optional float value will be used for the weight/cost of each edge
    :param float|None scale: Scale factor for positions.
        Not used unless fixed is None. If scale is ``None``, no re-scaling is
        performed. (``default=1.0``)
    :param list center: Coordinate pair around which to center
        the layout. Not used unless fixed is ``None``. (``default=None``)
    :param int seed: An optional seed to use for the random number generator

    :returns: A dictionary of positions keyed by node id.
    :rtype: dict
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@spring_layout.register(PyDiGraph)
def _digraph_spring_layout(
    graph,
    pos=None,
    fixed=None,
    k=None,
    repulsive_exponent=2,
    adaptive_cooling=True,
    num_iter=50,
    tol=1e-6,
    weight_fn=None,
    default_weight=1,
    scale=1,
    center=None,
    seed=None,
):
    return digraph_spring_layout(
        graph,
        pos,
        fixed,
        k,
        repulsive_exponent,
        adaptive_cooling,
        num_iter,
        tol,
        weight_fn,
        default_weight,
        scale,
        center,
        seed,
    )


@spring_layout.register(PyGraph)
def _graph_spring_layout(
    graph,
    pos=None,
    fixed=None,
    k=None,
    repulsive_exponent=2,
    adaptive_cooling=True,
    num_iter=50,
    tol=1e-6,
    weight_fn=None,
    default_weight=1,
    scale=1,
    center=None,
    seed=None,
):
    return graph_spring_layout(
        graph,
        pos,
        fixed,
        k,
        repulsive_exponent,
        adaptive_cooling,
        num_iter,
        tol,
        weight_fn,
        default_weight,
        scale,
        center,
        seed,
    )


def networkx_converter(graph):
    """Convert a networkx graph object into a retworkx graph object.

    .. note::

        networkx is **not** a dependency of retworkx and this function
        is provided as a convenience method for users of both networkx and
        retworkx. This function will not work unless you install networkx
        independently.

    :param networkx.Graph graph: The networkx graph to convert.

    :returns: A retworkx graph, either a :class:`~retworkx.PyDiGraph` or a
        :class:`~retworkx.PyGraph` based on whether the input graph is directed
        or not.
    :rtype: :class:`~retworkx.PyDiGraph` or :class:`~retworkx.PyGraph`
    """
    if graph.is_directed():
        new_graph = PyDiGraph(multigraph=graph.is_multigraph())
    else:
        new_graph = PyGraph(multigraph=graph.is_multigraph())
    nodes = list(graph.nodes)
    node_indices = dict(zip(nodes, new_graph.add_nodes_from(nodes)))
    new_graph.add_edges_from(
        [
            (node_indices[x[0]], node_indices[x[1]], x[2])
            for x in graph.edges(data=True)
        ]
    )
    return new_graph


@functools.singledispatch
def bipartite_layout(
    graph,
    first_nodes,
    horizontal=False,
    scale=1,
    center=None,
    aspect_ratio=4 / 3,
):
    """Generate a bipartite layout of the graph

    :param graph: The graph to generate the layout for. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param set first_nodes: The set of node indexes on the left (or top if
        horitontal is true)
    :param bool horizontal: An optional bool specifying the orientation of the
        layout
    :param float scale: An optional scaling factor to scale positions
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position
    :param float aspect_ratio: An optional number for the ratio of the width to
        the height of the layout.

    :returns: The bipartite layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@bipartite_layout.register(PyDiGraph)
def _digraph_bipartite_layout(
    graph,
    first_nodes,
    horizontal=False,
    scale=1,
    center=None,
    aspect_ratio=4 / 3,
):
    return digraph_bipartite_layout(
        graph,
        first_nodes,
        horizontal=horizontal,
        scale=scale,
        center=center,
        aspect_ratio=aspect_ratio,
    )


@bipartite_layout.register(PyGraph)
def _graph_bipartite_layout(
    graph,
    first_nodes,
    horizontal=False,
    scale=1,
    center=None,
    aspect_ratio=4 / 3,
):
    return graph_bipartite_layout(
        graph,
        first_nodes,
        horizontal=horizontal,
        scale=scale,
        center=center,
        aspect_ratio=aspect_ratio,
    )


@functools.singledispatch
def circular_layout(graph, scale=1, center=None):
    """Generate a circular layout of the graph

    :param graph: The graph to generate the layout for. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param float scale: An optional scaling factor to scale positions
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position

    :returns: The circular layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@circular_layout.register(PyDiGraph)
def _digraph_circular_layout(graph, scale=1, center=None):
    return digraph_circular_layout(graph, scale=scale, center=center)


@circular_layout.register(PyGraph)
def _graph_circular_layout(graph, scale=1, center=None):
    return graph_circular_layout(graph, scale=scale, center=center)


@functools.singledispatch
def shell_layout(graph, nlist=None, rotate=None, scale=1, center=None):
    """
    Generate a shell layout of the graph

    :param graph: The graph to generate the layout for. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param list nlist: The list of lists of indexes which represents each shell
    :param float rotate: Angle (in radians) by which to rotate the starting
        position of each shell relative to the starting position of the
        previous shell
    :param float scale: An optional scaling factor to scale positions
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position

    :returns: The shell layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@shell_layout.register(PyDiGraph)
def _digraph_shell_layout(graph, nlist=None, rotate=None, scale=1, center=None):
    return digraph_shell_layout(
        graph, nlist=nlist, rotate=rotate, scale=scale, center=center
    )


@shell_layout.register(PyGraph)
def _graph_shell_layout(graph, nlist=None, rotate=None, scale=1, center=None):
    return graph_shell_layout(
        graph, nlist=nlist, rotate=rotate, scale=scale, center=center
    )


@functools.singledispatch
def spiral_layout(
    graph, scale=1, center=None, resolution=0.35, equidistant=False
):
    """
    Generate a spiral layout of the graph

    :param graph: The graph to generate the layout for. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`
    :param float scale: An optional scaling factor to scale positions
    :param tuple center: An optional center position. This is a 2 tuple of two
        ``float`` values for the center position
    :param float resolution: The compactness of the spiral layout returned.
        Lower values result in more compressed spiral layouts.
    :param bool equidistant: If true, nodes will be plotted equidistant from
        each other.

    :returns: The spiral layout of the graph.
    :rtype: Pos2DMapping
    """
    raise TypeError("Invalid Input Type %s for graph" % type(graph))


@spiral_layout.register(PyDiGraph)
def _digraph_spiral_layout(
    graph, scale=1, center=None, resolution=0.35, equidistant=False
):
    return digraph_spiral_layout(
        graph,
        scale=scale,
        center=center,
        resolution=resolution,
        equidistant=equidistant,
    )


@spiral_layout.register(PyGraph)
def _graph_spiral_layout(
    graph, scale=1, center=None, resolution=0.35, equidistant=False
):
    return graph_spiral_layout(
        graph,
        scale=scale,
        center=center,
        resolution=resolution,
        equidistant=equidistant,
    )


@functools.singledispatch
def num_shortest_paths_unweighted(graph, source):
    """Get the number of unweighted shortest paths from a source node

    :param PyDiGraph graph: The graph to find the number of shortest paths on
    :param int source: The source node to find the shortest paths from

    :returns: A mapping of target node indices to the number of shortest paths
        from ``source`` to that node. If there is no path from ``source`` to
        a node in the graph that node will not be preset in the output mapping.
    :rtype: NodesCountMapping
    """
    raise TypeError("Invalid input type %s for graph" % type(graph))


@num_shortest_paths_unweighted.register(PyDiGraph)
def _digraph_num_shortest_paths_unweighted(graph, source):
    return digraph_num_shortest_paths_unweighted(graph, source)


@num_shortest_paths_unweighted.register(PyGraph)
def _graph_num_shortest_paths_unweighted(graph, source):
    return graph_num_shortest_paths_unweighted(graph, source)


@functools.singledispatch
def betweenness_centrality(graph, normalized=True, endpoints=False):
    """Returns the betweenness centrality of each node in the graph.

    Betweenness centrality of a node `v` is the sum of the fraction
    of shortest paths between all pairs of nodes that pass through `v`.

    :param graph: The graph to compute the betweenness centralities for.
    :param bool normalized: If true, normalize the betweenness scores by the number
        of distinct paths between all pairs of nodes.
    :param bool endpoints: Whether to include the endpoints of paths in pathlengths
        used to compute the betweenness.

    :returns: A dictionary mapping each node index to its betweenness centrality.
    :rtype: dict
    """
    raise TypeError("Invalid input type %s for graph" % type(graph))


@betweenness_centrality.register(PyDiGraph)
def _digraph_betweenness_centrality(graph, normalized=True, endpoints=False):
    return digraph_betweenness_centrality(
        graph, normalized=normalized, endpoints=endpoints
    )


@betweenness_centrality.register(PyGraph)
def _graph_betweenness_centrality(graph, normalized=True, endpoints=False):
    return graph_betweenness_centrality(
        graph, normalized=normalized, endpoints=endpoints
    )


@functools.singledispatch
def vf2_mapping(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=True,
    subgraph=False,
    induced=True,
    call_limit=None,
):
    """
    Return an iterator over all vf2 mappings between two graphs.

    This funcion will run the vf2 algorithm used from
    :func:`~retworkx.is_isomorphic` and :func:`~retworkx.is_subgraph_isomorphic`
    but instead of returning a boolean it will return an iterator over all possible
    mapping of node ids found from ``first`` to ``second``. If the graphs are not
    isomorphic then the iterator will be empty. A simple example that retrieves
    one mapping would be::

            graph_a = retworkx.generators.path_graph(3)
            graph_b = retworkx.generators.path_graph(2)
            vf2 = retworkx.vf2_mapping(graph_a, graph_b, subgraph=True)
            try:
                mapping = next(vf2)
            except StopIteration:
                pass

    :param first: The first graph to find the mapping for
    :param second: The second graph to find the mapping for
    :param node_matcher: An optional python callable object that takes 2
        positional arguments, one for each node data object in either graph.
        If the return of this function evaluates to True then the nodes
        passed to it are viewed as matching.
    :param edge_matcher: A python callable object that takes 2 positional
        one for each edge data object. If the return of this
        function evaluates to True then the edges passed to it are viewed
        as matching.
    :param bool id_order: If set to ``False`` this function will use a
        heuristic matching order based on [VF2]_ paper. Otherwise it will
        default to matching the nodes in order specified by their ids.
    :param bool subgraph: If set to ``True`` the function will return the
        subgraph isomorphic found between the graphs.
    :param bool induced: If set to ``True`` this function will check the existence
        of a node-induced subgraph of first isomorphic to second graph.
        Default: ``True``.
    :param int call_limit: An optional bound on the number of states that VF2
        algorithm visits while searching for a solution. If it exceeds this limit,
        the algorithm will stop. Default: ``None``.

    :returns: An iterator over dicitonaries of node indices from ``first`` to node
        indices in ``second`` representing the mapping found.
    :rtype: Iterable[NodeMap]
    """
    raise TypeError("Invalid Input Type %s for graph" % type(first))


@vf2_mapping.register(PyDiGraph)
def _digraph_vf2_mapping(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=True,
    subgraph=False,
    induced=True,
    call_limit=None,
):
    return digraph_vf2_mapping(
        first,
        second,
        node_matcher=node_matcher,
        edge_matcher=edge_matcher,
        id_order=id_order,
        subgraph=subgraph,
        induced=induced,
        call_limit=call_limit,
    )


@vf2_mapping.register(PyGraph)
def _graph_vf2_mapping(
    first,
    second,
    node_matcher=None,
    edge_matcher=None,
    id_order=True,
    subgraph=False,
    induced=True,
    call_limit=None,
):
    return graph_vf2_mapping(
        first,
        second,
        node_matcher=node_matcher,
        edge_matcher=edge_matcher,
        id_order=id_order,
        subgraph=subgraph,
        induced=induced,
        call_limit=call_limit,
    )
