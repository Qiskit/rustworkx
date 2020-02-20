retworkx API
=============

.. py:class:: PyDAG
   A class for creating direct acyclic graphs.

   The PyDAG class is constructed using the Rust library `petgraph`_ around
   the ``StableGraph`` type. The limitations and quirks with this library and
   type dictate how this operates. The biggest thing to be aware of when using
   the PyDAG class is that an integer node and edge index is used for accessing
   elements on the DAG, not the data/weight of nodes and edges. By default the
   PyDAG realtime cycle checking is disabled for performance, however you can
   opt-in to having the PyDAG class ensure that no cycles are added by setting
   the ``check_cycle`` attribute to True. For example::

       import retworkx
       dag = retworkx.PyDAG()
       dag.check_cycle = True

   With check_cycle set to true any calls to :method:`PyDAG.add_edge` will
   ensure that no cycles are added, ensuring that the PyDAG class truly
   represents a directed acyclic graph.

     .. note::
          When using ``copy.deepcopy()`` or pickling node indexes are not
          guaranteed to be preserved.

    .. py:method:: __init__(self):
        Initialize an empty DAG.

    .. py:method:: __len__(self):
        Return the number of nodes in the graph. Use via ``len()`` function

    .. py:method:: edges(self):
        Return a list of all edge data.

        :returns: A list of all the edge data objects in the DAG
        :rtype: list

    .. py:method:: has_edge(self, node_a, node_b):
        Return True if there is an edge from node_a to node_b.

        :param int node_a: The source node index to check for an edge
        :param int node_b: The destination node index to check for an edge

        :returns: True if there is an edge false if there is no edge
        :rtype: bool

    .. py:method:: nodes(self):
        Return a list of all node data.

        :returns: A list of all the node data objects in the DAG
        :rtype: list

    .. py:method:: successors(self, node):
        Return a list of all the node successor data.

        :param int node: The index for the node to get the successors for

        :returns: A list of the node data for all the child neighbor nodes
        :rtype: list

    .. py:method:: predecessors(self, node):
        Return a list of all the node predecessor data.

        :param int node: The index for the node to get the predecessors for

        :returns: A list of the node data for all the parent neighbor nodes
        :rtype: list

    .. py:method:: get_node_data(self, node):
        Return the node data for a given node index

        :param int node: The index for the node

        :returns: The data object set for that node
        :raises IndexError: when an invalid node index is provided

    .. py:method:: get_edge_data(self, node_a, node_b):
        Return the edge data for the edge between 2 nodes.

        :param int node_a: The index for the first node
        :param int node_b: The index for the second node

        :returns: The data object set for the edge
        :raises: When there is no edge between nodes

    .. py:method:: get_all_edge_data(self, node_a, node_b):
        Return the edge data for all the edges between 2 nodes.

        :param int node_a: The index for the first node
        :param int node_b: The index for the second node

        :returns: A list with all the data objects for the edges between nodes
        :rtype: list
        :raises: When there is no edge between nodes

    .. py:method:: remove_node(self, node):
        Remove a node from the DAG.

        NOTE: Removal of a node may change the index for other nodes in the
        DAG. The last node will shift to the index of the removed node to take
        its place.

        :param int node: The index of the node to remove

    .. py:method:: add_edge(self, parent, child, edge):
        Add an edge between 2 nodes.

        Use add_child() or add_parent() to create a node with an edge at the
        same time as an edge for better performance. Using this method will
        enable adding duplicate edges between nodes.

        :param int parent: Index of the parent node
        :param int child: Index of the child node
        :param edge: The object to set as the data for the edge. It can be any
            python object.

        :raises: When the new edge will create a cycle

    .. py:method:: add_node(self, obj):
        Add a new node to the dag.

        :param obj: The python object to attach to the node

        :returns index: The index of the newly created node
        :rtype: int

    .. py:method:: add_child(self, parent, obj, edge):
        Add a new child node to the dag.

        This will create a new node on the dag and add an edge from the parent
        to that new node.

        :param int parent: The index for the parent node
        :param obj: The python object to attach to the node
        :param edge: The python object to attach to the edge

        :returns index: The index of the newly created child node
        :rtype: int

    .. py:method:: add_parent(self, child, obj, edge):
        Add a new parent node to the dag.

        This create a new node on the dag and add an edge to the child from
        that new node

        :param int child: The index of the child node
        :param obj: The python object to attach to the node
        :param edge: The python object to attach to the edge

        :returns index: The index of the newly created parent node
        :rtype: int

    .. py:method:: adj(self, node):
        Get the index and data for the neighbors of a node.

        This will return a dictionary where the keys are the node indexes of
        the adjacent nodes (inbound or outbound) and the value is the edge data
        objects between that adjacent node and the provided node.

        :param int node: The index of the node to get the neighbors

        :returns neighbors: A dictionary where the keys are node indexes and
            the value is the edge data object for all nodes that share an
            edge with the specified node.
        :rtype: dict

    .. py:method:: adj_direction(self, node, direction):
        Get the index and data for either the parent or children of a node.

        This will return a dictionary where the keys are the node indexes of
        the adjacent nodes (inbound or outbound as specified) and the value
        is the edge data objects for the edges between that adjacent node
        and the provided node.

        :param int node: The index of the node to get the neighbors
        :param bool direction: The direction to use for finding nodes,
            True means inbound edges and False means outbound edges.

        :returns neighbors: A dictionary where the keys are node indexes and
            the value is the edge data object for all nodes that share an
            edge with the specified node.
        :rtype: dict
        :raises NoEdgeBetweenNodes if the DAG is broken and an edge can't be
            found to a neighbor node

    .. py:method:: in_edges(self, node):
        Get the index and edge data for all parents of a node.

        This will return a list of tuples with the parent index the node index
        and the edge data. This can be used to recreate add_edge() calls.

        :param int node: The index of the node to get the edges for

        :returns in_edges: A list of tuples of the form:
            (parent_index, node_index, edge_data)
        :rtype: list
        :raises NoEdgeBetweenNodes if the DAG is broken and an edge can't be
            found to a neighbor node

    .. py:method:: out_edges(self, node):
        Get the index and edge data for all children of a node.

        This will return a list of tuples with the child index the node index
        and the edge data. This can be used to recreate add_edge() calls.

        :param int node: The index of the node to get the edges for

        :returns out_edges: A list of tuples of the form:
            (node_index, child_index, edge_data)
        :rtype: list
        :raises NoEdgeBetweenNodes if the DAG is broken and an edge can't be
            found to a neighbor node

    .. py:method:: in_degree(self, node):
        Get the degree of a node for inbound edges.

        :param int node: The index of the node to find the inbound degree of

        :returns degree: The inbound degree for the specified node
        :rtype: int

    .. py:method:: out_degree(self, node):
        Get the degree of a node for outbound edges.

        :param int node: The index of the node to find the outbound degree of

        :returns degree: The outbound degree for the specified node
        :rtype: int

    .. py:method:: remove_edge(self, parent, child):
        Remove an edge between 2 nodes.

        Note if there are multiple edges between the specified nodes only one
        will be removed.

        :param int parent: The index for the parent node.
        :param int child: The index of the child node.

        :raises NoEdgeBetweenNodes: If there are no edges between the nodes
            specified

    .. py:method:: remove_edge_from_index(self, edge):
        Remove an edge identified by the provided index

        :param int edge: The index of the edge to remove

.. _petgraph: https://github.com/bluss/petgraph

.. py:function:: dag_longest_path(graph):
    Find the the longest path in a graph.

    :param PyDAG graph: The graph to find the longest path on

    :returns path: The node indexes of the longest path on the graph
    :rtype: list

    :raises Exception: If an unexpected error occurs and a path can't be found

.. py:function:: dag_longest_path_length(graph):
    Find the length of the longest path in a graph.

    :param PyDAG graph: The graph to find the longest path on

    :returns length: The longest path length on the graph
    :rtype: int

    :raises Exception: If an unexpected error occurs and a path can't be found

.. py:function:: number_weakly_connected_components(graph):
    Find the number of weakly connected components in a DAG.

    :param PyDAG graph: The graph to find the number of weakly connected
        components on

    :returns number: The number of weakly connected components in the DAG
    :rtype: int

.. py:function:: is_directed_acyclic_graph(graph):
    Check that the DAG doesn't have a cycle (should always return True)

    :param PyDAG graph: The graph to check for cycles

    :returns is_dag: True if there are no cycles and False if a cycle is found
    :rtype: bool

.. py:function:: is_isomorphic(first, second):
    Determine if 2 DAGS are structurally isomorphic.

    This checks if 2 graphs are structurally isomorphic (it doesn't match
    the contents of the nodes or edges on the dag).

    :param PyDAG first: The first DAG to compare
    :param PyDAG second: The second DAG to compare

    :returns is_isomorphic: True if the 2 PyDAGs are structurally isomorphic
        False if they are not.
    :rtype: bool

.. py:function:: is_isomorphic_node_match(first, second, matcher):
    Determine if 2 DAGS are structurally isomorphic.

    This checks if 2 graphs are isomorphic both structurally and also comparing
    the node data using the provided matcher function. The matcher function
    takes in 2 node data objects and will compare them. A simple example that
    checks if they're just equal would be::

        graph_a = retworkx.PyDAG()
        graph_b = retworkx.PyDAG()
        retworkx.is_isomorphic_node_match(graph_a, graph_b,
                                          lambda x, y: x == y)

    :param PyDAG first: The first DAG to compare
    :param PyDAG second: The second DAG to compare
    :param function matcher: A python callable object that takes 2 positional
        arguments one for each node data object. If the return of this
        function evaluates to True then the nodes passed to it are vieded as
        matching.

    :returns is_isomorphic: True if the 2 PyDAGs are isomorphic
        False if they are not.
    :rtype: bool

.. py:function:: topological_sort(graph):
    Return the topological sort of node indexes from the provided graph

    :param PyDAG graph: The DAG to get the topological sort on

    :returns nodes: A list of node indexes topologically sorted.
    :rtype: list

    :raises DAGHasCycle: if a cycle is encountered while sorting the graph

.. py:function:: lexicogrpahical_topological_sort(dag, key):
    Get the lexicographical topological sorted nodes' data from the provided dag

    This function returns a list of nodes in a graph lexicographically
    topologically sorted using the provided key function.

    :param PyDAG dag: The DAG to get the topological sorted nodes from
    :param function key: Takes in a python function or other callable that
        gets passed a single argument the node data from the graph and is
        expected to return a string.

    :returns nodes: A list of node's data lexicographically topologically
        sorted.
    :rtype: list

.. py:function:: ancestors(graph, node):
    Return the ancestors of a node in a graph.

    This differs from :py:meth:`PyDAG.predecessors` method  in that
    predecessors returns only nodes with a direct edge into the provided node.
    While this function returns all nodes that have a path into the provided
    node.

    :param PyDAG graph: The DAG to get the descendants from
    :param int node: The index of the dag node to get the ancestors for

    :returns nodes: A list of node indexes of ancestors of provided node.
    :rtype: list

.. py:function:: descendants(graph, node):
    Return the descendants of a node in a graph.

    This differs from :py:meth:`PyDAG.successors` method in that
    predecessors returns only nodes with a direct edge out of the provided node.
    While this function returns all nodes that have a path from the provided
    node.

    :param PyDAG graph: The DAG to get the descendants from
    :param int node: The index of the dag node to get the descendants for

    :returns nodes: A list of node indexes of descendants of provided node.
    :rtype: list

.. py:function:: bfs_successors(graph, node):
    Return successors in a breadth-first-search from a source node.

    The return format is [(Parent Node, [Children Nodes])] in a bfs order from
    the source node provided.

    :param PyDAG graph: The DAG to get the bfs_successors from
    :param int node: The index of the dag node to get the bfs successors for

    :returns nodes: A list of nodes's data and their children in bfs order
    :rtype: list

.. py:function:: floyd_warshall(graph):
    Return the shortest path lengths between every pair of nodes that has a
    path connecting them.

    The runtime is :math:`O(|N|^3 + |E|)` where :math:`|N|` is the number
    of nodes and :math:`|E|` is the number of edges.

    This is done with the Floyd Warshall algorithm:
    
    1. Process all edges by setting the distance from the parent to
       the child equal to the edge weight.
    2. Iterate through every pair of nodes (source, target) and an additional
       itermediary node (w). If the distance from source :math:`\rightarrow` w
       :math:`\rightarrow` target is less than the distance from source
       :math:`\rightarrow` target, update the source :math:`\rightarrow` target
       distance (to pass through w).

    The return format is ``{Source Node: {Target Node: Distance}}``.

    .. note::

        Paths that do not exist are simply not found in the return dictionary,
        rather than setting the distance to infinity, or -1.

    .. note::

        Edge weights are restricted to 1 in the current implementation.

    :param PyDAG graph: The DAG to get all shortest paths from

.. py:function:: layers(graph, first_layer):
    Return a list of layers

    A layer is a subgraph whose nodes are disjoint, i.e.,
    a layer has depth 1. The layers are constructed using a greedy algorithm.

    :param PyDAG graph: The DAG to get the layers from
    :param list first_layer: A list of node ids for the first layer. This
        will be the first layer in the output

    :returns layers: A list of layers, each layer is a list of node data
    :rtype: list
