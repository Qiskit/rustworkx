###########################
retworkx for networkx users
###########################

This is an introductory guide for existing networkx users on how to use
retworkx, how it differs from networkx, and key differences to keep in mind.

Some Key Differences
====================

retworkx (as the name implies) was inspired by networkx and the goal of the
project is to provide a similar level of functionality and utility to what
networkx offers but with much faster performance. However, because of
limitations in the boundary between rust and python, different design
decisions, and other differences the libraries are not identical.

The biggest difference to keep in mind is networkx is a very dynamic in how it
can be used. It allows you to treat a graph object associatively (like a python
dictionary) and interact with the graph using the objects you're putting
on the graph. For example::

    import networkx as nx
    
    graph = nx.MultiDiGraph()
    graph.add_node('my_node_a')
    graph.add_node('my_node_b')
    graph.add_edge('my_node_a', 'my_node_b')

While retworkx being written in Rust puts more constraints on how
you interact with graph objects. With retworkx you can still attach any Python
object on the a graph but each node and edge is assigned an integer index.
That index must be used for accessing nodes and edges on the graph.
In retworkx the above example would be something like::

    import retworkx as rx
    
    graph = rx.PyDiGraph()
    node_a = graph.add_node('my_node_a')
    node_b = graph.add_node('my_node_b')
    graph.add_edge(node_a, node_b, None)

where ``node_a == 0`` and ``node_b == 1``. These node indices can be used with a
graph object to access the objects set as the payload object via the python
mapping protocol (**not** the sequence protocol because the indices are not
guaranteed to be a sequence after nodes or edges are removed from a graph). Continuing
from the above retworkx example::

    assert 'my_node_a' == graph[node_a]
    assert 'my_node_b' == graph[node_b]

The use of integer indexes for everything is normally the biggest difference that
existing networkx users have to adapt to when migrating to retworkx.

Similarly when there are algorithm functions that operate on a node or edge
data, callback functions are used in retworkx to return statically typed data
from node or edge payloads to use for various algorithms. In networkx this is
typically done using named attributes of nodes or edges (the typical example of
a node or edge attribute named ``weight`` is used by default for functions that
need a numerical weight).

For example, in networkx::

    import networkx as nx
    
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(0, 1, {'weight': 1}), (0, 2, {'weight': 2}),
                          (1, 3, {'weight': 2}), (3, 0, {'weight': 3})])
    dist_matrix = nx.floyd_warshall_numpy(graph, weight='weight')
    
while in retworkx you would use::
    
    import retworkx as rx
    
    graph = rx.PyDiGraph()
    graph.extend_from weighted_edge_list(
        [(0, 1, {'weight': 1}), (0, 2, {'weight': 2}),
         (1, 3, {'weight': 2}), (3, 0, {'weight': 3})])
    dist_matrix = rx.digraph_floyd_warshall_numpy(
        graph, weight_fn=lambda edge: edge[weight])

or more concisely::

    import retworkx as rx
    
    graph = rx.PyDiGraph()
    graph.extend_from weighted_edge_list(
        [(0, 1, 1), (0, 2, 2),
         (1, 3, 2), (3, 0, 3)])
    dist_matrix = rx.digraph_floyd_warshall_numpy(graph,
                                                  weight_fn=lambda edge: edge)

The other large difference to keep in mind is that most functions in retworkx
are explicitly typed. This means that they either always return or accept
either a :class:`~retworkx.PyDiGraph` or a :class:`~retworkx.PyGraph` but not
both. The exception to this are the :ref:`universal-functions` which will
dispatch to the statically typed equivalent based on the object they receive.
This is different from networkx where everything is pretty much dynamically
typed and you can pass a graph object to any function and it will work as
expected (unless it isn't supported and then it will raise an exception).

Graph Data and Attributes
=========================


Nodes
-----

In networkx a node can be any hashable python object. That object is then used
to access or refer to a node. Additionally, you can set optional attributes
on a node. This is described in more detail below.

In retworkx any python object (hashable or not) can be used as a node, however
nodes can only be accessed by an integer node index (which is returned by any
function adding a node). There are no optional attributes for nodes. If this
is required that is expected to be added to the node's data payload.

Edges
-----

Edges in networkx are accessible by the tuple of the nodes the edge is between.
Edges only have optional attributes (as described below) and no other object 
payload.

In retworkx any python object can be an edge and have a unique integer index
assigned to it, just like nodes. However, edges are in most functions/methods
referenced by the tuple of the indices of the nodes the edge is between
instead of the edge's index.


Attributes
----------

networkx has a concept of
`graph <https://networkx.org/documentation/stable/tutorial.html#graph-attributes>`__,
`node <https://networkx.org/documentation/stable/tutorial.html#node-attributes>`__,
and `edge attributes <https://networkx.org/documentation/stable/tutorial.html#edge-attributes>`__
in addition to the hashable object used for a node's payload. Retworkx
has no analogous concept. Instead, the payloads for nodes and edges are any 
python object (hashable or not). This enables you to build similar structures 
to the attributes concept, but also use alternative structures specific to 
your use case.

For example, something like::

    import networkx as nx

    graph = nx.Graph()
    graph.add_node(1, time='5pm')
    graph.add_nodes_from([3], time='2pm')
    graph.nodes[1]['room'] = 714

can be accomplished by using a ``dict`` for node weights::

    import retworkx as rx

    graph = rx.PyGraph()
    node_a = graph.add_node({'time': '5pm'})
    node_b = graph.add_nodes_from([{'time': '2pm'}])
    graph[node_a]['room'] = 714

Examining elements of a graph
-----------------------------

networkx provides 4 attributes on graph objects ``nodes``, ``edges``, ``adj``,
and ``degree`` which act as set like views for the nodes, edges, neighbors, and
degrees of nodes respectively. These properties provide a real time view into
the different properties of the graphs and provide additional methods on those
attributes for looking at graph properties in different ways.

retworkx doesn't offer views, but instead provides different accessor methods
that return copies of the analogous data. There are different functions/methods
that offer different views on that data. For example,
:meth:`~retworkx.PyDiGraph.edge_list` is analogous to networkx's ``edges`` view
and :meth:`~retworkx.PyDiGraph.weighted_edge_list` is equivalent to networkx's
``edges(data=True)``.

Additionally, since everything in retworkx is integer indexed, to access node
data the :class:`~retworkx.PyDiGraph` and :class:`~retworkx.PyGraph` classes
implement the python mapping protocol so you can access node's data using a
node's index.

API Equivalents
===============

Class Constructors
------------------

.. list-table::
   :header-rows: 1

   * - networkx
     - retworkx
     - Notes
   * - ``Graph()``
     - :class:`PyGraph(multigraph=False) <retworkx.PyGraph>`
     - Only in multigraph flag added in retworkx>= 0.8.0 prior releases
       always allow multiple edges
   * - ``DiGraph()``
     - :class:`PyDiGraph(multigraph=False) <retworkx.PyDiGraph>`
     - Only in multigraph flag added in retworkx>= 0.8.0 prior releases
       always allow multiple edges
   * - ``MultiGraph()``
     - :class:`PyGraph() <retworkx.PyGraph>`
     -
   * - ``MultiDiGraph()``
     - :class:`PyDiGraph() <retworkx.PyDiGraph>`
     -

The other thing to note here is that retworkx does not allow initialization
of a graph when the constructor is called. You will need to call an appropriate
method of the object to add nodes or edges or use an alternative constructor
method:

.. list-table::
   :header-rows: 1

   * - networkx
     - retworkx
     - Notes
   * - .. code-block::

         Graph([(0, 1), (1, 0)])

     - .. code-block::

         graph = PyGraph()
         graph.extend_from_edge_list([(0, 1), (1, 0)])

     - retworkx input must be a list of 2-tuples, while networkx can be an
       iterator
   * - .. code-block::

         Graph([(0, 1, {'weight': 2}), (1, 0, {'weight': 1})])

     - .. code-block::

         graph = PyGraph()
         graph.extend_from_edge_list([(0, 1, 2), (1, 0, 1)])

     - retworkx input must be a list of 3-tuples, while networkx can be an
       iterator
   * - .. code-block::

        Graph(np.array([[0, 1, 1], [1, 0, 1], [1, 0, 1]]))

     - .. code-block::

        PyGraph.from_adjacency_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=np.float64))

     - retworkx :meth:`~retworkx.PyDiGraph.from_adjacency_matrix` can only take
       a float dtype numpy array, you can use
       ``.astype(np.float64, copy=False)`` to adapt a non-float array.

Graph Modifiers
---------------

.. list-table::
   :header-rows: 1
 
   * - networkx
     - retworkx
     - Notes
   * - ``add_node()``
     - :meth:`~retworkx.PyDiGraph.add_node`
     - retworkx returns a node index for the newly created node
   * - ``add_nodes_from``
     - :meth:`~retworkx.PyDiGraph.add_nodes_from`
     - retworkx requires the input to be a list of objects and will return a
       list of node indices for the newly created nodes
   * - ``add_edge``
     - :meth:`~retworkx.PyDiGraph.add_edge`
     - retworkx requires 3 parameters be used, the 2 node indices and the payload
       (networkx works with either 2 or 3)
   * - ``add_edges_from``
     - :meth:`~retworkx.PyDiGraph.add_edges_from`,
       :meth:`~retworkx.PyDiGraph.add_edges_from_no_data`,
       :meth:`~retworkx.PyDiGraph.extend_from_edge_list`,
       :meth:`~retworkx.PyDiGraph.extend_from_weighted_edge_list`
     - retworkx requires a list of either a 3 or 2 tuple (depending on whether
       weights/data are expected or not). The difference between the retworkx
       ``extend_from*`` and ``add_edges_from*`` methods are that the
       ``extend_from*`` will create new nodes with a weight/data payload of
       ``None`` if any node indices are missing.

(note the retworkx version links to the :class:`~retworkx.PyDiGraph` version,
but there are also equivalent :class:`~retworkx.PyGraph` methods available)

.. _networkx_converter:

Converting from a networkx graph
================================

If you're using a function or an external library that is already generating a
networkx graph then you can use :func:`retworkx.networkx_converter` to convert
that networkx ``Graph`` object into an equivalent retworkx
:class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph` object. Note that
networkx is **not** a dependency for retworkx and you are responsible for
installing networkx to use this function. Accordingly, there is not equivalent
function provided to convert the reverse direction (because doing so would add
an unwanted dependency on networkx, even an optional one) but writing such a
function is straightforward, for example::

    import networkx as nx
    import retworkx as rx


    def convert_retworkx_to_networkx(graph):
        """Convert a retworkx PyGraph or PyDiGraph to a networkx graph."""
        edge_list = [(
            graph[x[0]], graph[x[1]],
            {'weight': x[2]}) for x in graph.weighted_edge_list()]

        if isinstance(graph, rx.PyGraph):
            if graph.multigraph:
                return nx.MultiGraph(edge_list)
            else:
                return nx.Graph(edge_list)
        else:
            if graph.multigraph:
                return nx.MultiDiGraph(edge_list)
            else:
                return nx.DiGraph(edge_list)


Functionality Gaps
==================

networkx is a mature library that has a wide user base and extensive feature set,
while retworkx, by comparison, is a much younger library and is missing a lot
of the features that networkx offers. If you encounter a feature that networkx
offers which is missing from retworkx that you would like to use please open an
"Enhancement request" issue at: https://github.com/Qiskit/retworkx/issues/new/choose
Once an issue is opened we can prioritize working on adding an equivalent
feature to retworkx.
