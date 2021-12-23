########################
Introduction to retworkx
########################

This guide serves as an introduction to working with retworkx. If you're a
current or past `NetworkX <https:://networkx.org>`__ user who is looking at
using retworkx as a replacement for NetworkX, you can also refer to
:ref:`networkx` for a detailed guide comparing using retworkx with NetworkX.

Creating a Graph
================

To create an empty undirected graph :math:`G` with no nodes and no edges you
can run the following code:

.. jupyter-execute::

    import retworkx as rx
    G = rx.PyGraph()

A :class:`~retworkx.PyGraph` is an collection of nodes (vertices)
along with unordered pairs of nodes (called edges, links, etc). Nodes and
edges in retworkx are assigned any Python object (e.g. a numeric value, a
string, an image, an xml object, another graph, a custom node object, etc)
for a data payload (also referred to as a weight in the API and
documentation). Nodes and edges are uniquely identified by an integer ``index``
which is stable for the lifetime of that node or edge in the graph. These
indices are not guaranteed to be a contiguous sequence as removed nodes or
edges will leave holes in the set of indices and an index can be reused after
a removal.

Nodes
=====

The graph :math:`G` can be grown in several ways, there are graph generator
functions and functions to read and write graphs in different foramts.

To get started we can add a single node:

.. jupyter-execute::

    G.add_node(1)

The :meth:`~retworkx.PyGraph.add_node` method returns an integer. This integer
is the index used to uniquely identify this new node in the graph. You can use
this index to identify the node as long as node remains in the graph.

It is also possible to add nodes to :math:`G` from any
`sequence <https://docs.python.org/3/glossary.html#term-sequence>`__ of
elements such as a
`list <https://docs.python.org/3/library/stdtypes.html#list>`__ or
`range <https://docs.python.org/3/library/stdtypes.html#ranges>`__:

.. jupyter-execute::

    indices = G.add_nodes_from(range(5))
    print(indices)

Just as with :meth:`~retworkx.PyGraph.add_node` the
:meth:`~retworkx.PyGraph.add_nodes` method returns the indices for the nodes
added as a :class:`~retworkx.NodeIndices` object, which is a custom sequence
type that contains the index of each node in the order it's added from the input
sequence.

In the above cases we were adding nodes with a data payload integers (``1`` in
the single :meth:`~retworkx.PyGraph.add_node` case and the sequence ``0``,
``1``, ``2``, ``3``, ``4`` in the :meth:`~retworkx.PyGraph.add_nodes_from` case)
however retworkx doesn't place constraints on what the node data payload can
be so you can use more involved objects including types which are not
`hashable <https://docs.python.org/3/glossary.html#term-hashable>`__. For
example, we can add a node with a data payload that's a a
`dict <https://docs.python.org/3/library/stdtypes.html#dict>`__:

.. jupyter-execute::

    G.add_node({
        "color": "green",
        "size": 42,
    })

A discussion of how to select what to use for your data payload is in the
:ref:`data_payload` section.

Edges
=====

The graph :math:`G` can also be grown by adding one edge at a time

.. jupyter-execute::

    G.add_edge(1, 2, None)

This will add an edge between node index ``1`` and node index ``2`` with the
data payload of None. Just as with :meth:`~retworkx.PyGraph.add_node` the
:meth:`~retworkx.PyGraph.add_edge` method returns the edge index used to
uniquely identify

Examining elements of a graph
=============================

We can examine the nodes and edges of a graph in retworkx fairly easily. The
first thing to do is get a list of node and edge indices using
:meth:`~retworkx.PyGraph.node_indices` and
meth:`~retworkx.PyGraph.edge_indices`:

.. jupyter-execute::

    node_indices = G.node_indices()
    edge_indices = G.edge_indices()
    print(node_indices)
    print(edge_indices)

Since indices are the unique identifiers for nodes and edges they're your
handle to elements in the graph. This is especially important for edges in the
cases of multigraphs or where you have identical data payloads between multiple
nodes. You can use the indices to access the data payload. For nodes the
:class:`~retworkx.PyGraph` object behaves like a
`mapping <https://docs.python.org/3/glossary.html#term-mapping>` with the
index:

.. jupyter-execute::

    first_index_data = G[node_indices[0]]
    print(first_index_data)

For edges, you can use the :meth:`~retworkx.PyGraph.get_edge_data_by_index`
method to access the data payload for a given edge and
::meth:`~retworkx.PyGraph.get_edge_endpoints_by_index` to get the endpoints
of a given edge from its index:

.. jupyter-execute::

    first_index_data = G.get_edge_data_by_index(edge_indices[0])
    first_index_edgepoints = G.get_edge_endpoints_by_index(edge_indices[0])
    print(first_index_edgepoints)
    print(first_index_data)

For edges since we don't implement the mapping protocol there is also a helper
method available to get the mapping of edge indices to the edge endpoints and
data payloads, :meth:`~retworkx.PyGraph.edge_index_map`:

.. jupyter-execute::

    print(G.edge_index_map())

Additionally, you can access the list of node and edge data payloads directly
with :meth:`~retworkx.PyGraph.nodes` and :meth:`~retworkx.PyGraph.edges`

.. jupyter-execute::

    print("Node data payloads")
    print(G.nodes())
    print("Edge data payloads")
    print(G.edges())

.. _tutorial_removal:

Removing elements from a graph
===============================

You can remove a node or edge from a graph in a similar manner to adding
elements to the graph. There are methods :meth:`~retworkx.PyGraph.remove_node`,
:meth:`~retworkx.PyGraph.remove_nodes_from`,
:meth:`~retworkx.PyGraph.remove_edge`, :meth:`~remove_edge_from_index`, and
:meth:`~retworkx.PyGraph.remove_edges_from` to remove nodes and edges from
the graph. One thing to note is that on removal there can be holes in the
list of indices for nodes and/or edges in the graph. For example:

.. jupyter-execute::

    import retworkx

    graph = retworkx.PyGraph()
    graph.add_nodes_from(list(range(5)))
    graph.add_nodes_from(list(range(2)))
    graph.remove_node(2)
    print(graph.node_indices())

You can see here that the indices for the nodes in ``graph`` are missing ``2``.
Also, after a removal the index of the removed node or edge will be reused on
subsequent additions. For example, building off the previous example if you ran

.. jupyter-execute::

    graph.add_node("New Node")

this new node is assigned index 2 again.

.. _data_payload:

What to use for node and edge data payload
==========================================

In the above examples for the most part we use integers, strings, and ``None``
for the data payload of nodes and edges in graphs (mostly for simplicity).
However, retworkx allows the use of any Python object as the data payload for
nodes and edges. This flexibility to use any python object is very powerful as
it allows you to create graphs that contain other graphs, graphs that contain
files, graphs with functions, etc. This means you only need to keep a reference
to the integer index returned by retworkx for the objects you use as a data
payloads to find those objects in the graph. For example, one approach you can
take is to store the index as an attribute on the object you add to the graph:

.. jupyter-execute::

    class GraphNode:

        def __init__(self, value):
            self.value = value
            self.index = None

    graph = rx.PyGraph()
    index = graph.add_node(GraphNode("A"))
    graph[index].index = index

Additionally, at any time you can find the index mapping to the data payload
and build a mapping or update a reference to it. For example, building on the
above example you can update the index references all at once after creation:

.. jupyter-execute::

    class GraphNode:
        def __init__(self, value):
            self.index = None
            self.value = value

        def __str__(self):
            return f"GraphNode: {self.value} @ index: {self.index}"

    class GraphEdge:
        def __init__(self, value):
            self.index = None
            self.value = value

        def __str__(self):
            return f"EdgeNode: {self.value} @ index: {self.index}"

    graph = rx.PyGraph()
    graph.add_nodes_from([GraphNode(i) for i in range(5)])
    graph.add_edges_from([(i, i + 1, GraphEdge(i)) for i in range(4)])
    # Populate index attribute in GraphNode objects
    for index in graph.node_indices():
        graph[index].index = index
    # Populate index attribute in GraphEdge objects
    for index, data in graph.edge_index_map().items():
        data[2].index = index
    print("Nodes:")
    for node in graph.nodes():
        print(node)
    print("Edges:")
    for edge in graph.edges():
        print(edge)

Accessing edges and neighbors
=============================

You can access edges from a node using the :meth:`~rx.PyGraph.incident_edges`
method:

.. jupyter-execute::

    print(G.incident_edges(2))

which will return the edge indices of the edges incident to node ``2``. You
can also find the neighbor nodes using the :meth:`~rx.PyGraph.neighbors`
method:

.. jupyter-execute::

    print(G.neighbors(2))

which returns the node indices of any neighbors of node ``2``.

Directed Graphs
===============

A directed graph is a graph that is made up of a set of nodes connected by
directed edges (often called arcs). Edges have a directionality which is
different from undirected graphs where edges have no notion of a direction to
them. In retworkx the :class:`~retworkx.PyDiGraph` class is used to create
directed graphs. For example:

.. jupyter-execute::

    import retworkx as rx
    from retworkx.visualization import mpl_draw

    path_graph = rx.generators.directed_path_graph(5)
    mpl_draw(path_graph)

In this example we created a 5 node directed path graph. This shows the
directionality of the edges in the graph visualization with the arrow head
pointing to the target node.

Multigraphs
===========

By default all graphs in retworkx are multigraphs. This means that each
graph object can contain parallel edges between nodes. However, you can set
the ``multigraph`` argument to ``False`` on the :class:`~retworkx.PyGraph` and
:class:`~retworkx.PyDiGraph` constructors when creating a new graph object will
not allow you to add parallel edges. When ``multigraph`` is set to ``False`` if
a method call is made that would add a parallel edge it will instead update the
existing edge’s weight/data payload. For example:

.. jupyter-execute::

    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(3))
    graph.add_edges_from([(0, 1, 'A'), (0, 1, 'B'), (1, 2, 'C')])
    mpl_draw(graph, with_labels=True, edge_labels=str)

In this example when we attempted to add a parallel edge between nodes ``0``
and ``1`` the edge data payload is updated from ``'A'`` to ``'B'``.

Graph Generators and operations
===============================

In addition to constructing graphs one node and edge at a time, you can also
create graphs using the :ref:`generator_funcs`, :ref:`random_generators`,
and :ref:`graph-ops` to quickly generate graphs and/or apply different
operations on the graph. For example:

.. jupyter-execute::

    lolipop_graph = rx.generators.lollipop_graph(4, 3)
    mesh_graph = rx.generators.mesh_graph(4)
    combined_graph = rx.cartesian_product(lolipop_graph, mesh_graph)[0]
    mpl_draw(combined_graph)

Additionally there are alternate constructors such as
:meth:`~retworkx.read_edge_list` or :meth:`~retworkx.from_adjacency_matrix`
for building graphs from files or other inputs. For example:

.. jupyter-execute::

    import tempfile

    with tempfile.NamedTemporaryFile('wt') as fd:
        path = fd.name
        fd.write('0 1\n')
        fd.write('0 2\n')
        fd.write('0 3\n')
        fd.write('1 2\n')
        fd.write('2 3\n')
        fd.flush()
        graph = rx.PyGraph.read_edge_list(path)
    mpl_draw(graph)

Analyzing graphs
================

The structure of a graph :math:`G` can be analyzed using the graph algorithm
functions. For example:

.. jupyter-execute::

    G = rx.PyGraph()
    G.extend_from_edge_list([(0, 1), (0, 2)])
    new_node = G.add_node("spam")
    print(rx.connected_components(G))
    degrees = []
    for node in G.node_indices():
        degrees.append(G.degree(node))
    print(degrees)

.. jupyter-execute::

    G.remove_node(new_node)
    G.extend_from_edge_list([(0, 3), (0, 4), (1, 2)])
    rx.transitivity(G)

See the :ref:`algorithm_api` API documentation section for a list of example
functions and how to use them.

Drawing graphs
==============

There are two visualization functions provided in retworkx for visualizing
graphs. First there is :func:`~retworkx.visualization.mpl_draw` which uses the
`matplotlib <https://matplotlib.org/>`__ library to render the
visualization of the graph. The :func:`~retworkx.visualization.mpl_draw`
function relies on the :ref:`layout-functions` provided with retworkx to
generate a layout (the coordinates to draw the nodes of the graph) for the
graph (by default :func:`~retworkx.spring_layout` is used). For example:

.. jupyter-execute::

    import matplotlib.pyplot as plt

    G = rx.generators.generalized_petersen_graph(5, 2)
    subax1 = plt.subplot(121)
    mpl_draw(G, with_labels=True, ax=subax1)
    subax2 = plt.subplot(122)
    layout = rx.shell_layout(G, nlist=[range(5, 10), range(5)])
    mpl_draw(G, pos=layout, with_labels=True, ax=subax2)

The second function is :func:`~retworkx.visualization.graphviz_draw` which
uses `Graphviz <https://graphviz.org/>`__ to generate visualizations. For
example:

.. jupyter-execute::

    from retworkx.visualization import graphviz_draw

    G = rx.generators.heavy_hex_graph(7)
    # set data payload to index
    for node in G.node_indices():
        G[node] = node

    def node_attr_fn(node):
        attr_dict = {
            "style": "filled",
            "shape": "circle",
            "label": str(node)
        }
        # Data nodes are yellow
        if node < 7 * 7:
            attr_dict["color"] = "yellow"
            attr_dict["fill_color"] = "yellow"
        # Syndrome nodes are black
        elif node >= 7 * 7 and node < (7 * 7) + ((7 - 1) * (7 + 1) / 2):
            attr_dict["color"] = "black"
            attr_dict["fill_color"] = "black"
            attr_dict["fontcolor"] = "white"
        # Flag quits are blue
        else:
            attr_dict["color"] = "blue"
            attr_dict["fill_color"] = "blue"
        return attr_dict

    graphviz_draw(G, node_attr_fn=node_attr_fn, method="neato")

Generally when deciding which visualization function to use there are a few
considerations to make. :func:`~retworkx.visualization.mpl_draw` is a better
choice for smaller graphs or cases where you want to integrate your graph
drawing as part of a larger visualization.
:func:`~retworkx.visualization.graphviz_draw` is typically a better choice
for larger graphs because Graphviz is a dedicated tool for drawing graphs.
