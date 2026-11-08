===================================
Working with Betweenness Centrality
===================================

The betweenness centrality of a graph is a measure of centrality based on shortest
paths. For every pair of nodes in a connected graph, there is at least a single
shortest path between the nodes such that the number of edges the path passes
through is minimized. The betweenness centrality for a given graph node is the number
of these shortest paths that pass through the node.

This is defined as:

.. math::

    c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

where :math:`V` is the set of nodes, :math:`\sigma(s, t)` is the number of
shortest :math:`(s, t)` paths, and :math:`\sigma(s, t|v)` is the number of
those paths passing through some  node :math:`v` other than :math:`s, t`.
If :math:`s = t`, :math:`\sigma(s, t) = 1`, and if :math:`v \in {s, t}`,
:math:`\sigma(s, t|v) = 0`

This tutorial will take you through the process of calculating the betweenness
centrality of a graph and visualizing it.

Generate a Graph
----------------

To start we need to generate a graph:

.. jupyter-execute::

    import rustworkx as rx
    from rustworkx.visualization import mpl_draw

    graph = rx.generators.hexagonal_lattice_graph(4, 4)
    mpl_draw(graph)


Calculate the Betweeness Centrality
-----------------------------------

The :func:`~rustworkx.betweenness_centrality` function can be used to calculate
the betweenness centrality for each node in the graph.

.. jupyter-execute::

    import pprint

    centrality = rx.betweenness_centrality(graph)
    # Print the centrality value for the first 5 nodes in the graph
    pprint.pprint({x: centrality[x] for x in range(5)})

The output of :func:`~rustworkx.betweenness_centrality` is a
:class:`.CentralityMapping` which is a custom
`mapping <https://docs.python.org/3/glossary.html#term-mapping>`__ type that
maps the node index to the centrality value as a float. This is a mapping and
not a sequence because node indices (and edge indices too, which is not
relevant here) are not guaranteed to be a contiguous sequence if there are
removals.

Visualize the Betweenness Centrality
------------------------------------

Now that we've found the betweenness centrality for ``graph`` lets visualize it.
We'll color each node in the output visualization using its calculated
centrality:

.. jupyter-execute::

    import matplotlib.pyplot as plt

    # Generate a color list
    colors = []
    for node in graph.node_indices():
        colors.append(centrality[node])
    # Generate a visualization with a colorbar
    plt.rcParams['figure.figsize'] = [15, 10]
    ax = plt.gca()
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(
        vmin=min(centrality.values()),
        vmax=max(centrality.values())
    ))
    plt.colorbar(sm, ax=ax)
    plt.title("Betweenness Centrality of a 4 x 4 Hexagonal Lattice Graph")
    mpl_draw(graph, node_color=colors, ax=ax)

Alternatively, you can use :func:`~rustworkx.visualization.graphviz_draw`:

.. jupyter-execute::

    from rustworkx.visualization import graphviz_draw
    import matplotlib

    # For graphviz visualization we need to assign the data payload for each
    # node to its centrality value so that we can color based on this
    for node, btw in centrality.items():
        graph[node] = btw

    # Leverage matplotlib for color map
    colormap = matplotlib.colormaps["magma"]
    norm = matplotlib.colors.Normalize(
        vmin=min(centrality.values()),
        vmax=max(centrality.values())
    )

    def color_node(node):
        rgba = matplotlib.colors.to_hex(colormap(norm(node)), keep_alpha=True)
        return {
            "color": f"\"{rgba}\"",
            "fillcolor": f"\"{rgba}\"",
            "style": "filled",
            "shape": "circle",
            "label": "%.2f" % node,
        }

    graphviz_draw(graph, node_attr_fn=color_node, method="neato")
