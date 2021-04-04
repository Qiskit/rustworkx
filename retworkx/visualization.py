# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# NetworkX is distributed with the 3-clause BSD license.
#
#   Copyright (C) 2004-2020, NetworkX Developers
#   Aric Hagberg <hagberg@lanl.gov>
#   Dan Schult <dschult@colgate.edu>
#   Pieter Swart <swart@lanl.gov>
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of the NetworkX Developers nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This code is forked from networkx's networkx_pylab.py module and adapted to
# work with retworkx instead. The original source can be found at:
#
# https://github.com/retworkx/retworkx/blob/80b1afa2ae50314a8312998c214a8c1a356adcf1/retworkx/drawing/retworkx_pylab.py

"""Draw a retworkx graph with matplotlib."""

from collections.abc import Iterable
from itertools import islice, cycle
from numbers import Number

import numpy as np

import retworkx


__all__ = [
    "draw",
    "draw_retworkx",
    "draw_retworkx_nodes",
    "draw_retworkx_edges",
    "draw_retworkx_labels",
    "draw_retworkx_edge_labels",
]


def draw(graph, pos=None, ax=None, **kwds):
    """Draw a graph with Matplotlib.

    Draw the graph as a simple representation with no node
    labels or edge labels and using the full Matplotlib figure area
    and no axis labels by default.  See draw_retworkx() for more
    full-featured drawing that allows title, axis labels etc.

    Parameters
    ----------
    graph: A retworkx graph

    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a spring layout positioning will be computed.
        See :py:mod:`retworkx.drawing.layout` for functions that
        compute node positions.

    ax : Matplotlib Axes object, optional
        Draw the graph in specified Matplotlib axes.

    kwds : optional keywords
        See retworkx.draw_retworkx() for a description of optional keywords.

    With pyplot use::

        import matplotlib.pyplot as plt
        G = retworkx.directed_random_gnp_graph()
        retworkx.draw(G)  # retworkx draw()
        plt.draw()  # pyplot draw()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running "
                          "retworkx.draw_retworkx_nodes(). You can install "
                          "matplotlib with:\n'pip install matplotlib'") from e
    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor("w")
    if ax is None:
        if cf._axstack() is None:
            ax = cf.add_axes((0, 0, 1, 1))
        else:
            ax = cf.gca()

    if "with_labels" not in kwds:
        kwds["with_labels"] = "labels" in kwds

    draw_retworkx(graph, pos=pos, ax=ax, **kwds)
    ax.set_axis_off()
    plt.draw_if_interactive()
    return cf


def draw_retworkx(graph, pos=None, arrows=True, with_labels=True, **kwds):
    r"""Draw the graph using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    graph: A retworkx :class:`~retworkx.PyDiGraph` or
        :class:`~retworkx.PyGraph`

    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a spring layout positioning will be computed.
        See :mod:`retworkx.drawing.layout` for functions that
        compute node positions.

    arrows : bool (default=True)
        For directed graphs, if True draw arrowheads.
        Note: Arrows will be the same color as edges.

    arrowstyle : str (default='-\|>')
        For directed graphs, choose the style of the arrowsheads.
        See `matplotlib.patches.ArrowStyle` for more options.

    arrowsize : int (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. See `matplotlib.patches.FancyArrowPatch` for attribute
        `mutation_scale` for more info.

    with_labels :  bool (default=True)
        Set to True to draw labels on the nodes.

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    nodelist : list (default=graph.node_indexes())
        Draw only specified nodes

    edgelist : list (default=list(graph.edge_list()))
        Draw only specified edges

    node_size : scalar or array (default=300)
        Size of nodes.  If an array is specified it must be the
        same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as nodelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.

    node_shape :  string (default='o')
        The shape of the node.  Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    alpha : float or None (default=None)
        The node and edge transparency

    cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of nodes

    vmin,vmax : float, optional
        Minimum and maximum for node colormap scaling

    linewidths : scalar or sequence (default=1.0)
        Line width of symbol border

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax
        parameters.

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    style : string (default=solid line)
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    labels : dictionary (default=None)
        Node labels in a dictionary of text labels keyed by node

    font_size : int (default=12 for nodes, 10 for edges)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    label : string, optional
        Label for graph legend

    kwds : optional keywords
        See :func:`retworkx.visualization.draw_retworkx_nodes()`,
        :func:`retworkx.visualization.draw_retworkx_edges()`, and
        :func:`retworkx.visualization.draw_retworkx_labels()` for a
        description of optional keywords.

    Notes
    -----
    For directed graphs, arrows  are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running "
                          "retworkx.draw_retworkx_nodes(). You can install "
                          "matplotlib with:\n'pip install matplotlib'") from e

    valid_node_kwds = (
        "nodelist",
        "node_size",
        "node_color",
        "node_shape",
        "alpha",
        "cmap",
        "vmin",
        "vmax",
        "ax",
        "linewidths",
        "edgecolors",
        "label",
    )

    valid_edge_kwds = (
        "edgelist",
        "width",
        "edge_color",
        "style",
        "alpha",
        "arrowstyle",
        "arrowsize",
        "edge_cmap",
        "edge_vmin",
        "edge_vmax",
        "ax",
        "label",
        "node_size",
        "nodelist",
        "node_shape",
        "connectionstyle",
        "min_source_margin",
        "min_target_margin",
    )

    valid_label_kwds = (
        "labels",
        "font_size",
        "font_color",
        "font_family",
        "font_weight",
        "alpha",
        "bbox",
        "ax",
        "horizontalalignment",
        "verticalalignment",
    )

    valid_kwds = valid_node_kwds + valid_edge_kwds + valid_label_kwds

    if any([k not in valid_kwds for k in kwds]):
        invalid_args = ", ".join([k for k in kwds if k not in valid_kwds])
        raise ValueError(f"Received invalid argument(s): {invalid_args}")

    node_kwds = {k: v for k, v in kwds.items() if k in valid_node_kwds}
    edge_kwds = {k: v for k, v in kwds.items() if k in valid_edge_kwds}
    label_kwds = {k: v for k, v in kwds.items() if k in valid_label_kwds}

    # TODO: switch default to use spring layout when #280 is closed
    if pos is None:
        pos = retworkx.random_layout(graph)  # default to random layout

    draw_retworkx_nodes(graph, pos, **node_kwds)
    draw_retworkx_edges(graph, pos, arrows=arrows, **edge_kwds)
    if with_labels:
        draw_retworkx_labels(graph, pos, **label_kwds)
    plt.draw_if_interactive()


def draw_retworkx_nodes(
    graph,
    pos,
    nodelist=None,
    node_size=300,
    node_color="#1f78b4",
    node_shape="o",
    alpha=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ax=None,
    linewidths=None,
    edgecolors=None,
    label=None,
):
    """Draw the nodes of the graph.

    This draws only the nodes of the graph.

    :param graph: A retworkx graph, either a :class:`~retworkx.PyGraph` or a
        :class:`~retworkx.PyDiGraph`.

    :param dict pos: A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    :param Axes ax: An optional Matplotlib Axes object, if specified it will
        draw the graph in the specified Matplotlib axes.

    :param list nodelist: If specified only draw the specified node indices.
        If not specified all nodes in the graph will be drawn.

    :param float|array node_size: Size of nodes. If an array it must be the
        same length as nodelist. Defaults to 300

    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as nodelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.

    node_shape :  string (default='o')
        The shape of the node.  Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    alpha : float or array of floats (default=None)
        The node transparency.  This can be a single alpha value,
        in which case it will be applied to all the nodes of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    cmap : Matplotlib colormap (default=None)
        Colormap for mapping intensities of nodes

    vmin,vmax : floats or None (default=None)
        Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence] (default=1.0)
        Line width of symbol border

    edgecolors : [None | scalar | sequence] (default = node_color)
        Colors of node borders

    label : [None | string]
        Label for legend

    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.

    """
    try:
        import matplotlib as mpl
        import matplotlib.collections  # call as mpl.collections
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running "
                          "retworkx.draw_retworkx_nodes(). You can install "
                          "matplotlib with:\n'pip install matplotlib'") from e

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = graph.node_indexes()

    # empty nodelist, no drawing
    if len(nodelist) == 0:
        return mpl.collections.PathCollection(None)

    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as e:
        raise IndexError(f"Node {e} has no position.") from e

    if isinstance(alpha, Iterable):
        node_color = apply_alpha(node_color, alpha, nodelist, cmap, vmin, vmax)
        alpha = None

    node_collection = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_size,
        c=node_color,
        marker=node_shape,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
        label=label,
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    node_collection.set_zorder(2)
    return node_collection


def draw_retworkx_edges(
    graph,
    pos,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle=None,
    arrowsize=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    arrows=True,
    label=None,
    node_size=300,
    nodelist=None,
    node_shape="o",
    connectionstyle="arc3",
    min_source_margin=0,
    min_target_margin=0,
):
    r"""Draw the edges of the graph.

    This draws only the edges of the graph.

    Parameters
    ----------
    graph: A retworkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edgelist : collection of edge tuples (default=graph.edge_list())
        Draw only specified edges

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax
        parameters.

    style : string (default=solid line)
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    alpha : float or None (default=None)
        The edge transparency

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    arrows : bool, optional (default=True)
        For directed graphs, if True set default to drawing arrowheads.
        Otherwise set default to no arrowheads. Ignored if `arrowstyle` is set.

        Note: Arrows will be the same color as edges.

    arrowstyle : str (default='-\|>' if directed else '-')
        For directed graphs and `arrows==True` defaults to '-\|>',
        otherwise defaults to '-'.

        See `matplotlib.patches.ArrowStyle` for more options.

    arrowsize : int (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. See `matplotlib.patches.FancyArrowPatch` for attribute
        `mutation_scale` for more info.

    connectionstyle : string (default="arc3")
        Pass the connectionstyle parameter to create curved arc of rounding
        radius rad. For example, connectionstyle='arc3,rad=0.2'.
        See `matplotlib.patches.ConnectionStyle` and
        `matplotlib.patches.FancyArrowPatch` for more info.

    node_size : scalar or array (default=300)
        Size of nodes. Though the nodes are not drawn with this function, the
        node size is used in determining edge positioning.

    nodelist : list, optional (default=graph.node_indexes())
       This provides the node order for the `node_size` array (if it is an
       array).

    node_shape :  string (default='o')
        The marker used for nodes, used in determining edge positioning.
        Specification is as a `matplotlib.markers` marker, e.g. one of
        'so^>v<dph8'.

    label : None or string
        Label for legend

    min_source_margin : int (default=0)
        The minimum margin (gap) at the begining of the edge at the source.

    min_target_margin : int (default=0)
        The minimum margin (gap) at the end of the edge at the target.

    Returns
    -------
    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False or by passing an arrowstyle without
    an arrow on the end.

    Be sure to include `node_size` as a keyword argument; arrows are
    drawn considering the size of nodes.
    """
    try:
        import matplotlib as mpl
        import matplotlib.colors  # call as mpl.colors
        import matplotlib.patches  # call as mpl.patches
        import matplotlib.path  # call as mpl.path
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running "
                          "retworkx.draw_retworkx_nodes(). You can install "
                          "matplotlib with:\n'pip install matplotlib'") from e

    if arrowstyle is None:
        if isinstance(graph, retworkx.PyDiGraph) and arrows:
            arrowstyle = "-|>"
        else:
            arrowstyle = "-"

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = graph.edge_list()

    if len(edgelist) == 0:  # no edges!
        return []

    if nodelist is None:
        nodelist = list(graph.node_indexes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
            np.iterable(edge_color) and (len(edge_color) == len(edge_pos))
            and np.alltrue([isinstance(c, Number) for c in edge_color])):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    # Note: Waiting for someone to implement arrow to intersection with
    # marker.  Meanwhile, this works well for polygons with more than 4
    # sides and circle.

    def to_marker_edge(marker_size, marker):
        if marker in "s^>v<d":  # `large` markers need extra space
            return np.sqrt(2 * marker_size) / 2
        else:
            return np.sqrt(marker_size) / 2

    # Draw arrows with `matplotlib.patches.FancyarrowPatch`
    arrow_collection = []
    mutation_scale = arrowsize  # scale factor of arrow head

    # compute view
    miretworkx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    w = maxx - miretworkx
    h = maxy - miny

    base_connection_style = mpl.patches.ConnectionStyle(connectionstyle)

    # Fallback for self-loop scale. Left outside of _connectionstyle so it is
    # only computed once
    max_nodesize = np.array(node_size).max()

    def _connectionstyle(posA, posB, *args, **kwargs):
        # check if we need to do a self-loop
        if np.all(posA == posB):
            # Self-loops are scaled by view extent, except in cases the extent
            # is 0, e.g. for a single node. In this case, fall back to scaling
            # by the maximum node size
            selfloop_ht = 0.005 * max_nodesize if h == 0 else h
            # this is called with _screen space_ values so covert back
            # to data space
            data_loc = ax.transData.inverted().transform(posA)
            v_shift = 0.1 * selfloop_ht
            h_shift = v_shift * 0.5
            # put the top of the loop first so arrow is not hidden by node
            path = [
                # 1
                data_loc + np.asarray([0, v_shift]),
                # 4 4 4
                data_loc + np.asarray([h_shift, v_shift]),
                data_loc + np.asarray([h_shift, 0]),
                data_loc,
                # 4 4 4
                data_loc + np.asarray([-h_shift, 0]),
                data_loc + np.asarray([-h_shift, v_shift]),
                data_loc + np.asarray([0, v_shift]),
            ]

            ret = mpl.path.Path(ax.transData.transform(path),
                                [1, 4, 4, 4, 4, 4, 4])
        # if not, fall back to the user specified behavior
        else:
            ret = base_connection_style(posA, posB, *args, **kwargs)

        return ret

    # FancyArrowPatch doesn't handle color strings
    arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
    for i, (src, dst) in enumerate(edge_pos):
        x1, y1 = src
        x2, y2 = dst
        shrink_source = 0  # space from source to tail
        shrink_target = 0  # space from  head to target
        if np.iterable(node_size):  # many node sizes
            source, target = edgelist[i][:2]
            source_node_size = node_size[nodelist.index(source)]
            target_node_size = node_size[nodelist.index(target)]
            shrink_source = to_marker_edge(source_node_size, node_shape)
            shrink_target = to_marker_edge(target_node_size, node_shape)
        else:
            shrink_source = shrink_target = to_marker_edge(node_size,
                                                           node_shape)

        if shrink_source < min_source_margin:
            shrink_source = min_source_margin

        if shrink_target < min_target_margin:
            shrink_target = min_target_margin

        if len(arrow_colors) == len(edge_pos):
            arrow_color = arrow_colors[i]
        elif len(arrow_colors) == 1:
            arrow_color = arrow_colors[0]
        else:  # Cycle through colors
            arrow_color = arrow_colors[i % len(arrow_colors)]

        if np.iterable(width):
            if len(width) == len(edge_pos):
                line_width = width[i]
            else:
                line_width = width[i % len(width)]
        else:
            line_width = width

        arrow = mpl.patches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=arrowstyle,
            shrinkA=shrink_source,
            shrinkB=shrink_target,
            mutation_scale=mutation_scale,
            color=arrow_color,
            linewidth=line_width,
            connectionstyle=_connectionstyle,
            linestyle=style,
            zorder=1,
        )  # arrows go behind nodes

        arrow_collection.append(arrow)
        ax.add_patch(arrow)

    # update view
    padx, pady = 0.05 * w, 0.05 * h
    corners = (miretworkx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return arrow_collection


def draw_retworkx_labels(
    graph,
    pos,
    labels=None,
    font_size=12,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    clip_on=True,
):
    """Draw node labels on the graph.

    Parameters
    ----------
    graph: A retworkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    labels : dictionary (default={n: n for n in graph})
        Node labels in a dictionary of text labels keyed by node.
        Node-keys in labels should appear as keys in `pos`.
        If needed use: `{n:lab for n,lab in labels.items() if n in pos}`

    font_size : int (default=12)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, (default is Matplotlib's ax.text default)
        Specify text box properties (e.g. shape, color etc.) for node labels.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline',
                            'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    clip_on : bool (default=True)
        Turn on clipping of node labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed on the nodes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running "
                          "retworkx.draw_retworkx_nodes(). You can install "
                          "matplotlib with:\n'pip install matplotlib'") from e

    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = {n: n for n in graph.node_indexes()}

    text_items = {}  # there is no text collection so we'll fake one
    for n, label in labels.items():
        (x, y) = pos[n]
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transData,
            bbox=bbox,
            clip_on=clip_on,
        )
        text_items[n] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def draw_retworkx_edge_labels(
    graph,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
):
    """Draw edge labels.

    Parameters
    ----------
    graph: A retworkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline',
                            'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running "
                          "retworkx.draw_retworkx_nodes(). You can install "
                          "matplotlib with:\n'pip install matplotlib'") from e

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in graph.weighted_edge_list()}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0),
                        fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def apply_alpha(colors, alpha, elem_list, cmap=None, vmin=None, vmax=None):
    """Apply an alpha (or list of alphas) to the colors provided.

    Parameters
    ----------

    colors : color string or array of floats (default='r')
        Color of element. Can be a single color format string,
        or a sequence of colors with the same length as nodelist.
        If numeric values are specified they will be mapped to
        colors using the cmap and vmin,vmax parameters.  See
        matplotlib.scatter for more details.

    alpha : float or array of floats
        Alpha values for elements. This can be a single alpha value, in
        which case it will be applied to all the elements of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    elem_list : array of retworkx objects
        The list of elements which are being colored. These could be nodes,
        edges or labels.

    cmap : matplotlib colormap
        Color map for use if colors is a list of floats corresponding to points
        on a color mapping.

    vmin, vmax : float
        Minimum and maximum values for normalizing colors if a colormap is used

    Returns
    -------

    rgba_colors : numpy ndarray
        Array containing RGBA format values for each of the node colours.

    """
    try:
        import matplotlib as mpl
        import matplotlib.colors  # call as mpl.colors
        import matplotlib.cm  # call as mpl.cm
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running "
                          "retworkx.draw_retworkx_nodes(). You can install "
                          "matplotlib with:\n'pip install matplotlib'") from e

    # If we have been provided with a list of numbers as long as elem_list,
    # apply the color mapping.
    if len(colors) == len(elem_list) and isinstance(colors[0], Number):
        mapper = mpl.cm.ScalarMappable(cmap=cmap)
        mapper.set_clim(vmin, vmax)
        rgba_colors = mapper.to_rgba(colors)
    # Otherwise, convert colors to matplotlib's RGB using the colorConverter
    # object.  These are converted to numpy ndarrays to be consistent with the
    # to_rgba method of ScalarMappable.
    else:
        try:
            rgba_colors = np.array([mpl.colors.colorConverter.to_rgba(colors)])
        except ValueError:
            rgba_colors = np.array(
                [mpl.colors.colorConverter.to_rgba(color) for color in colors]
            )
    # Set the final column of the rgba_colors to have the relevant alpha values
    try:
        # If alpha is longer than the number of colors, resize to the number of
        # elements.  Also, if rgba_colors.size (the number of elements of
        # rgba_colors) is the same as the number of elements, resize the array,
        # to avoid it being interpreted as a colormap by scatter()
        if len(alpha) > len(rgba_colors) or rgba_colors.size == len(elem_list):
            rgba_colors = np.resize(rgba_colors, (len(elem_list), 4))
            rgba_colors[1:, 0] = rgba_colors[0, 0]
            rgba_colors[1:, 1] = rgba_colors[0, 1]
            rgba_colors[1:, 2] = rgba_colors[0, 2]
        rgba_colors[:, 3] = list(islice(cycle(alpha), len(rgba_colors)))
    except TypeError:
        rgba_colors[:, -1] = alpha
    return rgba_colors
