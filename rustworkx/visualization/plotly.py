# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import subprocess
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import numpy as np

import rustworkx
from rustworkx import PyDiGraph, PyGraph
from rustworkx.visualization.utils import has_graphviz

try:
    import plotly.graph_objects as go  # type: ignore

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

if TYPE_CHECKING:
    import plotly.graph_objects as go  # type: ignore

NodeT = TypeVar("NodeT")
EdgeT = TypeVar("EdgeT")

__all__ = ["plotly_draw"]

GRAPHVIZ_METHODS = frozenset({"twopi", "neato", "circo", "fdp", "sfdp", "dot"})
METHODS = frozenset(GRAPHVIZ_METHODS | {"spring"})
ARROW_POS = 0.7
ARROW_SIZE = 12


def _render_bezier_spline(control_points: np.ndarray, steps_per_segment: int = 20) -> np.ndarray:
    """Render a cubic bezier spline from (3n+1, 2) control points.

    Returns an (m, 2) array of interpolated points.
    """
    num_segments = (len(control_points) - 1) // 3
    if num_segments == 0:
        return control_points

    segments = np.empty((num_segments, 4, 2))
    segments[:, 0, :] = control_points[:-1:3]
    segments[:, 1:, :] = control_points[1:].reshape(num_segments, 3, 2)

    steps = np.linspace(0, 1, steps_per_segment, endpoint=True)
    basis = np.stack(
        [
            (1 - steps) ** 3,
            3 * (1 - steps) ** 2 * steps,
            3 * (1 - steps) * steps**2,
            steps**3,
        ],
        axis=1,
    )

    return np.einsum("sf,nfd->nsd", basis, segments).reshape(-1, 2)


def _graphviz_layout(
    graph: PyDiGraph[NodeT, EdgeT] | PyGraph[NodeT, EdgeT],
    prog: str,
    graph_attr: dict[str, str] | None,
) -> tuple[dict[int, tuple[float, float]], dict[int, np.ndarray]]:
    """Run graphviz ``-Tplain`` and return node positions and edge spline control points.

    Returns:
        node_positions: ``{node_index: (x, y)}``
        edge_splines: ``{edge_index: np.ndarray of shape (n, 2)}``
    """
    dot_str = cast(str, graph.to_dot(graph_attr=graph_attr))
    gv_result = subprocess.run(
        [prog, "-Tplain"],
        input=dot_str.encode("utf-8"),
        capture_output=True,
        check=True,
    )

    node_positions: dict[int, tuple[float, float]] = {}
    # Collect edges keyed by (tail, head), with a list to handle multigraph
    edge_splines_by_endpoints: defaultdict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    for line in gv_result.stdout.decode("utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "node":
            node_idx = int(parts[1])
            node_positions[node_idx] = (float(parts[2]), float(parts[3]))
        elif parts[0] == "edge":
            tail = int(parts[1])
            head = int(parts[2])
            num_points = int(parts[3])
            coords = np.empty((num_points, 2))
            coords[:, 0] = [float(parts[4 + 2 * i]) for i in range(num_points)]
            coords[:, 1] = [float(parts[5 + 2 * i]) for i in range(num_points)]
            edge_splines_by_endpoints[(tail, head)].append(coords)

    # Map edge indices to spline control points by consuming from the lists in order
    # (graphviz emits edges in the same order as the DOT file, matching edge_index_map)
    consumed: defaultdict[tuple[int, int], int] = defaultdict(int)
    edge_splines: dict[int, np.ndarray] = {}
    for edge_idx, (src, tgt, _weight) in graph.edge_index_map().items():
        key = (src, tgt)
        idx = consumed[key]
        spline_list = edge_splines_by_endpoints.get(key, [])
        if idx < len(spline_list):
            edge_splines[edge_idx] = spline_list[idx]
        consumed[key] = idx + 1

    return node_positions, edge_splines


def _spring_layout(
    graph: PyDiGraph[NodeT, EdgeT] | PyGraph[NodeT, EdgeT],
    spring_attr: dict | None = None,
) -> tuple[dict[int, tuple[float, float]], dict[int, np.ndarray]]:
    """Compute layout using ``rustworkx.spring_layout`` with straight-line edges."""
    kwargs: dict = {"repulsive_exponent": 3}
    if spring_attr is not None:
        kwargs.update(spring_attr)
    pos = rustworkx.spring_layout(graph, **kwargs)
    node_positions = {idx: (coord[0], coord[1]) for idx, coord in pos.items()}

    edge_splines: dict[int, np.ndarray] = {}
    for edge_idx, (src, tgt, _weight) in graph.edge_index_map().items():
        src_pos = node_positions.get(src, (0.0, 0.0))
        tgt_pos = node_positions.get(tgt, (0.0, 0.0))
        edge_splines[edge_idx] = np.array([src_pos, tgt_pos])

    return node_positions, edge_splines


def plotly_draw(
    graph: "PyDiGraph[NodeT, EdgeT] | PyGraph[NodeT, EdgeT]",  # noqa
    node_attr_fn: Callable[[NodeT], dict] | None = None,
    edge_attr_fn: Callable[[EdgeT], dict] | None = None,
    graph_attr: dict[str, str] | None = None,
    method: Literal["twopi", "neato", "circo", "fdp", "sfdp", "dot", "spring"] | None = None,
    show_node_indices: bool = True,
    show_edge_indices: bool = False,
    spring_attr: dict | None = None,
) -> go.Figure:
    """Draw a graph or directed graph object using plotly.

    This function uses plotly for interactive rendering of a graph. It produces a
    ``plotly.graph_objects.Figure`` that can be displayed in Jupyter notebooks or saved to HTML. It
    can use either graphviz or :func:`~rustworkx.spring_layout` as the backend for computing the
    graph layout and edge splines.

    .. note::

        This function requires that plotly be installed. Plotly can be installed via
        pip with ``pip install plotly``. Graphviz must be installed to use any layout method other
        than ``method='spring'` You can refer to the Graphviz
        `documentation <https://graphviz.org/download/#executable-packages>`__
        for instructions on how to install it.

    :param graph: The rustworkx graph object to draw, can be a
        :class:`~rustworkx.PyGraph` or a :class:`~rustworkx.PyDiGraph`
    :param node_attr_fn: An optional callable that will be passed the
        weight/data payload for every node in the graph and expected to return
        a dictionary of plotly node attributes. Supported keys are:
        ``color``, ``size``, ``label``, ``symbol``, ``opacity``,
        ``line_color``, ``line_width``.
    :param edge_attr_fn: An optional callable that will be passed the
        weight/data payload for each edge in the graph and expected to return
        a dictionary of plotly edge attributes. Supported keys are:
        ``color``, ``width``, ``label``, ``dash``, ``opacity``.
    :param dict graph_attr: An optional dictionary that specifies any Graphviz
        graph attributes for the layout. The key and value of this dictionary
        must be a string. Common options include ``rankdir``, ``ranksep``,
        ``nodesep``.
    :param str method: The layout method to use. Available options are
        ``'dot'``, ``'twopi'``, ``'neato'``, ``'circo'``, ``'fdp'``,
        ``'sfdp'`` (all Graphviz), and ``'spring'``
        (:func:`~rustworkx.spring_layout`, no Graphviz required).
        By default ``'dot'`` is used, falling back to ``'spring'`` if
        Graphviz is not installed.
    :param bool show_node_indices: Whether to display node index integers
        as text labels on nodes. If ``node_attr_fn`` returns a ``"label"``
        key for a node, that label is used instead of the index. Defaults
        to ``True``.
    :param bool show_edge_indices: Whether to display edge index integers
        as text annotations on edges. Defaults to ``False``.
    :param dict spring_attr: An optional dictionary of keyword arguments
        passed to :func:`~rustworkx.spring_layout` when using ``'spring'``
        layout. Common keys include ``seed``, ``k``, ``num_iter``, and
        ``repulsive_exponent``. Has no effect on Graphviz methods. By
        default ``repulsive_exponent=3`` is used.

    :returns: A ``plotly.graph_objects.Figure`` of the generated visualization.
    :rtype: plotly.graph_objects.Figure

    .. jupyter-execute::

        import rustworkx as rx
        from rustworkx.visualization import plotly_draw

        graph = rx.generators.directed_star_graph(weights=list(range(8)))
        plotly_draw(graph)

    """
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is necessary to use plotly_draw(). " "It can be installed with 'pip install plotly'."
        )

    if method is not None and method not in METHODS:
        raise ValueError(
            f"The specified value for the method argument, '{method}' is "
            f"not a valid choice. It must be one of: {METHODS}"
        )

    is_directed = isinstance(graph, PyDiGraph)

    # Compute layout
    if method == "spring":
        if graph_attr is not None:
            warnings.warn(
                "'graph_attr' is ignored when method='spring'.",
                stacklevel=2,
            )
        node_positions, edge_splines = _spring_layout(graph, spring_attr=spring_attr)
    elif has_graphviz():
        prog = method if method is not None else "dot"
        node_positions, edge_splines = _graphviz_layout(graph, prog, graph_attr)
    else:
        if method is not None or graph_attr is not None:
            warnings.warn(
                "Graphviz is not available; 'method' and 'graph_attr' arguments "
                "are ignored. Falling back to spring_layout.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                "Graphviz is not available. Falling back to spring_layout.",
                stacklevel=2,
            )
        node_positions, edge_splines = _spring_layout(graph, spring_attr=spring_attr)

    # Collect per-node attributes
    node_attrs: dict[int, dict] = {}
    if node_attr_fn is not None:
        for idx in graph.node_indices():
            node_attrs[idx] = node_attr_fn(graph[idx])

    # Collect per-edge attributes
    edge_attrs: dict[int, dict] = {}
    if edge_attr_fn is not None:
        for edge_idx, (_src, _tgt, weight) in graph.edge_index_map().items():
            edge_attrs[edge_idx] = edge_attr_fn(weight)

    # Build edge traces — group edges by style for efficient rendering
    edge_style_groups: dict[tuple, list[int]] = {}
    for edge_idx in graph.edge_index_map():
        a = edge_attrs.get(edge_idx, {})
        style_key = (
            a.get("color", "#888"),
            a.get("width", 1),
            a.get("dash", "solid"),
            a.get("opacity", 1.0),
        )
        edge_style_groups.setdefault(style_key, []).append(edge_idx)

    edge_traces = []
    for (color, width, dash, opacity), indices in edge_style_groups.items():
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_hover: list[str | None] = []
        marker_symbols: list[str] = []
        marker_sizes: list[int] = []

        for edge_idx in indices:
            src, tgt = graph.get_edge_endpoints_by_index(edge_idx)
            src_pos = node_positions.get(src, (0.0, 0.0))
            tgt_pos = node_positions.get(tgt, (0.0, 0.0))

            spline = edge_splines.get(edge_idx)
            if spline is not None:
                rendered = _render_bezier_spline(spline)
                # Prepend source node center and append target node center
                # so edges visually connect to the nodes (graphviz splines
                # stop at the node boundary, not the center).
                xs = [src_pos[0]] + rendered[:, 0].tolist() + [tgt_pos[0]]
                ys = [src_pos[1]] + rendered[:, 1].tolist() + [tgt_pos[1]]
            else:
                xs = [src_pos[0], tgt_pos[0]]
                ys = [src_pos[1], tgt_pos[1]]

            n_pts = len(xs)
            edge_x.extend(xs)
            edge_y.extend(ys)
            edge_hover.extend([str(edge_idx)] * n_pts)

            if is_directed and n_pts > 1:
                arrow_idx = int(n_pts * ARROW_POS)
                arrow_idx = max(0, min(arrow_idx, n_pts - 1))
                syms = ["circle"] * n_pts
                sizes = [0] * n_pts
                syms[arrow_idx] = "arrow"
                sizes[arrow_idx] = ARROW_SIZE
                marker_symbols.extend(syms)
                marker_sizes.extend(sizes)
            else:
                marker_symbols.extend(["circle"] * n_pts)
                marker_sizes.extend([0] * n_pts)

            # Separator between edges
            edge_x.append(None)
            edge_y.append(None)
            edge_hover.append(None)
            marker_symbols.append("circle")
            marker_sizes.append(0)

        if is_directed:
            trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines+markers",
                line=dict(color=color, width=width, dash=dash, shape="spline", smoothing=1.0),
                opacity=opacity,
                hovertext=edge_hover,
                hoverinfo="text",
                showlegend=False,
                marker=dict(
                    size=marker_sizes,
                    symbol=marker_symbols,
                    color=color,
                    angleref="previous",
                ),
            )
        else:
            trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color=color, width=width, dash=dash, shape="spline", smoothing=1.0),
                opacity=opacity,
                hovertext=edge_hover,
                hoverinfo="text",
                showlegend=False,
            )
        edge_traces.append(trace)

    # Node trace
    sorted_indices = sorted(node_positions.keys())
    node_x = [node_positions[idx][0] for idx in sorted_indices]
    node_y = [node_positions[idx][1] for idx in sorted_indices]

    node_colors = []
    node_sizes = []
    node_symbols = []
    node_opacities = []
    node_line_colors = []
    node_line_widths = []
    node_labels = []
    node_hover_texts = []
    has_labels = show_node_indices

    for idx in sorted_indices:
        attrs = node_attrs.get(idx, {})
        node_colors.append(attrs.get("color", "#1f77b4"))
        node_sizes.append(attrs.get("size", 15))
        node_symbols.append(attrs.get("symbol", "circle"))
        node_opacities.append(attrs.get("opacity", 1.0))
        node_line_colors.append(attrs.get("line_color", "#000"))
        node_line_widths.append(attrs.get("line_width", 1))
        label = attrs.get("label", str(idx))
        node_labels.append(label)
        node_hover_texts.append(str(idx))
        if "label" in attrs:
            has_labels = True

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if has_labels else "markers",
        marker=dict(
            color=node_colors,
            size=node_sizes,
            symbol=node_symbols,
            opacity=node_opacities,
            line=dict(color=node_line_colors, width=node_line_widths),
        ),
        text=node_labels if has_labels else None,
        textposition="top center",
        hovertext=node_hover_texts,
        hoverinfo="text",
        showlegend=False,
    )

    # Edge label/index annotations
    annotations = []
    for edge_idx, (_src, _tgt, weight) in graph.edge_index_map().items():
        attrs = edge_attrs.get(edge_idx, {})
        label = attrs.get("label")
        text = label if label else (str(edge_idx) if show_edge_indices else None)
        if text:
            spline = edge_splines.get(edge_idx)
            if spline is not None:
                rendered = _render_bezier_spline(spline)
                mid = rendered[len(rendered) // 2]
                mid_x, mid_y = float(mid[0]), float(mid[1])
            else:
                src, tgt = graph.get_edge_endpoints_by_index(edge_idx)
                src_pos = node_positions.get(src, (0.0, 0.0))
                tgt_pos = node_positions.get(tgt, (0.0, 0.0))
                mid_x = (src_pos[0] + tgt_pos[0]) / 2
                mid_y = (src_pos[1] + tgt_pos[1]) / 2
            annotations.append(
                dict(
                    x=mid_x,
                    y=mid_y,
                    text=text,
                    showarrow=False,
                    font=dict(size=10),
                )
            )

    # Assemble figure
    fig = go.Figure(
        data=[*edge_traces, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            annotations=annotations if annotations else None,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig
