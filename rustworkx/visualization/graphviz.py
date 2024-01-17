# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import subprocess
import tempfile
import io

__all__ = ["graphviz_draw", "have_dot", "is_format_supported"]

METHODS = {"twopi", "neato", "circo", "fdp", "sfdp", "dot"}

_NO_PILLOW_MSG = """
Pillow is necessary to use graphviz_draw() it can be installed
with 'pip install pydot pillow.
"""

_NO_DOT_MSG = """
Graphviz could not be found or run. This function requires that
Graphviz is installed. If you need to install Graphviz you can
refer to: https://graphviz.org/download/#executable-packages for
instructions.
"""

try:
    import PIL

    HAVE_PILLOW = True
except Exception:
    HAVE_PILLOW = False


# Return True if `dot` is found and executes.
def have_dot():
    try:
        subprocess.run(
            ["dot", "-V"],
            cwd=tempfile.gettempdir(),
            check=True,
            capture_output=True,
        )
    except Exception:
        return False
    return True


def _capture_support_string():
    try:
        subprocess.check_output(
            ["dot", "-T", "bogus_format"],
            cwd=tempfile.gettempdir(),
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exerr:
        return exerr.output.decode()


# Return collection of image formats supported by dot, as
# a `set` of `str`.
def _supported_image_formats():
    error_string = _capture_support_string()
    # 7 is a magic number based error message.
    # The words following the first seven are the formats.
    return set(error_string.split()[7:])


def is_format_supported(image_format: str):
    """Return true if `image_format` is supported by the installed graphviz."""
    return image_format in _supported_image_formats()


def graphviz_draw(
    graph,
    node_attr_fn=None,
    edge_attr_fn=None,
    graph_attr=None,
    filename=None,
    image_type=None,
    method=None,
):
    """Draw a :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph` object
    using graphviz

    .. note::

        This requires that pydot, pillow, and graphviz be installed. Pydot can
        be installed via pip with ``pip install pydot pillow`` however graphviz
        will need to be installed separately. You can refer to the
        Graphviz
        `documentation <https://graphviz.org/download/#executable-packages>`__
        for instructions on how to install it.

    :param graph: The rustworkx graph object to draw, can be a
        :class:`~rustworkx.PyGraph` or a :class:`~rustworkx.PyDiGraph`
    :param node_attr_fn: An optional callable object that will be passed the
        weight/data payload for every node in the graph and expected to return
        a dictionary of Graphviz node attributes to be associated with the node
        in the visualization. The key and value of this dictionary **must** be
        a string.
    :param edge_attr_fn: An optional callable that will be passed the
        weight/data payload for each edge in the graph and expected to return a
        dictionary of Graphviz edge attributes to be associated with the edge
        in the visualization file. The key and value of this dictionary
        must be a string.
    :param dict graph_attr: An optional dictionary that specifies any Graphviz
        graph attributes for the visualization. The key and value of this
        dictionary must be a string.
    :param str filename: An optional path to write the visualization to. If
        specified the return type from this function will be ``None`` as the
        output image is saved to disk.
    :param str image_type: The image file format to use for the generated
        visualization. The support image formats are:
        ``'canon'``, ``'cmap'``, ``'cmapx'``, ``'cmapx_np'``, ``'dia'``,
        ``'dot'``, ``'fig'``, ``'gd'``, ``'gd2'``, ``'gif'``, ``'hpgl'``,
        ``'imap'``, ``'imap_np'``, ``'ismap'``, ``'jpe'``, ``'jpeg'``,
        ``'jpg'``, ``'mif'``, ``'mp'``, ``'pcl'``, ``'pdf'``, ``'pic'``,
        ``'plain'``, ``'plain-ext'``, ``'png'``, ``'ps'``, ``'ps2'``,
        ``'svg'``, ``'svgz'``, ``'vml'``, ``'vmlz'``, ``'vrml'``, ``'vtx'``,
        ``'wbmp'``, ``'xdot'``, ``'xlib'``. It's worth noting that while these
        formats can all be used for generating image files when the ``filename``
        kwarg is specified, the Pillow library used for the returned object can
        not work with all these formats.
    :param str method: The layout method/Graphviz command method to use for
        generating the visualization. Available options are ``'dot'``,
        ``'twopi'``, ``'neato'``, ``'circo'``, ``'fdp'``, and ``'sfdp'``.
        You can refer to the
        `Graphviz documentation <https://graphviz.org/documentation/>`__ for
        more details on the different layout methods. By default ``'dot'`` is
        used.

    :returns: A ``PIL.Image`` object of the generated visualization, if
        ``filename`` is not specified. If ``filename`` is specified then
        ``None`` will be returned as the visualization was written to the
        path specified in ``filename``
    :rtype: PIL.Image

    .. jupyter-execute::

        import rustworkx as rx
        from rustworkx.visualization import graphviz_draw

        def node_attr(node):
          if node == 0:
            return {'color': 'yellow', 'fillcolor': 'yellow', 'style': 'filled'}
          if node % 2:
            return {'color': 'blue', 'fillcolor': 'blue', 'style': 'filled'}
          else:
            return {'color': 'red', 'fillcolor': 'red', 'style': 'filled'}

        graph = rx.generators.directed_star_graph(weights=list(range(32)))
        graphviz_draw(graph, node_attr_fn=node_attr, method='sfdp')

    """
    _have_dot = have_dot()
    if not (HAVE_PILLOW and _have_dot):
        raise RuntimeError(_NO_DOT_MSG + _NO_PILLOW_MSG)
    if not HAVE_PILLOW:
        raise ImportError(_NO_PILLOW_MSG)
    if not _have_dot:
        raise RuntimeError(_NO_DOT_MSG)

    dot_str = graph.to_dot(node_attr_fn, edge_attr_fn, graph_attr)
    if image_type is None:
        output_format = "png"
    else:
        output_format = image_type

    supported_formats = _supported_image_formats()
    if output_format not in supported_formats:
        raise ValueError(
            "The specified value for the image_type argument, "
            f"'{output_format}' is not a valid choice. It must be one of: "
            f"{supported_formats}"
        )

    if method is None:
        prog = "dot"
    else:
        if method not in METHODS:
            raise ValueError(
                f"The specified value for the method argument, '{method}' is "
                f"not a valid choice. It must be one of: {METHODS}"
            )
        prog = method

    if not filename:
        dot_result = subprocess.run(
            [prog, "-T", output_format],
            input=dot_str.encode("utf-8"),
            capture_output=True,
            encoding=None,
            check=True,
            text=False,
        )
        dot_bytes_image = io.BytesIO(dot_result.stdout)
        image = PIL.Image.open(dot_bytes_image)
        return image
    else:
        subprocess.run(
            [prog, "-T", output_format, "-o", filename],
            input=dot_str,
            check=True,
            encoding="utf8",
            text=True,
        )
        return None
