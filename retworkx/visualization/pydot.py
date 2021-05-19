# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import tempfile

try:
    import pydot
    from PIL import Image

    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

__all__ = ["pydot_draw"]


def pydot_draw(
    graph,
    node_attr_fn=None,
    edge_attr_fn=None,
    graph_attr=None,
    filename=None,
    image_type=None,
):
    """Draw a :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph` object
    using graphviz via PyDot

    .. note::

        This requires that pydot, pillow, and graphviz be installed. Pydot can
        be installed via pip with ``pip install pydot pillow`` however graphviz
        will need to be installed separately. You can refer to the
        `Graphviz documentation <https://graphviz.org/download/#executable-packages>`__
        for instructions on how to install it

    :param :class:`~retworkx.PyGraph`|:class:`~retworkx.PyDiGraph` graph: The
            retworkx graph object to draw.
    :param node_attr_fn: An optional callable object that will be passed the
        weight/data payload for every node in the graph and expected to return
        a dictionary of Graphviz node attributes to be associated with the node
        in the visualization. The key and value of this dictionary **must** be
        a string.
    :param edge_attr: An optional callable that will be passed the weight/data
        payload for each edge in the graph and expected to return a dictionary
        of Graphviz edge attributes to be associated with the edge in the
        visualization file. The key and value of this dictionary **must** be a
        string.
    :param dict graph_attr: An optional dictionary that specifies any Graphviz
        graph attributes for the visualization. The key and value of this
        dictionary **must** be a string.
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
        ``'wbmp'``, ``'xdot'``, ``'xlib'``]

    :returns: A ``PIL.Image`` object of the generated visualization, if
        ``filename`` is not specified. If ``filename`` is specified then
        ``None`` will be returned as the visualization was written to the
        path specified in ``filename``
    :rtype: PIL.Image
    """
    if not HAS_PYDOT:
        raise ImportError(
            "Pydot and Pillow are necessary to use pydot_draw() "
            "it can be installed with 'pip install pydot pillow'"
        )

    dot_str = graph.to_dot(node_attr_fn, edge_attr_fn, graph_attr)
    dot = pydot.graph_from_dot_data(dot_str)[0]
    if image_type is None:
        output_format = "png"
    else:
        output_format = image_type

    if not filename:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "dag.png")
            dot.write(tmp_path, format=output_format)
            with Image.open(tmp_path) as temp_image:
                image = temp_image.copy()
            os.remove(tmp_path)
            return image
    else:
        dot.write(filename, format=output_format)
        return None
