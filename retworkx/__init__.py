# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import sys

from .retworkx import *
sys.modules['retworkx.generators'] = generators


class PyDAG(PyDiGraph):
    """A class for creating direct acyclic graphs.

    PyDAG is just an alias of the PyDiGraph class and behaves identically to
    the :class:`~retworkx.PyDiGraph` class and can be used interchangably
    with ``PyDiGraph``. It currently exists solely as a backwards
    compatibility alias for users of retworkx from prior to the
    0.4.0 release when there was no PyDiGraph class.

    The PyDAG class is used to create a directed graph. It can be a
    multigraph (have multiple edges between nodes). Each node and edge
    (although rarely used for edges) is indexed by an integer id. Additionally,
    each node and edge contains an arbitrary Python object as a weight/data
    payload.

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
    """
    pass
