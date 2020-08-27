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

    The PyDAG class is constructed using the Rust library
    `petgraph <https://github.com/petgraph/petgraph/>`___ around the
    ``StableGraph`` type. The limitations and quirks with this library and
    type dictate how this operates. The biggest thing to be aware of when using
    the PyDAG class is that an integer node and edge index is used for accessing
    elements on the DAG, not the data/weight of nodes and edges. By default the
    PyDAG realtime cycle checking is disabled for performance, however you can
    opt-in to having the PyDAG class ensure that no cycles are added by setting
    the ``check_cycle`` attribute to True. For example::

        import retworkx
        dag = retworkx.PyDAG()
        dag.check_cycle = True

    or at object creation::

        import retworkx
        dag = retworkx.PyDAG(check_cycle=True)

    With check_cycle set to true any calls to :meth:`PyDAG.add_edge` will
    ensure that no cycles are added, ensuring that the PyDAG class truly
    represents a directed acyclic graph.

    .. note::
        When using ``copy.deepcopy()`` or pickling node indexes are not
        guaranteed to be preserved.

    PyDAG is a subclass of the PyDiGraph class and behaves identically to
    the :class:`~retworkx.PyDiGraph` class.
    """
    pass
