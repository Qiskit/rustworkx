#######################
rustworkx Documentation
#######################

rustworkx is a Python package for working with graphs and complex networks. It
enables the creation, interaction with, and study of graphs and networks.

It provides:

 * Data structures for creating graphs including directed graphs and multigraphs
 * A library of standard graph algorithms
 * Generators for various types of graphs including random graphs
 * Visualization functions for graphs

It is licensed under the
`Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__ license and the
source code is hosted on Github at:

https://github.com/Qiskit/rustworkx

rustworkx is written in the
`Rust programming language <https://www.rust-lang.org/>`__ to leverage Rust's
inherent performance and safety. While this provides numerous advantages
including significantly improved performance it does mean that the library
needs to be compiled when being installed from source (as opposed to a pure
Python library which can just be installed). rustworkx supports and publishes
pre-compiled binaries for Linux on x86, x86_64, aarch64, s390x, and ppc64le,
MacOS on x86_64, and arm64, and Windows 32bit and 64bit systems. However, if
you are not running on one of these platforms, you will need a rust compiler
to install rustworkx.

rustworkx-core
--------------

In addition to the Python library, as part of rustworkx a standalone Rust library
is also included. This library provides a generic interface for Rust users
that wish to use the same graph algorithm implementations exposed in the Python
library. This library, rustworkx-core, is used to build the Python rustworkx
library but can be used for any Rust application that needs the same graph
functionality that rustworkx provides.

You can find more details on rustworkx-core here:

https://crates.io/crates/rustworkx-core

and the documentation for the library is hosted here:

https://docs.rs/rustworkx-core/0.13.0/rustworkx_core/


Project history
---------------

.. note::

    The project has been renamed to **rustworkx**. You can still use the legacy
    retworkx name for now but starting in the 1.0.0 release the retworkx name
    will no longer be supported.

rustworkx was originally called retworkx and was created to be a high
performance replacement for the Qiskit project's internal usage of the
`NetworkX <https://networkx.org/>`__ library (which is where the name came
from: Rust + NetworkX = rustworkx) but it is not a drop-in replacement for
NetworkX (see :ref:`networkx` for more details). However, since it was
originally created it has grown to be an independent high performance general
purpose graph library that can be used for any application that needs to
interact with graphs or complex networks.

Contents:

.. toctree::
   :maxdepth: 2

   About Rustworkx <self>
   Installation and Getting Started <install>
   Rustworkx Tutorials and Guides <tutorial/index>
   Rustworkx API <api/index>
   Visualization <visualization>
   Release Notes <release_notes>
   Contributing Guide <CONTRIBUTING>
   rustworkx for NetworkX users <networkx>
   Benchmarks <benchmarks>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
