===============
Getting Started
===============

rustworkx is a general purpose graph library for Python written in Rust to take
advantage of the performance and safety that Rust provides. It is designed to
provide a high performance general purpose graph library for any Python
application.

Installing Rustworkx
====================

rustworkx is published on pypi so on x86_64, i686, ppc64le, s390x, and aarch64
Linux systems, x86_64 and arm64 on macOS, and 32 and 64 bit Windows
installing is as simple as running::

    pip install rustworkx

This will install a precompiled version of rustworkx into your python
environment.

.. _install-unsupported:

Installing on a platform without precompiled binaries
-----------------------------------------------------

If there are no precompiled binaries published for your system you'll have to
build the package from source. However, to be able able to build the package from
the published source package you need to have Rust >= 1.64 installed (and also
cargo which is normally included with rust) You can use
`rustup <https://rustup.rs/>`_ (a cross platform installer for rust) to make this
simpler, or rely on
`other installation methods <https://forge.rust-lang.org/infra/other-installation-methods.html>`__.
A source package is also published on PyPI, so you still can run ``pip`` to install
it. Once you have Rust properly installed, running::

    pip install rustworkx

will build rustworkx for your local system from the source package and install it
just as it would if there was a prebuilt binary available.


.. note::

    To build from source you will need to ensure you have pip >=19.0.0
    installed, which supports PEP-517, or that you have manually installed
    setuptools-rust prior to running pip install rustworkx. If you recieve an
    error about ``setuptools-rust`` not being found you should upgrade pip with
    ``pip install -U pip`` or manually install ``setuptools-rust`` with:
    ``pip install setuptools-rust`` and try again.

.. _platform-suppport:

Platform Support
================

Rustworkx strives to support as many platforms as possible, but due to
limitations in available testing resources and platform availability, not all
platforms can be supported. Platform support for rustworkx is broken into 4
tiers with different levels of support for each tier. For platforms outside
these, rustworkx is probably still installable, but it’s not tested and you will
need a Rust compiler and have to build rustworkx (and likely Numpy too) from
source.

.. list-table:: Platform Support
   :header-rows: 1

   * - Operating System
     - CPU Architecture
     - Support Tier
     - Notes 
   * - Linux
     - x86_64
     - :ref:`tier-1`
     - Distributions compatible with the `manylinux 2014`_ packaging specification
   * - Linux
     - i686 
     - :ref:`tier-2` (Python < 3.10), :ref:`tier-3` (Python >= 3.10)
     - Distributions compatible with the `manylinux 2014`_ packaging specification
   * - Linux
     - aarch64
     - :ref:`tier-2`
     - Distributions compatible with the `manylinux 2014`_ packaging specification
   * - Linux
     - pp64le
     - :ref:`tier-3`
     - Distributions compatible with the `manylinux 2014`_ packaging specification
   * - Linux
     - s390x
     - :ref:`tier-4`
     - Distributions compatible with the `manylinux 2014`_ packaging specification
   * - Linux (musl)
     - x86_64
     - :ref:`tier-3`
     -
   * - Linux (musl)
     - aarch64
     - :ref:`tier-4`
     - 
   * - macOS (10.12 or newer)
     - x86_64
     - :ref:`tier-1`
     -
   * - macOS (11 or newer)
     - arm64
     - :ref:`tier-1` [#f1]_
     -
   * - Windows 64bit
     - x86_64
     - :ref:`tier-1`
     -
   * - Windows 32bit 
     - i686 or x86_64
     - :ref:`tier-2` (Python < 3.10), :ref:`tier-3` (Python >= 3.10)
     -


.. _manylinux 2014: https://peps.python.org/pep-0599/>

.. [#f1] Due to CI environment limitations tests for macOS arm64 are only run with
   Python >= 3.10. The published binaries are still built and tested for all supported
   Python versions, but the tests run on proposed changes are only run with Python >=3.10


.. _tier-1:

Tier 1
------

Tier 1 supported platforms are fully tested upstream as part of the development
process to ensure any proposed change will function correctly. Pre-compiled
binaries are built, tested, and published to PyPI as part of the release
process. These platforms are expected to be installable with just a functioning
Python environment.

.. _tier-2:

Tier 2
------

Tier 2 platforms are not tested upstream as part of the development process.
However, pre-compiled binaries are built, tested, and published to PyPI as part
of the release process and these packages can be expected to be installed with
just a functioning Python environment.

.. _tier-3:

Tier 3
------

Tier 3 platforms are not tested upstream as part of the development process.
Pre-compiled binaries are built, tested and published to PyPI as
part of the release process. However, they may not installable with just a
functioning Python environment and you may be required to build Numpy from
source, which requires a C/C++ compiler, as part of the installation process.

.. _tier-4:

Tier 4
------

Tier 4 platforms are not tested upstream as part of the development process.
Pre-compiled binaries are built and published to PyPI as part of the release
process, with no testing at all. They may not be installable with just a
functioning Python environment and may require a C/C++ compiler or additional
programs to build dependencies from source as part of the installation process.
Support for these platforms are best effort only.

Using rustworkx
===============

Once you have rustworkx installed you can use it by importing rustworkx. All the
functions and graph classes are off the root of the package. For example,
calculating the shortest path between A and C would be::

    import rustworkx as rx
    
    graph = rx.PyGraph()
    
    # Each time add node is called, it returns a new node index
    a = graph.add_node("A")
    b = graph.add_node("B")
    c = graph.add_node("C")
    
    # add_edges_from takes tuples of node indices and weights,
    # and returns edge indices
    graph.add_edges_from([(a, b, 1.5), (a, c, 5.0), (b, c, 2.5)])
    
    # Returns the path A -> B -> C
    rx.dijkstra_shortest_paths(graph, a, c, weight_fn=float)

You can refer to the :ref:`intro-tutorial` for more details on getting started
with rustworkx.
