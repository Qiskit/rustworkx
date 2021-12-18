***********************
Directed Acyclic Graphs
***********************

This tutorial will explore using retworkx to work with directed acyclic graphs
(also known as a dag).

Directed Graph
==============

A directed graph is a graph that is made up of a set of nodes connected by
directed edges (often called arcs). Edges have a directionality which is
different from undirected graphs where edges have no notion of a direction to
them. For example:

.. jupyter-execute::

    import retworkx as rx
    from retworkx.visualization import mpl_draw

    path_graph = rx.generators.directed_path_graph(5)
    mpl_draw(path_graph)

In this example we created a 5 node directed path graph. This shows the
directionality of the edges in the graph with the arrow head pointing to the
target node. Each edge has a source and target

Directed Acyclic Graphs
=======================

A directed acyclic graph is a directed graph where the graph is directed and
acyclic, in other words it's a directed graph that doesn't contain any cycles.
A cycle is a non-empty trail in which the first and last nodes in the trail
are the same. For example:

.. jupyter-execute::

   cycle_graph = rx.generators.directed_cycle_graph(5)
   mpl_draw(cycle_graph)

This example is **not** a acyclic.

In retworkx you can create a new :class:`.PyDiGraph` object that enforces
that no cycle is added by using the ``cycle_check``


Applications of DAGs
====================

Topological Sorting
-------------------

A common application of dags is to use a toplogical sort is to schedule a
sequence of jobs or tasks based on their dependencies.

Qiskit's Compiler
-----------------

Another application using directed acyclic graphs is the compiler in
`Qiskit <https://qiskit.org>`__. Qiskit is an SDK for working with
quantum computing.

Retworkx was originally written to accelerate the performance of the Qiskit
compiler's use of directed acyclic graphs.
