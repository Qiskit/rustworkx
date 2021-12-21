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

A directed acyclic graph is a directed graph which also doesn't contain
any cycles. A cycle is a non-empty trail in which the first and last nodes in
the trail are the same. For example:

.. jupyter-execute::

   cycle_graph = rx.generators.directed_cycle_graph(5)
   mpl_draw(cycle_graph)

is **not** acyclic. While the earlier path graph is acyclic.

In retworkx you can create a new :class:`.PyDiGraph` object that enforces
that no cycle is added by using the ``check_cycle`` constructor property

.. jupyter-execute::

    dag = rx.PyDiGraph(check_cycle=True)

or after a :class:`~retworkx.PyDiGraph` object is created with the
:attr:`~retworkx.PyDiGraph.check_cycle` attribute:

.. jupyter-execute::

    dag.check_cycle = True

You can also check the status of cycle checking by inspecting the
:attr:`~retworkx.PyDiGraph.check_cycle` attribute:

.. jupyter-execute::

    print(dag.check_cycle)

With ``check_cycle`` set to ``True`` whenever a
:meth:`.PyDiGraph.add_edge` or related functions that add an arbitrary edge to
the graph an error will be raised if a cycle would be introduced. This checking
does come with a noticeable (and potentially large) runtime overhead so it's
best to only use it if it's strictly necessary. You can also avoid this
overhead by using the :meth:`.PyDiGraph.add_parent` and
:meth:`.PyDiGraph.add_child` as both add a new node and edge simultaneously and
neither can introduce a cycle.


Applications of DAGs
====================

Topological Sorting
-------------------

A toplogical sort of a directed acyclic graph defined as :math:`G = (V,E)` is
a linear ordering of all its nodes such that if :math:`G` contains an edge
:math:`(u, v)` then :math:`u` appears before `v`. This only works with dags
because if there was a cycle in the graph :math:`G` then it's not possible
to find such a linear ordering.

A common application of dags is to use a toplogical sort is to schedule a
sequence of jobs or tasks based on their dependencies. These jobs are
represented by nodes and if there is an edge from :math:`u` to :math:`v` if
job :math:`u` must be completed before job :math:`v`. A topological sort
of a directed acyclic graph will give an order in which to perform these
jobs. For example:

.. jupyter-execute::

    # Create a job dag
    dependency_dag = rx.PyDiGraph(check_cycle=True)
    job_a = dependency_dag.add_node("Job A")
    job_b = dependency_dag.add_child(job_a, "Job B", None)
    job_c = dependency_dag.add_child(job_b, "Job C", None)
    job_d = dependency_dag.add_child(job_a, "Job D", None)
    job_e = dependency_dag.add_parent(job_d, "Job E", None)
    job_f = dependency_dag.add_child(job_e, "Job F", None)
    dependency_dag.add_edge(job_a, job_f, None)
    dependency_dag.add_edge(job_c, job_d, None)
    mpl_draw(
        dependency_dag,
        with_labels=True,
        node_size=800,
        node_color="yellow",
        labels=str,
    )

Above we define a dag with 6 jobs and dependency relationship between these
jobs. Now if we run the :func:`~retworkx.topological_sort` function on the
graph it will return a linear order to execute the jobs that will respect
the dependency releationship.

.. jupyter-execute::

    topo_sorted = rx.topological_sort(dependency_dag)
    # Print job labels
    print([dependency_dag[job_index] for job_index in topo_sorted])

Qiskit's Compiler
-----------------

Another application using directed acyclic graphs is the compiler in
`Qiskit <https://qiskit.org>`__. Qiskit is an SDK for working with
quantum computing.

Retworkx was originally written to accelerate the performance of the Qiskit
compiler's use of directed acyclic graphs.
