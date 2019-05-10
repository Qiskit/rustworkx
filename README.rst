retworkx
========

retworkx is a rust graph library interface to python3. For right now it's scope
is as an experiment in being a potential replacement for `qiskit-terra`_'s
networkx usage (hence the name). The scope might grow or change over time, but
to start it's just about building a DAG and operating on it with the performance
and safety that Rust provides. It is also a personal exercise in learning how
to interface rust with python.

.. _qiskit-terra: https://github.com/Qiskit/qiskit-terra

Installing retworkx
-------------------

retworkx uses PyO3 and setuptools-rust to build the python interface.
Unfortunately this means you need to use nightly rust because PyO3 only works
with nightly at this point. You can use rustup to install rust nightly.

Once you have nightly rust and cargo installed you can easily install retworkx
into your python environment using pip. Once you have a local clone of the repo
you can install retworkx into your python env with::

  pip install .

Assuming your current working directory is the root of the repo. Otherwise
you can run::

  pip install $PATH_TO_REPO_ROOT

Using retworkx
--------------

Once you have retworkx installed you can use it by importing retworkx. All
the functions and the PyDAG class are off the root of the package. For example,
building a DAG and adding 2 nodes with an edge between them would be::

    import retworkx

    my_dag = retworkx.PyDAG()
    # add_node(), add_child(), and add_parent() return the node index
    root_node = my_dag.add_node("MyRoot")
    my_dag.add_child(root_node, "AChild")
