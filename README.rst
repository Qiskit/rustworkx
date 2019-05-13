retworkx
========

* You can see the full rendered docs at: https://retworkx.readthedocs.io/en/latest/index.html


retworkx is a rust graph library interface to python3. For right now it's scope
is as an experiment in being a potential replacement for `qiskit-terra`_'s
networkx usage (hence the name). The scope might grow or change over time, but
to start it's just about building a DAG and operating on it with the performance
and safety that Rust provides. It is also a personal exercise in learning how
to interface rust with python.

.. _qiskit-terra: https://github.com/Qiskit/qiskit-terra

Installing retworkx
-------------------

retworkx is published on pypi so on x86_64 and i686 linux systems or Mac OSX
systems installing is as simple as running::

  pip install retworkx

This will install a precompiled version of retworkx into your python
environment. However if there are no precompiled binaries published for your
system you'll have to compile the code. The source package is also published on
pypi so you can also run the above command to install it. However, there are 2
preconditions for this to work, first you need to have cargo/rustc **nightly**
in your PATH. You can use `rustup`_ to make this step simpler. Secondly, you
need to have ``setuptools-rust`` installed in your python environment. This can
can be done by simply running::

  pip install setuptools-rust

prior to running::

  pip install retworkx

If you have rust nightly properly installed pip will compile retworkx for your
local system and it should run just as the prebuilt binaries would.

.. _rustup: https://rustup.rs/

Building from source
--------------------

The first step for building retworkx from source is to clone it locally with::

  git clone https://github.com/mtreinish/retworkx.git

retworkx uses PyO3 and setuptools-rust to build the python interface.
Unfortunately, this means you need to use nightly rust because PyO3 only works
with nightly at this point. You can use `rustup`_ to install rust nightly.

.. _rustup: https://rustup.rs/

Once you have nightly rust and cargo installed you can easily install retworkx
into your python environment using pip. Once you have a local clone of the repo
you can install retworkx into your python env with::

  pip install .

Assuming your current working directory is the root of the repo. Otherwise
you can run::

  pip install $PATH_TO_REPO_ROOT

which will install it the same way. Then retworkx in your local python
environment. There are 2 things to note when doing this though, first if you
try to run python from the repo root using this method it will not work as you
expect. There is a name conflict in the repo root because of the local python
package shim used in building the package. Simply run your python scripts or
programs using retworkx outside of the repo root. The second issue is that any
local changes you make to the rust code will not be reflected live in the
python you'll need to recompile the source by rerunning pip install to have any
changes reflected in your python environment.

Using retworkx
--------------

Once you have retworkx installed you can use it by importing retworkx. All
the functions and the PyDAG class are off the root of the package. For example,
building a DAG and adding 2 nodes with an edge between them would be::

    import retworkx

    my_dag = retworkx.PyDAG()
    # add_node(), add_child(), and add_parent() return the node index
    # The sole argument here can be any python object
    root_node = my_dag.add_node("MyRoot")
    # The second and third arguments can be any python object
    my_dag.add_child(root_node, "AChild", ["EdgeData"])
