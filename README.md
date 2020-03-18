# retworkx

[![License](https://img.shields.io/github/license/Qiskit/retworkx.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://img.shields.io/travis/com/Qiskit/retworkx/master.svg?style=popout-square)](https://travis-ci.com/Qiskit/retworkx)[![](https://img.shields.io/github/release/Qiskit/retworkx.svg?style=popout-square)](https://github.com/Qiskit/retworkx/releases)[![](https://img.shields.io/pypi/dm/retworkx.svg?style=popout-square)](https://pypi.org/project/retworkx/)

  - You can see the full rendered docs at:
    <https://retworkx.readthedocs.io/en/latest/index.html>

retworkx is a rust graph library interface to python3. For right now
it's scope is as a replacement for
[qiskit-terra](https://github.com/Qiskit/qiskit-terra)'s previous (and
current) networkx usage (hence the name). The scope might grow or change
over time, but to start it's just about building a DAG and operating on
it with the performance and safety that Rust provides.

## Installing retworkx

retworkx is published on pypi so on x86\_64, i686, ppc64le, and aarch64
Linux systems, x86\_64 on Mac OSX, and 32 and 64 bit Windows installing
is as simple as running:

```bash
pip install retworkx
```

This will install a precompiled version of retworkx into your python
environment.

However, if there are no precompiled binaries published for your system
you'll have to compile the code. The source package is also published on
pypi so you can also run the above command to install it. However, there
are 2 preconditions for this to work, first you need to have cargo/rustc
**nightly** in your PATH. You can use [rustup](https://rustup.rs/) to
make this step simpler. Secondly, you need to have `setuptools-rust`
installed in your python environment. This can can be done by simply
running:

```bash
pip install setuptools-rust
```

prior to running:

```bash
pip install retworkx
```

If you have rust nightly properly installed pip will compile retworkx
for your local system and it should run just as the prebuilt binaries
would.

## Building from source

The first step for building retworkx from source is to clone it locally
with:

```bash
git clone https://github.com/Qiskit/retworkx.git
```

retworkx uses [PyO3](https://github.com/pyo3/pyo3) and
[setuptools-rust](https://github.com/PyO3/setuptools-rust) to build the
python interface. Unfortunately, this means you need to use nightly rust
because PyO3 only works with nightly at this point. You can use
[rustup](https://rustup.rs/) to install rust nightly.

Once you have nightly rust and cargo installed you can easily install
retworkx into your python environment using pip. Once you have a local
clone of the repo, change your current working directory to the root of
the repo. To set the compiler for `retworkx` to nightly rust using
`rustup`, run the following from this directory:

```bash
rustup override set nightly
```

(MacOS users shoud instead run `rustup default nightly` to enable
nightly rust globally.)

Then, you can install retworkx into your python env with:
```bash
pip install .
```

Assuming your current working directory is still the root of the repo.
Otherwise you can run:
```bash
pip install $PATH_TO_REPO_ROOT
```

which will install it the same way. Then retworkx is installed inyour
local python environment. There are 2 things to note when doing this
though, first if you try to run python from the repo root using this
method it will not work as you expect. There is a name conflict in the
repo root because of the local python package shim used in building the
package. Simply run your python scripts or programs using retworkx
outside of the repo root. The second issue is that any local changes you
make to the rust code will not be reflected live in the python you'll
need to recompile the source by rerunning pip install to have any
changes reflected in your python environment.

## Using retworkx

Once you have retworkx installed you can use it by importing retworkx.
All the functions and the PyDAG class are off the root of the package.
For example, building a DAG and adding 2 nodes with an edge between them
would be:

```python
import retworkx

my_dag = retworkx.PyDAG()
# add_node(), add_child(), and add_parent() return the node index
# The sole argument here can be any python object
root_node = my_dag.add_node("MyRoot")
# The second and third arguments can be any python object
my_dag.add_child(root_node, "AChild", ["EdgeData"])
```
