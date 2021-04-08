# Contributing

First read the overall Qiskit project contribution guidelines. These are all
included in the Qiskit documentation:

https://qiskit.org/documentation/contributing_to_qiskit.html

While it's not all directly applicable since most of it is about the Qiskit
project itself and retworkx is an independent library developed in tandem
with Qiskit; the general guidelines and advice still apply here.

## Contributing to retworkx

In addition to the general guidelines there are specific details for
contributing to retworkx, these are documented below.

### Tests

Once you've made a code change, it is important to verify that your change
does not break any existing tests and that any new tests that you've added
also run successfully. Before you open a new pull request for your change,
you'll want to run the test suite locally.

The easiest way to run the test suite is to use
[**tox**](https://tox.readthedocs.io/en/latest/#). You can install tox
with pip: `pip install -U tox`. Tox provides several advantages, but the
biggest one is that it builds an isolated virtualenv for running tests. This
means it does not pollute your system python when running. However, by default
tox will recompile retworkx from source every time it is run even if there
are no changes made to the rust code. To avoid this you can use the
`--skip-pkg-install` package if you'd like to rerun tests without recompiling.
Note, you only want to use this flag if you recently ran tox and there are no
rust code (or packaged python code) changes to the repo since then. Otherwise
the retworkx package tox installs in it's virtualenv will be out of date (or
missing).

Note, if you run tests outside of tox that you can **not** run the tests from
the root of the repo, this is because retworkx packaging shim will conflict
with imports from retworkx the installed version of retworkx (which contains
the compiled extension).

#### Running subsets of tests

If you just want to run a subset of tests you can pass a selection regex to the
test runner. For example, if you want to run all tests that have "dag" in the
test id you can run: `tox -epy -- dag`. You can pass arguments directly to the
test runner after the bare `--`. To see all the options on test selection you
can refer to the stestr manual:

https://stestr.readthedocs.io/en/stable/MANUAL.html#test-selection

If you want to run a single test module, test class, or individual test method
you can do this faster with the `-n`/`--no-discover` option. For example:

to run a module:
```
tox -epy -- -n test_max_weight_matching
```
or to run the same module by path:
```
tox -epy -- -n graph/test_nodes.py
```
to run a class:
```
tox -epy -- -n graph.test_nodes.TestNodes
```
to run a method:
```
tox -epy -- -n graph.test_nodes.TestNodes.test_no_nodes
```

It's important to note that tox will be running from the `tests/` directory in
the repo, so any paths you pass to the test runner via path need to be relative
to that directory.

#### Visualization Tests

When running the visualization tests, each test will generate a visualization
and only fail if an exception is raised by the call. Each test saves the output
image to the current working directory (which if running tests with `tox` is
`tests/`) to ensure the generated image is usable. However to not clutter the
system each test cleans up this generated image and by default a test run does
not include any way to view the images from the visualization tests.

If you want to inspect the output from the visualization tests (which is common
if you're working on visualizations) you can set the
`RETWORKX_TEST_PRESERVE_IMAGES` environment variable to any value and this will
skip the cleanup. This will enable you to look at the output image and ensure the
visualization is correct. For example, running:

```
RETWORKX_TEST_PRESERVE_IMAGES=1 tox -epy
```

will run the visualization tests and preserve the generated image files after
the run finishes so you can inspect the output.

### Style

#### Rust

Rust is the primary language of retworkx and all the functional code in the
libraries is written in Rust. The Rust code in retworkx uses
[rustfmt](https://github.com/rust-lang/rustfmt) to enforce consistent style.
CI jobs are configured to ensure to check this. Luckily adapting your code is
as simple as running:

```bash
cargo fmt
```

locally. This will automatically restyle the rust code in retworkx to match
what CI is checking.

##### Lint

An additional step is to run [clippy](https://github.com/rust-lang/rust-clippy)
on your changes. While this is not run in CI (because it's very dependent on
the rust/cargo version) it can often catch issues in your code. You can run it
by running:

```bash
cargo clippy
```

#### Python

Python is used primarily for tests and some small pieces of packaging
and namespace configuration code in the actual library.
[flake8](https://flake8.pycqa.org/en/latest/) is used to enforce consistent
style in the python code in the repository. You can run it via tox using:

```bash
tox -elint
```

This will also run `cargo fmt` in check mode to ensure that you ran `cargo fmt`
and will fail if the Rust code doesn't conform to the style rules.

### Building documentation

Just like with tests building documentation is done via tox. This will handle
compiling retworkx, installing the python dependencies, and then building the
documentation in an isolated venv. You can run just the docs build with:
```
tox -edocs
```
which will output the html rendered documentation in `docs/build/html` which
you can view locally in a web browser.

### Release Notes

It is important to document any end user facing changes when we release a new
version of retworkx.  The expectation is that if your code contribution has
user facing changes that you will write the release documentation for these
changes. This documentation must explain what was changed, why it was changed,
and how users can either use or adapt to the change. The idea behind release
documentation is that when a naive user with limited internal knowledge of the
project is upgrading from the previous release to the new one, they should be
able to read the release notes, understand if they need to update their
program which uses retworkx, and how they would go about doing that. It
ideally should explain why they need to make this change too, to provide the
necessary context.

To make sure we don't forget a release note or if the details of user facing
changes over a release cycle we require that all user facing changes include
documentation at the same time as the code. To accomplish this we use the
[reno](https://docs.openstack.org/reno/latest/) tool which enables a git based
workflow for writing and compiling release notes.

#### Adding a new release note

Making a new release note is quite straightforward. Ensure that you have reno
installed with::

    pip install -U reno

Once you have reno installed you can make a new release note by running in
your local repository checkout's root::

    reno new short-description-string

where short-description-string is a brief string (with no spaces) that describes
what's in the release note. This will become the prefix for the release note
file. Once that is run it will create a new yaml file in releasenotes/notes.
Then open that yaml file in a text editor and write the release note. The basic
structure of a release note is restructured text in yaml lists under category
keys. You add individual items under each category and they will be grouped
automatically by release when the release notes are compiled. A single file
can have as many entries in it as needed, but to avoid potential conflicts
you'll want to create a new file for each pull request that has user facing
changes. When you open the newly created file it will be a full template of
the different categories with a description of a category as a single entry
in each category. You'll want to delete all the sections you aren't using and
update the contents for those you are. For example, the end result should
look something like::

```yaml
features:
  - |
    Added a new function, :func:`~retworkx.foo` that adds support for doing
    something to :class:`~retworkx.PyDiGraph` objects.
  - |
    The :class:`~retworkx.PyDiGraph` class has a new method
    :meth:`~retworkx.PyDiGraph.foo``. This is the equivalent of calling the
    :func:`~retworkx.foo` function to do something to your
    :class:`~retworkx.PyDiGraph` object, but provides the convenience of running
    it natively on an object. For example::

      from retworkx import PyDiGraph

      g = PyDiGraph.
      g.foo()

deprecations:
  - |
    The ``retworkx.bar`` function has been deprecated and will be removed in a
    future release. It has been superseded by the
    :meth:`~retworkx.PyDiGraph.foo` method and :func:`~retworkx.foo` function
    which provides similar functionality but with more accurate results and
    better performance. You should update your calls
    ``retworkx.bar()`` calls to use ``retworkx.foo()`` instead.
```

You can also look at other release notes for other examples.

You can use any
[sphinx feature](https://www.sphinx-doc.org/en/3.x/usage/restructuredtext/)
in them (code sections, tables, enumerated lists, bulleted list, etc) to express
what is being changed as needed. In general you want the release notes to
include as much detail as needed so that users will understand what has changed,
why it changed, and how they'll have to update their code.

After you've finished writing your release notes you'll want to add the note
file to your commit with `git add` and commit them to your PR branch to make
sure they're included with the code in your PR.

##### Linking to issues

If you need to link to an issue or other Github artifact as part of the release
note this should be done using an inline link with the text being the issue
number. For example you would write a release note with a link to issue 12345
as:

```yaml
fixes:
  - |
    Fixes a race condition in the function ``foo()``. Refer to
    `#12345 <https://github.com/Qiskit/retworkx/issues/12345>`__ for more
    details.
```

#### Generating the release notes

After release notes have been added if you want to see what the full output of
the release notes. Reno is used to combine the release note yaml files into a
single rst (ReStructuredText) document that
[sphinx](https://www.sphinx-doc.org/en/master/) will then compile for us as part
of the documentation builds. If you want to generate the rst file you
use the ``reno report`` command. If you want to generate the full retworkx
release notes for all releases (since we started using reno during 0.8) you just
run::

    reno report

but you can also use the ``--version`` argument to view a single release (after
it has been tagged::

    reno report --version 0.8.0

#### Building release notes locally

Building the release notes is part of the standard retworkx documentation
builds. To check what the rendered html output of the release notes will look
like for the current state of the repo you can run: `tox -edocs` which will
build all the documentation into `docs/_build/html` and the release notes in
particular will be located at `docs/_build/html/release_notes.html`
