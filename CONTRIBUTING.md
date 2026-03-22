# Contributing

First read the overall Qiskit project contribution guidelines. These are all
included in the Qiskit documentation:

https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md

While it's not all directly applicable since most of it is about the Qiskit
project itself and rustworkx is an independent library developed in tandem
with Qiskit; the general guidelines and advice still apply here.

## Contributing to rustworkx

In addition to the general guidelines there are specific details for
contributing to rustworkx, these are documented below.

### Making changes to the code

Rustworkx is implemented primarily in Rust with a thin layer of Python.
Because of that, most of your code changes will involve modifications to
Rust files in `src`. To understand which files you need to change, we invite
you for an overview of our simplified source tree:

```
├── src/
│   ├── lib.rs
│   ├── tiny.rs
│   ├── large/
│   │   ├── mod.rs
│   │   ├── pure_rust_code.rs
│   │   └── more_pure_rust_code.rs
```

#### Module exports in `lib.rs`

To add new functions, you will need to export them in `lib.rs`. `lib.rs` will
import functions defined in Rust modules (see the next section), and export
them to Python using `m.add_wrapped(wrap_pyfunction!(your_new_function))?;`

#### Adding and changing functions in modules

To add and change functions, you will need to modify module files. Modules contain pyfunctions
that will be exported, and can be defined either as a single file such as `tiny.rs` or as a
directory with `mod.rs` such as `large/`.

Rust functions that are exported to Python are annotated with `#[pyfunction]`. The
annotation gives them power to interact both with the Python interpreter and pure
Rust code. To change an existing function, search for its name and edit the code that
already exists.

If you want to add a new function, find the module you'd like to insert it in
or create a new one like `your_module.rs`. Then, start with the boilerplate bellow:

```rust
/// Docstring containing description of the function
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn your_new_function(
    py: Python,
    graph: &graph::PyGraph,
) -> PyResult<()> {
    /* Your code goes here */
}
```

> __NOTE:__  If you create a new `your_module.rs`, remember to declare and import it in `lib.rs`:
> ```rust
> mod your_module;
> use your_module::*;
> ```

#### Module directories: when a single file is not enough

Sometimes you will find that it is hard to organize a module in a tiny
file like `tiny.rs`. In those cases, we suggest moving the files to a directory
and splitting them following the structure of `large/`.

Module directories have a `mod.rs` file containing the pyfunctions. The pyfunctions
in that file then delegate most of logic by importing and calling pure Rust code from
`pure_rust_code.rs` and `more_pure_rust_code.rs`.

> __NOTE:__ Do you still have questions about making your contribution?
> Contact us at the [\#rustworkx channel in Qiskit Slack](https://qiskit.slack.com/messages/rustworkx/)

### rustworkx-core

If you're working on writing a pure rust function and it can be made generic
such that it works for any petgraph graph (if applicable) and that it has
no dependency on Python or pyo3, it probably makes sense in `rustworkx-core`.
`rustworkx-core` is a standalone rust library that's used to provide a Rust API
to both rustworkx and other rust applications or libraries. Unlike rustworkx
it's a Rust library and not a Python library and is designed to be an add-on
library on top of petgraph that provides additional graph algorithms and
functionality.

When contributing to rustworkx-core the key differences to keep in mind are that
the public rust interface needs to be treated as a stable interface, which is
different from rustworkx where the stable rust interface compatibility doesn't
matter only the exported Python API. Additionally documentation and testing
should be done via cargo doc and cargo test. It is expected that any new
functionality or changes to rustworkx-core is also being used by rustworkx so
test coverage is needed both via python in the rustworkx tests and via the
rustworkx-core rust interface.

### Tests

Once you've made a code change, it is important to verify that your change
does not break any existing tests and that any new tests that you've added
also run successfully. Before you open a new pull request for your change,
you'll want to run the test suite locally.

The easiest way to run the test suite is to use
[**Nox**](https://nox.thea.codes/en/stable/). You can install Nox
with pip: `pip install -U "nox[uv]"`. Nox provides several advantages, but the
biggest one is that it builds an isolated virtualenv for running tests. This
means it does not pollute your system python when running. However, by default
Nox will recompile rustworkx from source every time it is run even if there
are no changes made to the rust code. To avoid this you can use the
`--no-install` package if you'd like to rerun tests without recompiling.
Note, you only want to use this flag if you recently ran Nox and there are no
rust code (or packaged python code) changes to the repo since then. Otherwise
the rustworkx package Nox installs in its virtualenv will be out of date (or
missing).

Note, if you run tests outside of Nox that you can **not** run the tests from
the root of the repo, this is because rustworkx packaging shim will conflict
with imports from rustworkx the installed version of rustworkx (which contains
the compiled extension).

#### Running tests with a specific Python version

If you want to run the tests with a specific version of Python, use the `test_with_version`
target. For example, to launch a test with version 3.11 the command is:

```python
nox --python 3.11 -e test_with_version
```

#### Running subsets of tests

If you just want to run a subset of tests you can pass a selection regex to the
test runner. For example, if you want to run all tests that have "dag" in the
test id you can run: `nox -e test -- dag`. You can pass arguments directly to the
test runner after the bare `--`. To see all the options on test selection you
can refer to the stestr manual:

https://stestr.readthedocs.io/en/stable/MANUAL.html#test-selection

If you want to run a single test module, test class, or individual test method
you can do this faster with the `-n`/`--no-discover` option. For example:

to run a module:
```
nox -e test -- -n test_max_weight_matching
```
or to run the same module by path:
```
nox -e test -- -n graph/test_nodes.py
```
to run a class:
```
nox -e test -- -n graph.test_nodes.TestNodes
```
to run a method:
```
nox -e test -- -n graph.test_nodes.TestNodes.test_no_nodes
```

It's important to note that Nox will be running from the `tests/` directory in
the repo, so any paths you pass to the test runner via path need to be relative
to that directory.

#### Visualization Tests

When running the visualization tests, each test will generate a visualization
and only fail if an exception is raised by the call. Each test saves the output
image to the current working directory (which if running tests with `nox` is
`tests/`) to ensure the generated image is usable. However to not clutter the
system each test cleans up this generated image and by default a test run does
not include any way to view the images from the visualization tests.

If you want to inspect the output from the visualization tests (which is common
if you're working on visualizations) you can set the
`RUSTWORKX_TEST_PRESERVE_IMAGES` environment variable to any value and this will
skip the cleanup. This will enable you to look at the output image and ensure the
visualization is correct. For example, running:

```
RUSTWORKX_TEST_PRESERVE_IMAGES=1 nox -e test
```

will run the visualization tests and preserve the generated image files after
the run finishes so you can inspect the output.

#### rustworkx-core tests

As rustworkx-core is a standalone rust crate with its own public interface it
needs its own testing. These tests can be a combination of doc tests (embedded
code examples in the docstrings in the rust code) or standalone tests. You
can refer to the rust book on how to add tests:

https://doc.rust-lang.org/book/ch11-01-writing-tests.html

The rustworkx-core tests can be run with:
```
cargo test --workspace
```

### Fuzz Testing in rustworkx

We use cargo-fuzz to test rustworkx for unexpected crashes or undefined behavior. Follow these steps to run fuzzing locally.

#### Building Fuzz Targets
To build the fuzzing targets, first install cargo-fuzz:
```sh
cargo install cargo-fuzz
```
then run the following command:

```sh
cargo fuzz build
```
#### To run a fuzz test (e.g., test_traversal_node_coverage):

List all the targets:
```sh
cargo fuzz list
```
Run the tests from the list
```sh
cargo fuzz run test_traversal_node_coverage
```
For nightly toolchain:
```sh
cargo +nightly fuzz run test_traversal_node_coverage
```
Limit fuzz testing to a specific time (e.g., 60 seconds):
```sh
cargo +nightly fuzz run test_traversal_node_coverage -- -max_total_time=60
```
#### Interpreting Failures
Failures are stored in the fuzz/artifacts/ directory.

#### Contributing to Fuzzing

Add Fuzz Targets: Create new targets in the fuzz directory.
Fix Failures: Investigate and fix bugs found by fuzz tests.

Fuzz tests can be resource-heavy. Run them locally to save resources.
Submit fuzz tests with detailed documentation and commit messages.

### Style

#### Rust

Rust is the primary language of rustworkx and all the functional code in the
libraries is written in Rust. The Rust code in rustworkx uses
[rustfmt](https://github.com/rust-lang/rustfmt) to enforce consistent style.
CI jobs are configured to ensure to check this. Luckily adapting your code is
as simple as running:

```bash
cargo fmt
```

locally. This will automatically restyle the rust code in rustworkx to match
what CI is checking.

##### Lint

An additional step is to run [clippy](https://github.com/rust-lang/rust-clippy)
on your changes. You can execute it by running:

```bash
cargo clippy
```

If you want a more detailed feedback identical to CI, run instead:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

#### Python

Python is used primarily for tests and some small pieces of packaging
and namespace configuration code in the actual library.
[ruff](https://github.com/astral-sh/ruff) is used to enforce consistent
style in the python code in the repository. You can run them via Nox using:

```bash
nox -e lint
```

This will also run `cargo fmt` in check mode to ensure that you ran `cargo fmt`
and will fail if the Rust code doesn't conform to the style rules.

If ruff returns a code formatting error you can run `nox -e format` to automatically
update the code formatting to conform to the style.

### Building documentation

Just like with tests building documentation is done via Nox. This will handle
compiling rustworkx, installing the python dependencies, and then building the
documentation in an isolated venv. 

Our documentation setup requires that the [uv](https://github.com/astral-sh/uv)
backend for Nox is installed. That can be done with `pip install -U "nox[uv]"`.

You can run just the docs build with:
```
nox -e docs
```
which will output the html rendered documentation in `docs/build/html` which
you can view locally in a web browser.

> [!TIP]
> If you run `nox -e docs -- -j auto`, the documentation uses all CPUs and builds faster.

#### rustworkx-core documentation

To build the rustworkx-core documentation you will use rust-doc. You can do this
by running:
```
cargo doc -p rustworkx-core
```
After it's built the compiled documentation will be located in
`target/doc/rustworkx_core` (which is off the repo root not the `rustworkx-core`
dir)

You can build and open the documentation directly in your configured default
web browser by running:

```
cargo doc -p rustworkx-core --open
```

#### Updating documentation dependencies

The documentation workflow is currently our step with the most dependencies. Even
though `rustworkx` currently has very few Python dependencies, our documentation depends
on `sphinx` and many others. Therefore, `uv.lock` file contains a frozen list of what
can be used for the workflow.

If you need to add or remove dependencies, update `pyproject.toml` (specifically the `docs`
section in `dependency-groups`), and then run `uv sync --all-groups` to update `uv.lock`.

### Type Annotations

If you have added new methods, functions, or classes, and/or changed any
signatures, type annotations for Python are required to be included in a pull
request. Type annotations are added using type
[stub files](https://typing.readthedocs.io/en/latest/source/stubs.html) which
provide type annotations to python tooling which use type annotations. The stub
files are in the `rustworkx/` directory and have a `.pyi` file extension. They
contain annotated signatures for Python functions, stripped of their
implementation. You can find more details on typing in Python at:

 * https://mypy.readthedocs.io/en/stable/
 * https://typing.readthedocs.io/en/latest/
 * https://docs.python.org/3/library/typing.html

Having type annotations is very helpful for Python end-users. Adding
annotations lets users type check their code with [mypy](http://mypy-lang.org/),
which can be helpful for finding bugs when using rustworkx.

Just like with tests for the code, annotations are also tested via Nox.

```
nox -e stubs
```

One important thing to note is that if you're adding a new function to the Rust
module you will need to ensure that the signature with annotations is added to
`rustworkx/rustworkx.pyi`. Then it is also necessary to re-export the annotation
by adding an import line to `rustworkx/__init__.pyi` in the form:

```python
from .rustworkx import foo as foo
```

which ensures that mypy is able to find the type annotations when users import
from the root `rustworkx` package (which is the most common access pattern).

### Release Notes

It is important to document any end user facing changes when we release a new
version of rustworkx.  The expectation is that if your code contribution has
user facing changes that you will write the release documentation for these
changes. This documentation must explain what was changed, why it was changed,
and how users can either use or adapt to the change. The idea behind release
documentation is that when a naive user with limited internal knowledge of the
project is upgrading from the previous release to the new one, they should be
able to read the release notes, understand if they need to update their
program which uses rustworkx, and how they would go about doing that. It
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
    Added a new function, :func:`~rustworkx.foo` that adds support for doing
    something to :class:`~rustworkx.PyDiGraph` objects.
  - |
    The :class:`~rustworkx.PyDiGraph` class has a new method
    :meth:`~rustworkx.PyDiGraph.foo``. This is the equivalent of calling the
    :func:`~rustworkx.foo` function to do something to your
    :class:`~rustworkx.PyDiGraph` object, but provides the convenience of running
    it natively on an object. For example::

      from rustworkx import PyDiGraph

      g = PyDiGraph.
      g.foo()

deprecations:
  - |
    The ``rustworkx.bar`` function has been deprecated and will be removed in a
    future release. It has been superseded by the
    :meth:`~rustworkx.PyDiGraph.foo` method and :func:`~rustworkx.foo` function
    which provides similar functionality but with more accurate results and
    better performance. You should update your calls
    ``rustworkx.bar()`` calls to use ``rustworkx.foo()`` instead.
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
    `#12345 <https://github.com/Qiskit/rustworkx/issues/12345>`__ for more
    details.
```

#### Generating the release notes

After release notes have been added if you want to see what the full output of
the release notes. Reno is used to combine the release note yaml files into a
single rst (ReStructuredText) document that
[sphinx](https://www.sphinx-doc.org/en/master/) will then compile for us as part
of the documentation builds. If you want to generate the rst file you
use the ``reno report`` command. If you want to generate the full rustworkx
release notes for all releases (since we started using reno during 0.8) you just
run::

    reno report

but you can also use the ``--version`` argument to view a single release (after
it has been tagged::

    reno report --version 0.8.0

#### Building release notes locally

Building the release notes is part of the standard rustworkx documentation
builds. To check what the rendered html output of the release notes will look
like for the current state of the repo you can run: `nox -e docs` which will
build all the documentation into `docs/_build/html` and the release notes in
particular will be located at `docs/_build/html/release_notes.html`

### Pull request review, CI, and merge queue

After you've submitted a pull request to rustworkx it will need to pass CI and be
reviewed by an approved by a core team reviewer. CI runs get triggered
automatically when your pull request is opened and on every subsequent commit
made to your pull request's branch. Code review however may take some time,
sometimes even weeks or months, there are many new pull requests opened every
day and limited number of reviewers available, and while every proposed change
is a valuable addition to the project not everything is the highest priority.
You can help this process move more quickly by actively reviewing other open
PRs. While only members of the rustworkx core team have permission to provide
final approval and mark a PR as ready for merging, reviewing code is open to
everyone and all reviews are welcome and extremely valued contributions.
Helping with code review also helps reduce the burden on the core team and
enables them to review code faster.

The code review process is a bit of back and forth where you will receive
feedback and questions about your proposed changes to the project. You will
likely have multiple rounds of feedback with suggestions or changes requested
before approval. Please do not get discouraged as this is normal and part of
ensuring the quality of the rustworkx project and even what first appears as a
straightforward or simple change might have larger implications that aren't
obvious at first. If you receive feedback feel free to request re-review from
reviewers after you've adjusted your PR based on the comments received.

Another thing to keep in mind is that CI time is a constrained resource and not
infinite. While waiting for review and approval it is not necessary to keep your
PR branch up to date on every change to the `main` branch. Doing it periodically
is fine to make sure there are no regressions as the codebase changes, but
doing it too often will just needlessly waste CI resources. This will contribute
to resource starvation on CI, slowing down total throughput for the project. If
possible try to bundle updating your branch to the current HEAD on the `main`
branch with other changes made to the PR branch (like making adjustments from
code review). This will result in a single CI run instead of doing standalone
updates with no code changes.

Once your PR has the necessary approvals it will be tagged with the `automerge`
tag. This is a signal to the [mergify bot](https://mergify.io/) that the PR has
been approved and is ready for merging. The mergify bot will then enqueue the
PR onto its merge queue. At this point the process of updating a PR to the
current HEAD of the `main` branch is fully automated and once CI passes mergify
will merge the PR automatically. In an effort to conserve CI resources and
maximize throughput the mergify bot will only update a PR when it's next in the
merge queue. It might appear as activity on your PR is idle at this point, but
this likely just means the mergify merge queue is deep and/or CI has a backlog.
Do **not** manually update a PR branch to HEAD on the `main` branch after it
has the necessary approvals and is tagged as `automerge` unless it has a merge
conflict or has a failed CI run. Doing so will just waste CI resources and
delay everything from merging, including your PR.

### Stable Branch Policy and Backporting

The stable branch is intended to be a safe source of fixes for high-impact bugs,
documentation fixes, and security issues that have been fixed on main since a
release. When reviewing a stable branch PR, we must balance the risk of any given
patch with the value that it will provide to users of the stable branch. Only a
limited class of changes are appropriate for inclusion on the stable branch. A
large, risky patch for a major issue might make sense, as might a trivial fix
for a fairly obscure error-handling case. A number of factors must be weighed
when considering a change:

- The risk of regression: even the tiniest changes carry some risk of breaking
  something, and we really want to avoid regressions on the stable branch.
- The user visibility benefit: are we fixing something that users might actually
  notice, and if so, how important is it?
- How self-contained the fix is: if it fixes a significant issue but also
  refactors a lot of code, it’s probably worth thinking about what a less risky
  fix might look like.
- Whether the fix is already on main: a change must be a backport of a change
  already merged onto main, unless the change simply does not make sense on
  main.

Normally only bug fixes or non-code changes are allowed on a stable branch, the
primary exception to this is adding support for new python versions. If a new
python version is released backporting that feature change with that new support
is an acceptable backport.

In rustworkx at least until the 1.0 release we only maintain a single stable
branch at a time for the most recent minor version release.

#### Backporting procedure

In the normal case to backport a pull request all that needs to be done is
to tag it as `stable-backport-potential`, this will signal the
[mergify bot](https://mergify.io/) that the PR should be backported after it
merged. Once a PR tagged as `stable-backport-potential` merges mergify will
automatically open a new PR backporting it to the stable branch.

##### Manual backport procedure

If the mergify approach doesn't work for some reason and you need to manual
backport a PR this can be done with the following procedure. When backporting a
patch from main to stable, we want to keep a reference to the change on main.
When you create the branch for the stable PR, use:

```
$ git cherry-pick -x $main_commit_id
```

However, this only works for small self-contained patches from main. If you
need to backport a subset of a larger commit (from a squashed PR, for example)
from main, do this manually. In these cases, add:

```
Backported from: #main pr number
```

so that we can track the source of the change subset, even if a strict
cherry-pick doesn't make sense.

If the patch you’re proposing will not cherry-pick cleanly, you can help by
resolving the conflicts yourself and proposing the resulting patch. Please keep
Conflicts lines in the commit message to help review of the stable patch.
