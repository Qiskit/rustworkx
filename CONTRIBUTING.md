# Contributing

First read the overall project contributing guidelines. These are all
included in the qiskit documentation:

https://qiskit.org/documentation/contributing_to_qiskit.html

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
means it does not pollute your system python when running.

Note, if you run tests outside of tox that you can **not** run the tests from
the root of the repo, this is because retworkx packaging shim will conflict
with imports from retworkx the installed version of retworkx (which contains
the compiled extension).

### Building documentation

Just like with tests building documentation is done via tox. This will handle
compiling retworkx, installing the python dependencies, and then building the
documentation in an isolated venv. You can run just the docs build with:
```
tox -edocs
```
which will output the html rendered documentation in `docs/build/html` which
you can view locally in a web browser.
