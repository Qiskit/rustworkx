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

Right now tests can only be run manually by invoking python's unittest (or
a compatible runner) directly from a python environment with retworkx compiled
and installed. To do this you run:

```bash
python -m unittest discover .
```

from `./tests` in the repo (adjust `.` accordingly if you are running it from
another directoy). Note that you can **not** run the tests from the root of the
repo, this is because retworkx packaging shim will conflict with imports from
retworkx the installed version of retworkx (which contains the compiled
extension).

