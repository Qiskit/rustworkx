import nox
import os

nox.options.reuse_existing_virtualenvs = True
nox.options.stop_on_first_error = True

pyproject = nox.project.load_toml("pyproject.toml")

deps = nox.project.dependency_groups(pyproject, "test")
lint_deps = nox.project.dependency_groups(pyproject, "lint")
stubs_deps = nox.project.dependency_groups(pyproject, "stubs")

def install_rustworkx(session):
    session.install(*deps)
    session.install(".[all]", "-c", "constraints.txt")

# We define a common base such that -e test triggers a test with the current
# Python version of the interpreter and -e test_with_version launches
# a test with the specified version of Python.
def base_test(session):
    install_rustworkx(session)
    session.chdir("tests")
    # Convert file path style args (e.g. tests/test_graph6.py) to dotted module names
    # which stestr/unittest expects (e.g. tests.test_graph6). Leave other args alone.
    def _convert_arg(arg: str) -> str:
        if arg.endswith(".py"):
            p = os.path.normpath(arg)
            module = os.path.splitext(p)[0]
            # strip any leading ./
            if module.startswith("./"):
                module = module[2:]
            # replace path separators with dots
            module = module.replace(os.sep, ".")
            return module
        return arg

    converted = [_convert_arg(a) for a in session.posargs]
    # If we've chdir'ed into the tests directory, stestr expects module names
    # relative to that directory (e.g. 'test_graph6' instead of 'tests.test_graph6').
    def _strip_tests_prefix(name: str) -> str:
        if name.startswith("tests."):
            return name[len("tests."):]
        return name

    converted = [_strip_tests_prefix(c) for c in converted]
    session.run("stestr", "run", *converted)

@nox.session(python=["3"])
def test(session):
    base_test(session)

@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
def test_with_version(session):
    base_test(session)

@nox.session(python=["3"])
def lint(session):
    black(session)
    typos(session)
    session.install(*lint_deps)
    session.run("ruff", "check", "rustworkx", "retworkx", "setup.py")
    session.run("cargo", "fmt", "--all", "--", "--check", external=True)
    session.run("python", "tools/find_stray_release_notes.py")

# For uv environments, we keep the virtualenvs separate to avoid conflicts
@nox.session(python=["3"], venv_backend="uv", reuse_venv=False, default=False)
def docs(session):
    session.env["UV_PROJECT_ENVIRONMENT"] = session.virtualenv.location
    session.env["UV_FROZEN"] = "1"
    # faster build as generating docs already takes some time and we discard the env
    session.env["SETUPTOOLS_RUST_CARGO_PROFILE"] = "dev"
    session.run("uv", "sync", "--only-group", "docs")
    session.install(".")
    session.run(
        "uv", "run", "--", "python", "-m", "ipykernel", "install", "--user"
    )
    session.run("uv", "run", "jupyter", "kernelspec", "list")
    session.chdir("docs")
    session.run(
        "uv",
        "run",
        "sphinx-build",
        "-W",
        "-d",
        "build/.doctrees",
        "-b",
        "html",
        "source",
        "build/html",
        *session.posargs,
    )

@nox.session(python=["3"], default=False)
def docs_clean(session):
    session.chdir("docs")
    session.run("rm", "-rf", "build", "source/apiref", external=True)

@nox.session(python=["3"])
def black(session):
    session.install(*[d for d in lint_deps if "black" in d])
    session.run("black", "rustworkx", "tests", "retworkx", *session.posargs)

@nox.session(python=["3"])
def typos(session):
    session.install(*[d for d in lint_deps if "typos" in d])
    session.run("typos", "--exclude", "releasenotes")
    session.run("typos", "--no-check-filenames", "releasenotes")

@nox.session(python=["3"])
def stubs(session):
    install_rustworkx(session)
    session.install(*stubs_deps)
    session.chdir("tests")
    session.run("python", "-m", "mypy.stubtest", "--concise", "rustworkx")
