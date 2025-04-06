import nox

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
    session.run("stestr", "run", *session.posargs)

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
    session.run("ruff", "check", "rustworkx", "setup.py")
    session.run("cargo", "fmt", "--all", "--", "--check", external=True)
    session.run("python", "tools/find_stray_release_notes.py")

@nox.session(python=["3"])
def docs(session):
    install_rustworkx(session)
    session.install("-r", "docs/source/requirements.txt", "-c", "constraints.txt")
    session.run("python", "-m", "ipykernel", "install", "--user")
    session.run("jupyter", "kernelspec", "list")
    session.chdir("docs")
    session.run("sphinx-build", "-W", "-d", "build/.doctrees", "-b", "html", "source", "build/html", *session.posargs)

@nox.session(python=["3"])
def docs_clean(session):
    session.chdir("docs")
    session.run("rm", "-rf", "build", "source/apiref", external=True)

@nox.session(python=["3"])
def black(session):
    session.install(*[d for d in lint_deps if "black" in d])
    session.run("black", "rustworkx", "tests", *session.posargs)

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
