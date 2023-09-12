import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.stop_on_first_error = True

deps = [
  "setuptools-rust",
  "fixtures",
  "testtools>=2.5.0",
  "networkx>=2.5",
  "scipy>=1.7",
  "stestr",
]

extras = [
  "mpl",
  "graphviz",
]

lint_deps = [
    "black~=22.0",
    "flake8",
    "setuptools-rust",
    "Flake8-pyproject==1.2.3",
]

@nox.session(python=["3"])
def test(session):
    session.install(*deps)
    session.install(".", "-c", "constraints.txt")
    session.chdir("tests")
    session.run("python", "-m", "stestr", "run", *session.posargs)

@nox.session(python=["3"])
def lint(session):
    session.install(*deps)
    session.install(".", "-c", "constraints.txt")
    session.install(*lint_deps)
    session.run("black", "--check", "--diff", "rustworkx", "tests", "retworkx", *session.posargs)
    session.run("flake8p", "--per-file-ignores='rustworkx/__init__.py':F405,F403", "setup.py", "rustworkx", "retworkx")
    session.run("cargo", "fmt", "--all", "--", "--check", external=True)
    session.run("python", "tools/find_stray_release_notes.py")

@nox.session(python=["3"])
def docs(session):
    session.install(*deps)
    session.install(".", "-c", "constraints.txt")
    session.install("-r", "docs/source/requirements.txt", "-c", "constraints.txt")
    session.run("python", "-m", "ipykernel", "install", "--user")
    session.run("jupyter", "kernelspec", "list")
    session.chdir("docs")
    session.run("sphinx-build", "-W", "-d", "build/.doctrees", "-b", "html", "source", "build/html", *session.posargs)

@nox.session(python=["3"])
def black(session):
    session.install(*[d for d in lint_deps if "black" in d])
    session.run("black", "rustworkx", "tests", "retworkx", *session.posargs)

@nox.session(python=["3"])
def stubs(session):
    session.install(*deps)
    session.install(".", "-c", "constraints.txt")
    session.install("mypy==1.0.1")
    session.chdir("tests")
    session.run("python", "-m", "mypy.stubtest", "--concise", "--ignore-missing-stub", "rustworkx.rustworkx")
