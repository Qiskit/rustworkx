import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.stop_on_first_error = True

deps = [
  "setuptools-rust",
  "fixtures",
  "testtools>=2.5.0",
  "networkx>=2.5",
  "stestr",
]

extras = [
  "mpl",
  "graphviz",
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
    session.install("black~=22.0", "flake8", "setuptools-rust")
    session.run("black", "--check", "--diff", "rustworkx", "tests", "retworkx", *session.posargs)
    session.run("flake8", "--per-file-ignores='rustworkx/__init__.py':F405,F403", "setup.py", "rustworkx", "retworkx", ".")
    session.run("cargo", "fmt", "--all", "--", "--check")
    session.run("python", "tools/find_stray_release_notes.py")

@nox.session(python=["3"])
def docs(session):
    session.install(*deps)
    session.install(".", "-c", "constraints.txt")
    session.install("-r", "docs/source/requirements.txt")
    session.run("python", "-m", "ipykernel", "install", "--user")
    session.run("jupyter", "kernelspec", "list")
    session.chdir("docs")
    session.run("sphinx-build", "-W", "-d", "build/.doctrees", "-b", "html", "source", "build/html", *session.posargs)

@nox.session(python=["3"])
def black(session):
    session.install("black~=22.0")
    session.run("black", "rustworkx", "tests", "retworkx", *session.posargs)

@nox.session(python=["3"])
def stubs(session):
    session.install(*deps)
    session.install(".", "-c", "constraints.txt")
    session.install("mypy==1.0.1")
    session.run("python", "-m", "mypy.stubtest", "--concise", "--ignore-missing-stub", "rustworkx.rustworkx")

@nox.session(python=["3"])
def black(session):
    session.install("black~=22.0")
    session.run("black", "rustworkx", "tests", "retworkx", *session.posargs)