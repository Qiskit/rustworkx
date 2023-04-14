import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.stop_on_first_error = True

@nox.session(python=["3.7", "3.8", "3.9", "3.10", "3.11"])
def test(session):
    session.install("-r", "requirements.txt", "-c", "constraints.txt")
    session.run("python", "-m", "stestr", "run", *session.posargs)

@nox.session(python=["3"])
def lint(session):
    session.install("black~=22.0", "flake8", "setuptools-rust")
    session.run("black", "--check", "--diff", "../rustworkx", "../tests", "../retworkx", *session.posargs)
    session.run("flake8", "--per-file-ignores='../rustworkx/__init__.py:F405,F403'", "../setup.py", "../rustworkx", "../retworkx", ".")
    session.run("cargo", "fmt", "--all", "--", "--check")
    session.run("python", "tools/find_stray_release_notes.py")

@nox.session(python=["3"])
def docs(session):
    session.install("-r", "docs/source/requirements.txt")
    session.run("python", "-m", "ipykernel", "install", "--user")
    session.run("jupyter", "kernelspec", "list")
    session.chdir("docs")
    session.run("sphinx-build", "-W", "-d", "build/.doctrees", "-b", "html", "source", "build/html", *session.posargs)

@nox.session(python=["3"])
def black(session):
    session.install("black~=22.0")
    session.run("black", "../rustworkx", "../tests", "../retworkx", *session.posargs)

@nox.session(python=["3"])
def stubs(session):
    session.install("mypy==1.0.1")
    session.run("python", "-m", "mypy.stubtest", "--concise", "--ignore-missing-stub", "rustworkx.rustworkx")