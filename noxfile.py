import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.stop_on_first_error = True
nox.options.download_python = 'never'

pyproject = nox.project.load_toml("pyproject.toml")

deps = nox.project.dependency_groups(pyproject, "test")
lint_deps = nox.project.dependency_groups(pyproject, "lint")
stubs_deps = nox.project.dependency_groups(pyproject, "stubs")

requires_python = pyproject["project"]["requires-python"]
supported_python_versions = nox.project.python_versions(pyproject)
min_python_version = min(supported_python_versions)

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

@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
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
    session.run("python", "-m", "mypy.stubtest", "--concise", "rustworkx", "--allowlist", "stubs_allowlist.txt")

@nox.session(
    python=["3"], 
    venv_backend="uv", 
    reuse_venv=False,
    default=False, 
)
def compile_locks(session):    
    from pathlib import Path
    print("Supported Python versions:", supported_python_versions)
    
    dependency_groups = pyproject.get("dependency-groups")
        
    # Ensure requirements directory exists
    requirements_dir = Path("requirements")
    requirements_dir.mkdir(exist_ok=True)
    
    # Process each dependency group
    for group_name in dependency_groups.keys():
        session.log(f"Compiling dependency group: {group_name}")
        
        # Ensuring group directory exists
        group_dir = requirements_dir / group_name
        group_dir.mkdir(exist_ok=True)
        
        # Output file path
        output_file = group_dir / "pylock.toml"
        
        # Run uv pip compile
        uv_python = f"python{min_python_version}"
        session.run(
            "uv",
            "pip",
            "compile",
            "--python",
            uv_python,
            "--group",
            group_name,
            "-o",
            str(output_file),
        )
        
        session.log(f"✓ Created {output_file}")