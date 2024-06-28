# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os

from setuptools import setup
from setuptools_rust import Binding, RustExtension


# If RUST_DEBUG is set, force compiling in debug mode. Else, use the default    behavior of whether
# it's an editable installation.
rustworkx_debug = True if os.getenv("RUSTWORKX_DEBUG") == "1" else None


def readme():
    with open("README.md") as f:
        return f.read()


mpl_extras = ["matplotlib>=3.0"]
graphviz_extras = ["pillow>=5.4"]

PKG_NAME = os.getenv("RUSTWORKX_PKG_NAME", "rustworkx")
PKG_VERSION = "0.16.0"
PKG_PACKAGES = ["rustworkx", "rustworkx.visualization"]
PKG_INSTALL_REQUIRES = ["numpy>=1.16.0,<3"]
RUST_EXTENSIONS = [RustExtension("rustworkx.rustworkx", "Cargo.toml",
                                 binding=Binding.PyO3, debug=rustworkx_debug)]
RUST_OPTS ={"bdist_wheel": {"py_limited_api": "cp38"}}

retworkx_readme_compat = """# retworkx

retworkx is the **deprecated** package name for `rustworkx`. If you're using
the `retworkx` package (either as a requirement or an import) this should
be updated to use rustworkx instead. In the future only the `rustworkx` name
will be supported.

"""


README = readme()
if PKG_NAME == "retworkx":
    README = retworkx_readme_compat + README
    PKG_PACKAGES = ["retworkx"]
    # TODO: For final retworkx release change this to < 1.
    PKG_INSTALL_REQUIRES.append(f"rustworkx=={PKG_VERSION}")
    RUST_EXTENSIONS = []

setup(
    name=PKG_NAME,
    version=PKG_VERSION,
    description="A python graph library implemented in Rust",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Matthew Treinish",
    author_email="mtreinish@kortar.org",
    license="Apache 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="Networks network graph Graph Theory DAG",
    url="https://github.com/Qiskit/rustworkx",
    project_urls={
        "Bug Tracker": "https://github.com/Qiskit/rustworkx/issues",
        "Source Code": "https://github.com/Qiskit/rustworkx",
        "Documentation": "https://www.rustworkx.org/",
    },
    rust_extensions=RUST_EXTENSIONS,
    include_package_data=True,
    packages=PKG_PACKAGES,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=PKG_INSTALL_REQUIRES,
    extras_require={
        "mpl": mpl_extras,
        "graphviz": graphviz_extras,
        "all": mpl_extras + graphviz_extras,
    },
    options=RUST_OPTS,
)
