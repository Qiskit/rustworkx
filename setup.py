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

# If RUST_DEBUG is set, force compiling in debug mode. Else, use the default behavior of whether
# it's an editable installation.
rustworkx_debug = True if os.getenv("RUSTWORKX_DEBUG") == "1" else None

RUST_EXTENSIONS = [RustExtension("rustworkx.rustworkx", "Cargo.toml",
                                 binding=Binding.PyO3, debug=rustworkx_debug)]
RUST_OPTS ={"bdist_wheel": {"py_limited_api": "cp39"}}

setup(
    rust_extensions=RUST_EXTENSIONS,
    zip_safe=False,
    options=RUST_OPTS,
)
