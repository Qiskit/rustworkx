# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Backwards compatibility shim for retworkx -> rustworkx."""


import sys
import warnings

from rustworkx import *  # noqa

from . import namespace

warnings.warn(
    "The retworkx package is deprecated and has been renamed to rustworkx. Rustworkx is a "
    "drop-in replacement and can be used by replacing `import retworkx` with import `rustworkx`. ",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules["retworkx.generators"] = generators  # noqa
new_meta_path_finder = namespace.RetworkxImport("retworkx", "rustworkx")
sys.meta_path = [new_meta_path_finder] + sys.meta_path
