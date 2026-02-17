# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Shared utilities for rustworkx visualization modules."""

import subprocess
import tempfile


def has_graphviz() -> bool:
    """Check whether the graphviz ``dot`` command is available."""
    try:
        subprocess.run(
            ["dot", "-V"],
            cwd=tempfile.gettempdir(),
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False
