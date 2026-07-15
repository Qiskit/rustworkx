# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Typing regression tests for ``graphviz_draw`` (issue #1357).

Issue #1357 was that ``graphviz_draw`` was annotated/documented as returning
the ``PIL.Image`` *module* instead of the ``PIL.Image.Image`` *class*. This
file guards against that regression on two independent fronts:

* ``TestGraphvizDrawReturnTypeAnnotation`` -- a **runtime** guard over the
  implementation's own ``__annotations__`` and docstring. This is the part that
  actually catches the reported bug: the ``graphviz.pyi`` stub shadows
  ``graphviz.py`` for static type checkers, so a wrong annotation in the
  implementation is invisible to mypy/pyright but is exercised here via
  ``typing.get_type_hints``. Before the fix, the buggy ``Image | None``
  annotation makes ``get_type_hints`` raise ``TypeError`` (``module | None``).

* the ``TYPE_CHECKING`` block -- a **static** guard over the public *stub*,
  validated by mypy/pyright (see the ``stubs`` nox session). It exercises every
  overload so all return-type branches stay correct.
"""

from __future__ import annotations

import inspect
import unittest
from typing import TYPE_CHECKING, get_args, get_type_hints

import rustworkx
from rustworkx.visualization import graphviz_draw

if TYPE_CHECKING:
    # Type checkers only analyse this branch, so ``PILImageClass`` is always
    # bound to the class here (avoids "possibly unbound"/reassignment errors).
    from PIL.Image import Image as PILImageClass

    HAS_PIL = True
else:
    try:
        from PIL.Image import Image as PILImageClass

        HAS_PIL = True
    except ImportError:
        PILImageClass = None
        HAS_PIL = False


@unittest.skipUnless(HAS_PIL, "pillow is required for these tests")
class TestGraphvizDrawReturnTypeAnnotation(unittest.TestCase):
    """Runtime guard for the implementation's return type (issue #1357)."""

    def test_return_annotation_resolves_to_image_class(self) -> None:
        # get_type_hints evaluates the (stringized) annotation in the module
        # namespace. With the bug it references the PIL.Image *module* and
        # ``module | None`` raises TypeError; the fixed annotation resolves to
        # the PIL.Image.Image *class*.
        hints = get_type_hints(graphviz_draw)
        return_args = get_args(hints["return"])
        self.assertIn(PILImageClass, return_args)
        # None is a valid branch (returned when ``filename`` is provided).
        self.assertIn(type(None), return_args)
        # Guard specifically against regressing back to the module: no union
        # member should be a module (the pre-fix ``PIL.Image`` bug).
        self.assertFalse(any(inspect.ismodule(arg) for arg in return_args))

    def test_rtype_docstring_uses_image_class(self) -> None:
        doc = graphviz_draw.__doc__ or ""
        self.assertIn(":rtype: PIL.Image.Image", doc)
        # The pre-fix docstring had a bare ``:rtype: PIL.Image`` (module).
        self.assertNotIn(":rtype: PIL.Image\n", doc)


if TYPE_CHECKING:
    # Static guard for the public *stub* (rustworkx/visualization/graphviz.pyi).
    # Validated by mypy/pyright, never executed at runtime, so it needs neither
    # graphviz nor Pillow installed.
    from PIL.Image import Image
    from typing_extensions import assert_type

    graph: rustworkx.PyGraph[int, int] = rustworkx.PyGraph()
    digraph: rustworkx.PyDiGraph[int, int] = rustworkx.PyDiGraph()

    def _check_overload_return_types() -> None:
        # --- filename omitted / None -> returns a PIL.Image.Image instance ---
        assert_type(graphviz_draw(graph), Image)
        assert_type(graphviz_draw(digraph), Image)
        # image_type as a known Literal member
        assert_type(graphviz_draw(graph, image_type="png"), Image)
        # image_type as an arbitrary str (matches the broad str overload)
        arbitrary_type: str = "png"
        assert_type(graphviz_draw(graph, image_type=arbitrary_type), Image)
        # method kwarg does not change the return type
        assert_type(graphviz_draw(graph, method="neato"), Image)

        # --- filename provided -> writes to disk and returns None ---
        assert_type(graphviz_draw(graph, filename="out.png"), None)
        assert_type(graphviz_draw(digraph, filename="out.png"), None)
        assert_type(graphviz_draw(graph, filename="out.png", image_type="svg"), None)
        arbitrary_filename: str = "out.png"
        assert_type(graphviz_draw(graph, filename=arbitrary_filename), None)


if __name__ == "__main__":
    unittest.main()
