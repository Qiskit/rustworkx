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

# Based on the equivalent tests for the networkx matplotlib drawer:
# https://github.com/networkx/networkx/blob/ead0e65bda59862e329f2e6f1da47919c6b07ca9/networkx/drawing/tests/test_pylab.py

import os
import tempfile
import unittest

import retworkx
from retworkx.visualization import pydot_draw

try:
    import pydot
    import PIL

    pydot.call_graphviz("dot", ["--version"], tempfile.gettempdir())
    HAS_PYDOT = True
except Exception:
    HAS_PYDOT = False

SAVE_IMAGES = os.getenv("RETWORKX_TEST_PRESERVE_IMAGES", None)


def _save_image(image, path):
    if SAVE_IMAGES:
        image.save(path)


@unittest.skipUnless(
    HAS_PYDOT, "pydot and graphviz are required for running these tests"
)
class TestPyDotDraw(unittest.TestCase):
    def test_draw_no_args(self):
        graph = retworkx.generators.star_graph(24)
        image = pydot_draw(graph)
        self.assertIsInstance(image, PIL.Image.Image)
        _save_image(image, "test_pydot_draw.png")
