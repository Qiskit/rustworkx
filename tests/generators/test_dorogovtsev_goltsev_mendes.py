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

import unittest

import rustworkx


class TestDorogovtsevGoltsevMendesGraph(unittest.TestCase):
    def test_dorogovtsev_goltsev_mendes_graph(self):
        for n in range(0, 6):
            graph = rustworkx.generators.dorogovtsev_goltsev_mendes_graph(n)
            self.assertEqual(len(graph), (3 ** n + 3) // 2)
            self.assertEqual(len(graph.edges()), 3 ** n)
