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

import retworkx


class TestRandomLayout(unittest.TestCase):

    def setUp(self):
        self.graph = retworkx.generators.directed_path_graph(10)

    def test_random_layout(self):
        res = retworkx.digraph_random_layout(self.graph, seed=42)
        expected = {
            0: (0.2265125179283135, 0.23910669031859955),
            4: (0.8025885957751138, 0.37085692752109345),
            5: (0.23635127852185123, 0.9286365888207462),
            1: (0.760833410686741, 0.5278396573581516),
            3: (0.1879083014236631, 0.524657662927804),
            2: (0.9704763177409157, 0.37546268141451944),
            6: (0.462700947802672, 0.44025745918644743),
            7: (0.3125895420208278, 0.0893209773065271),
            8: (0.5567725240957387, 0.21079648777222115),
            9: (0.7586719404939911, 0.43090704138697045)
        }
        self.assertEqual(expected, res)

    def test_random_layout_center(self):
        res = retworkx.digraph_random_layout(self.graph, center=(0.5, 0.5),
                                             seed=42)
        expected = {
            1: [1.260833410686741, 1.0278396573581516],
            5: [0.7363512785218512, 1.4286365888207462],
            7: [0.8125895420208278, 0.5893209773065271],
            4: [1.3025885957751138, 0.8708569275210934],
            8: [1.0567725240957389, 0.7107964877722212],
            9: [1.2586719404939912, 0.9309070413869704],
            0: [0.7265125179283135, 0.7391066903185995],
            2: [1.4704763177409157, 0.8754626814145194],
            6: [0.962700947802672, 0.9402574591864474],
            3: [0.6879083014236631, 1.0246576629278041]
        }
        self.assertEqual(expected, res)

    def test_random_layout_no_seed(self):
        res = retworkx.digraph_random_layout(self.graph)
        # Random output, just assert  structurally correct
        self.assertIsInstance(res, retworkx.Pos2DMapping)
        self.assertEqual(len(res), 10)
        self.assertEqual(len(res[0]), 2)
        self.assertIsInstance(res[0][0], float)
