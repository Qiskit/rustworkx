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


class LayoutTest(unittest.TestCase):
    thres = 1e-6

    def assertLayoutEquiv(self, exp, res):
        for k in exp:
            ev = exp[k]
            rv = res[k]
            if abs(ev[0] - rv[0]) > self.thres or abs(ev[1] - rv[1]) > self.thres:
                self.fail(
                    f"The position for node {k}, {rv}, differs from the expected "
                    f"position, {ev} by more than the allowed threshold of {self.thres}"
                )


class TestRandomLayout(LayoutTest):
    def setUp(self):
        self.graph = rustworkx.generators.path_graph(10)

    def test_random_layout(self):
        res = rustworkx.graph_random_layout(self.graph, seed=42)
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
            9: (0.7586719404939911, 0.43090704138697045),
        }
        self.assertEqual(expected, res)

    def test_random_layout_center(self):
        res = rustworkx.graph_random_layout(self.graph, center=(0.5, 0.5), seed=42)
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
            3: [0.6879083014236631, 1.0246576629278041],
        }
        self.assertEqual(expected, res)

    def test_random_layout_no_seed(self):
        res = rustworkx.graph_random_layout(self.graph)
        # Random output, just assert  structurally correct
        self.assertIsInstance(res, rustworkx.Pos2DMapping)
        self.assertEqual(len(res), 10)
        self.assertEqual(len(res[0]), 2)
        self.assertIsInstance(res[0][0], float)


class TestBipartiteLayout(LayoutTest):
    def setUp(self):
        self.graph = rustworkx.generators.path_graph(10)

    def test_bipartite_layout_empty(self):
        res = rustworkx.bipartite_layout(rustworkx.PyGraph(), set())
        self.assertEqual({}, res)

    def test_bipartite_layout_hole(self):
        g = rustworkx.generators.path_graph(5)
        g.remove_nodes_from([1])
        res = rustworkx.bipartite_layout(g, set())
        expected = {
            0: (0.0, -1.0),
            2: (0.0, -0.3333333333333333),
            3: (0.0, 0.3333333333333333),
            4: (0.0, 1.0),
        }
        self.assertLayoutEquiv(expected, res)

    def test_bipartite_layout(self):
        res = rustworkx.bipartite_layout(self.graph, {0, 1, 2, 3, 4})
        expected = {
            0: (-1.0, -0.75),
            1: (-1.0, -0.375),
            2: (-1.0, 0.0),
            3: (-1.0, 0.375),
            4: (-1.0, 0.75),
            5: (1.0, -0.75),
            6: (1.0, -0.375),
            7: (1.0, 0.0),
            8: (1.0, 0.375),
            9: (1.0, 0.75),
        }
        self.assertLayoutEquiv(expected, res)

    def test_bipartite_layout_horizontal(self):
        res = rustworkx.bipartite_layout(self.graph, {0, 1, 2, 3}, horizontal=True)
        expected = {
            0: (1.0, -0.9),
            1: (0.3333333333333333, -0.9),
            2: (-0.333333333333333, -0.9),
            3: (-1.0, -0.9),
            4: (1.0, 0.6),
            5: (0.6, 0.6),
            6: (0.2, 0.6),
            7: (-0.2, 0.6),
            8: (-0.6, 0.6),
            9: (-1.0, 0.6),
        }
        self.assertLayoutEquiv(expected, res)

    def test_bipartite_layout_scale(self):
        res = rustworkx.bipartite_layout(self.graph, {0, 1, 2}, scale=2)
        expected = {
            0: (-2.0, -1.0714285714285714),
            1: (-2.0, 2.3790493384824785e-17),
            2: (-2.0, 1.0714285714285714),
            3: (0.8571428571428571, -1.0714285714285714),
            4: (0.8571428571428571, -0.7142857142857143),
            5: (0.8571428571428571, -0.35714285714285715),
            6: (0.8571428571428571, 2.3790493384824785e-17),
            7: (0.8571428571428571, 0.35714285714285704),
            8: (0.8571428571428571, 0.7142857142857141),
            9: (0.8571428571428571, 1.0714285714285714),
        }
        self.assertLayoutEquiv(expected, res)

    def test_bipartite_layout_center(self):
        res = rustworkx.bipartite_layout(self.graph, {4, 5, 6}, center=(0.5, 0.5))
        expected = {
            4: (-0.5, -0.0357142857142857),
            5: (-0.5, 0.5),
            6: (-0.5, 1.0357142857142856),
            0: (0.9285714285714286, -0.0357142857142857),
            1: (0.9285714285714286, 0.14285714285714285),
            2: (0.9285714285714286, 0.3214285714285714),
            3: (0.9285714285714286, 0.5),
            7: (0.9285714285714286, 0.6785714285714285),
            8: (0.9285714285714286, 0.857142857142857),
            9: (0.9285714285714286, 1.0357142857142856),
        }
        self.assertLayoutEquiv(expected, res)

    def test_bipartite_layout_ratio(self):
        res = rustworkx.bipartite_layout(self.graph, {2, 4, 8}, aspect_ratio=4)
        expected = {
            8: [-1.0, 0.17857142857142858],
            2: [-1.0, -0.17857142857142858],
            4: [-1.0, 0],
            0: [0.42857142857142855, -0.17857142857142858],
            1: [0.42857142857142855, -0.11904761904761907],
            3: [0.42857142857142855, -0.05952380952380952],
            5: [0.42857142857142855, 0],
            6: [0.42857142857142855, 0.05952380952380952],
            7: [0.42857142857142855, 0.11904761904761903],
            9: [0.42857142857142855, 0.17857142857142858],
        }
        self.assertLayoutEquiv(expected, res)


class TestCircularLayout(LayoutTest):
    def setUp(self):
        self.graph = rustworkx.generators.path_graph(10)

    def test_circular_layout_empty(self):
        res = rustworkx.circular_layout(rustworkx.PyGraph())
        self.assertEqual({}, res)

    def test_circular_layout_one_node(self):
        res = rustworkx.circular_layout(rustworkx.generators.path_graph(1))
        self.assertEqual({0: (0.0, 0.0)}, res)

    def test_circular_layout_hole(self):
        g = rustworkx.generators.path_graph(5)
        g.remove_nodes_from([1])
        res = rustworkx.circular_layout(g)
        expected = {
            0: (0.999999986090933, 2.1855693665697608e-08),
            2: (-3.576476059301554e-08, 1.0),
            3: (-0.9999999701976796, -6.556708099709282e-08),
            4: (1.987150711625619e-08, -0.9999999562886126),
        }
        self.assertLayoutEquiv(expected, res)

    def test_circular_layout(self):
        res = rustworkx.circular_layout(self.graph)
        expected = {
            0: (1.0, 2.662367085193061e-08),
            1: (0.8090170042900712, 0.5877852653564984),
            2: (0.3090169789580973, 0.9510565581329226),
            3: (-0.3090170206813483, 0.9510564985282783),
            4: (-0.8090170460133221, 0.5877852057518542),
            5: (-0.9999999821186069, -6.079910493992474e-08),
            6: (-0.8090169268040337, -0.5877853313184453),
            7: (-0.3090170802859925, -0.9510564452809367),
            8: (0.3090171279697079, -0.9510564452809367),
            9: (0.809016944685427, -0.587785271713801),
        }
        self.assertLayoutEquiv(expected, res)

    def test_circular_layout_scale(self):
        res = rustworkx.circular_layout(self.graph, scale=2)
        expected = {
            0: (2.0, 5.324734170386122e-08),
            1: (1.6180340085801423, 1.1755705307129969),
            2: (0.6180339579161946, 1.9021131162658451),
            3: (-0.6180340413626966, 1.9021129970565567),
            4: (-1.6180340920266443, 1.1755704115037084),
            5: (-1.9999999642372137, -1.2159820987984948e-07),
            6: (-1.6180338536080674, -1.1755706626368907),
            7: (-0.618034160571985, -1.9021128905618734),
            8: (0.6180342559394159, -1.9021128905618734),
            9: (1.618033889370854, -1.175570543427602),
        }
        self.assertLayoutEquiv(expected, res)

    def test_circular_layout_center(self):
        res = rustworkx.circular_layout(self.graph, center=(0.5, 0.5))
        expected = {
            0: (1.5, 0.5000000266236708),
            1: (1.3090170042900713, 1.0877852653564983),
            2: (0.8090169789580973, 1.4510565581329224),
            3: (0.1909829793186517, 1.4510564985282783),
            4: (-0.30901704601332214, 1.0877852057518542),
            5: (-0.49999998211860686, 0.4999999392008951),
            6: (-0.3090169268040337, -0.08778533131844535),
            7: (0.1909829197140075, -0.4510564452809367),
            8: (0.8090171279697079, -0.4510564452809367),
            9: (1.309016944685427, -0.08778527171380102),
        }
        self.assertLayoutEquiv(expected, res)


class TestShellLayout(LayoutTest):
    def setUp(self):
        self.graph = rustworkx.generators.path_graph(10)

    def test_shell_layout_empty(self):
        res = rustworkx.circular_layout(rustworkx.PyGraph())
        self.assertEqual({}, res)

    def test_shell_layout_one_node(self):
        res = rustworkx.shell_layout(rustworkx.generators.path_graph(1))
        self.assertEqual({0: (0.0, 0.0)}, res)

    def test_shell_layout_hole(self):
        g = rustworkx.generators.path_graph(5)
        g.remove_nodes_from([1])
        res = rustworkx.shell_layout(g)
        expected = {
            0: (-1.0, -8.742277657347586e-08),
            2: (1.1924880638503055e-08, -1.0),
            3: (1.0, 1.7484555314695172e-07),
            4: (-3.3776623808989825e-07, 1.0),
        }
        self.assertLayoutEquiv(expected, res)

    def test_shell_layout_hole_two_shells(self):
        g = rustworkx.generators.path_graph(5)
        g.remove_nodes_from([2])
        res = rustworkx.shell_layout(g, [[0, 1], [3, 4]])
        expected = {
            0: (-2.1855694143368964e-08, 0.5),
            1: (5.962440319251527e-09, -0.5),
            3: (-1.0, -8.742277657347586e-08),
            4: (1.0, 1.7484555314695172e-07),
        }
        self.assertLayoutEquiv(expected, res)

    def test_shell_layout(self):
        res = rustworkx.shell_layout(self.graph)
        expected = {
            0: (-1.0, -8.742277657347586e-08),
            1: (-0.8090169429779053, -0.5877853631973267),
            2: (-0.3090170919895172, -0.9510564804077148),
            3: (0.3090171217918396, -0.9510564804077148),
            4: (0.8090172410011292, -0.5877849459648132),
            5: (1.0, 1.7484555314695172e-07),
            6: (0.80901700258255, 0.5877852439880371),
            7: (0.30901679396629333, 0.9510565996170044),
            8: (-0.30901744961738586, 0.9510563611984253),
            9: (-0.8090168833732605, 0.5877854228019714),
        }
        self.assertLayoutEquiv(expected, res)

    def test_shell_layout_nlist(self):
        res = rustworkx.shell_layout(self.graph, nlist=[[0, 2], [1, 3], [4, 9], [8, 7], [6, 5]])
        expected = {
            0: (0.16180340945720673, 0.11755704879760742),
            2: (-0.16180339455604553, -0.11755707114934921),
            1: (0.12360679358243942, 0.3804226219654083),
            3: (-0.123606838285923, -0.38042259216308594),
            4: (-0.18541023135185242, 0.5706338882446289),
            9: (0.185410276055336, -0.5706338882446289),
            8: (-0.6472136378288269, 0.4702281653881073),
            7: (0.6472138166427612, -0.4702279567718506),
            6: (-1.0, -8.742277657347586e-08),
            5: (1.0, 1.7484555314695172e-07),
        }
        self.assertLayoutEquiv(expected, res)

    def test_shell_layout_rotate(self):
        res = rustworkx.shell_layout(
            self.graph, nlist=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]], rotate=0.5
        )
        expected = {
            0: (0.21939563751220703, 0.11985638737678528),
            1: (-0.21349650621414185, 0.13007399439811707),
            2: (-0.005899117328226566, -0.24993039667606354),
            3: (0.27015113830566406, 0.4207354784011841),
            4: (-0.4994432032108307, 0.023589985445141792),
            5: (0.229292094707489, -0.4443254768848419),
            6: (0.05305289849638939, 0.7481212615966797),
            7: (-0.6744184494018555, -0.3281154930591583),
            8: (0.6213656067848206, -0.420005738735199),
            9: (-0.416146844625473, 0.9092974066734314),
        }
        self.assertLayoutEquiv(expected, res)

    def test_shell_layout_scale(self):
        res = rustworkx.shell_layout(self.graph, nlist=[[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]], scale=2)
        expected = {
            0: (-4.371138828673793e-08, 1.0),
            1: (-0.9510565996170044, 0.30901679396629333),
            2: (-0.5877850651741028, -0.8090171217918396),
            3: (0.5877854824066162, -0.8090168237686157),
            4: (0.9510564208030701, 0.30901727080345154),
            9: (-2.0, -1.7484555314695172e-07),
            8: (-0.6180341839790344, -1.9021129608154297),
            7: (1.6180344820022583, -1.1755698919296265),
            6: (1.6180340051651, 1.1755704879760742),
            5: (-0.6180348992347717, 1.9021127223968506),
        }
        self.assertLayoutEquiv(expected, res)

    def test_shell_layout_center(self):
        res = rustworkx.shell_layout(
            self.graph,
            nlist=[[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]],
            center=(0.5, 0.5),
        )
        expected = {
            0: (0.49999997814430586, 1.0),
            1: (0.024471700191497803, 0.6545083969831467),
            2: (0.2061074674129486, 0.0954914391040802),
            3: (0.7938927412033081, 0.09549158811569214),
            4: (0.975528210401535, 0.6545086354017258),
            9: (-0.5, 0.4999999125772234),
            8: (0.1909829080104828, -0.45105648040771484),
            7: (1.3090172410011292, -0.08778494596481323),
            6: (1.30901700258255, 1.087785243988037),
            5: (0.19098255038261414, 1.4510563611984253),
        }
        self.assertLayoutEquiv(expected, res)


class TestSpiralLayout(LayoutTest):
    def setUp(self):
        self.graph = rustworkx.generators.path_graph(10)

    def test_spiral_layout_empty(self):
        res = rustworkx.spiral_layout(rustworkx.PyGraph())
        self.assertEqual({}, res)

    def test_spiral_layout_one_node(self):
        res = rustworkx.spiral_layout(rustworkx.generators.path_graph(1))
        self.assertEqual({0: (0.0, 0.0)}, res)

    def test_spiral_layout_hole(self):
        g = rustworkx.generators.path_graph(5)
        g.remove_nodes_from([1])
        res = rustworkx.spiral_layout(g)
        expected = {
            0: (-0.6415327868391166, -0.6855508729419231),
            2: (-0.03307913182988828, -0.463447951079834),
            3: (0.34927952438480797, 0.1489988240217569),
            4: (0.32533239428419697, 1.0),
        }
        self.assertLayoutEquiv(expected, res)

    def test_spiral_layout(self):
        res = rustworkx.spiral_layout(self.graph)
        expected = {
            0: (0.3083011152777303, -0.36841870322845377),
            1: (0.4448595378922136, -0.3185709877650719),
            2: (0.5306742824266687, -0.18111636841212878),
            3: (0.5252997033017661, 0.009878257518578544),
            4: (0.40713492048969163, 0.20460820654918466),
            5: (0.17874125121181098, 0.3468009691240852),
            6: (-0.1320415949011884, 0.3844997574641717),
            7: (-0.4754889029311045, 0.28057288841663486),
            8: (-0.7874803127675889, 0.021164283410983312),
            9: (-0.9999999999999999, -0.3794183030779839),
        }
        self.assertLayoutEquiv(expected, res)

    def test_spiral_layout_scale(self):
        res = rustworkx.spiral_layout(self.graph, scale=2)
        expected = {
            0: (0.6166022305554606, -0.7368374064569075),
            1: (0.8897190757844272, -0.6371419755301438),
            2: (1.0613485648533374, -0.36223273682425755),
            3: (1.0505994066035322, 0.01975651503715709),
            4: (0.8142698409793833, 0.4092164130983693),
            5: (0.35748250242362195, 0.6936019382481704),
            6: (-0.2640831898023768, 0.7689995149283434),
            7: (-0.950977805862209, 0.5611457768332697),
            8: (-1.5749606255351778, 0.042328566821966625),
            9: (-1.9999999999999998, -0.7588366061559678),
        }
        self.assertLayoutEquiv(expected, res)

    def test_spiral_layout_center(self):
        res = rustworkx.spiral_layout(self.graph, center=(1, 1))
        expected = {
            0: (1.3083011152777302, 0.6315812967715462),
            1: (1.4448595378922136, 0.681429012234928),
            2: (1.5306742824266686, 0.8188836315878713),
            3: (1.5252997033017661, 1.0098782575185785),
            4: (1.4071349204896917, 1.2046082065491848),
            5: (1.178741251211811, 1.3468009691240852),
            6: (0.8679584050988116, 1.3844997574641718),
            7: (0.5245110970688955, 1.2805728884166347),
            8: (0.2125196872324111, 1.0211642834109833),
            9: (1.1102230246251565e-16, 0.6205816969220161),
        }
        self.assertLayoutEquiv(expected, res)

    def test_spiral_layout_resolution(self):
        res = rustworkx.spiral_layout(self.graph, resolution=0.6)
        expected = {
            0: (0.14170895375949058, 0.22421978768273812),
            1: (0.2657196183870804, 0.30906004798138936),
            2: (0.2506009612140119, 0.5043065412934762),
            3: (0.039294315670400995, 0.6631957258449066),
            4: (-0.3014789232909145, 0.6301862160709318),
            5: (-0.602046830323471, 0.3302396035952633),
            6: (-0.66674476042188, -0.17472522299849289),
            7: (-0.3739394496041176, -0.6924895145748617),
            8: (0.2468861146093996, -0.9732085843739783),
            9: (1.0, -0.8207846005213728),
        }
        self.assertLayoutEquiv(expected, res)

    def test_spiral_layout_equidistant(self):
        res = rustworkx.spiral_layout(self.graph, equidistant=True)
        expected = {
            0: (-0.13161882865656718, -0.7449342807652114),
            1: (0.7160560542246066, -0.6335352483233974),
            2: (0.6864868274284994, -0.34165899654603915),
            3: (0.5679822628330004, -0.07281296883784087),
            4: (0.375237081214659, 0.14941210155952697),
            5: (0.12730720268992277, 0.30830226777240866),
            6: (-0.15470865936858091, 0.3939608192236113),
            7: (-0.4495426197217269, 0.4027809258196645),
            8: (-0.7371993206438128, 0.33662707199446507),
            9: (-1.0, 0.2018583081028111),
        }
        self.assertLayoutEquiv(expected, res)
