# Licensed under the Apache License, 

import unittest

import rustworkx


class TestLadderGraph(unittest.TestCase):
    def test_ladder_graph(self):
        graph = rustworkx.generators.ladder_graph(20)
        self.assertEqual(len(graph), 40)
        self.assertEqual(len(graph.edges()), 58)
    
    def test_ladder_graph_weights(self):
        graph = rustworkx.generators.ladder_graph(weights=list(range(40)))
        self.assertEqual(len(graph), 40)
        self.assertEqual([x for x in range(40)], graph.nodes())
        self.assertEqual(len(graph.edges()), 58)

    def test_ladder_no_weights_or_num(self):
        with self.assertRaises(IndexError):
            rustworkx.generators.ladder_graph()

    def test_zero_length_ladder_graph(self):
        graph = rustworkx.generators.ladder_graph(0)
        self.assertEqual(0, len(graph))