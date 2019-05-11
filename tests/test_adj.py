import unittest

import retworkx


class TestAdj(unittest.TestCase):
    def test_single_neighbor(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_a, 'c', {'a': 2})
        res = dag.adj(node_a)
        self.assertEqual({node_b: {'a': 1}, node_c: {'a': 2}}, res)

    def test_single_neighbor_dir(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {'a': 1})
        node_c = dag.add_child(node_a, 'c', {'a': 2})
        res = dag.adj_direction(node_a, False)
        self.assertEqual({node_b: {'a': 1}, node_c: {'a': 2}}, res)
        res = dag.adj_direction(node_a, True)
        self.assertEqual({}, res)
