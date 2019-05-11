import unittest

import retworkx


class TestLongestPath(unittest.TestCase):
    def test_linear(self):
        """Longest depth for a simple dag.

        a
        |
        b
        |\
        c d
        |
        e
        |\
        f g
        """
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        node_c = dag.add_child(node_b, 'c', {})
        dag.add_child(node_b, 'd', {})
        node_e = dag.add_child(node_c, 'e', {})
        dag.add_child(node_e, 'f', {})
        dag.add_child(node_e, 'g', {})
        self.assertEqual(4, retworkx.dag_longest_path_length(dag))

    def test_less_linear(self):
        dag = retworkx.PyDAG()
        node_a = dag.add_node('a')
        node_b = dag.add_child(node_a, 'b', {})
        node_c = dag.add_child(node_b, 'c', {})
        node_d = dag.add_child(node_c, 'd', {})
        node_e = dag.add_child(node_d, 'e', {})
        dag.add_edge(node_a, node_c, {})
        dag.add_edge(node_a, node_e, {})
        dag.add_edge(node_c, node_e, {})
        self.assertEqual(5, retworkx.dag_longest_path_length(dag))

    def test_degenerate_graph(self):
        dag = retworkx.PyDAG()
        dag.add_node(0)
        self.assertEqual(1, retworkx.dag_longest_path_length(dag))
