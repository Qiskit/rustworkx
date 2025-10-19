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

import json
import tempfile
import uuid

import unittest
import rustworkx
import networkx as nx


class TestNodeLinkJSON(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.node_link_json(graph)
        expected = {"attrs": None, "directed": False, "links": [], "multigraph": True, "nodes": []}
        self.assertEqual(json.loads(res), expected)

    def test_path_graph(self):
        graph = rustworkx.generators.path_graph(3)
        res = rustworkx.node_link_json(graph)
        expected = {
            "attrs": None,
            "directed": False,
            "links": [
                {"data": None, "id": 0, "source": 0, "target": 1},
                {"data": None, "id": 1, "source": 1, "target": 2},
            ],
            "multigraph": True,
            "nodes": [{"data": None, "id": 0}, {"data": None, "id": 1}, {"data": None, "id": 2}],
        }
        self.assertEqual(json.loads(res), expected)

    def test_path_graph_node_attrs(self):
        graph = rustworkx.generators.path_graph(3)
        for node in graph.node_indices():
            graph[node] = {"nodeLabel": f"node={node}"}
        res = rustworkx.node_link_json(graph, node_attrs=dict)
        expected = {
            "attrs": None,
            "directed": False,
            "links": [
                {"data": None, "id": 0, "source": 0, "target": 1},
                {"data": None, "id": 1, "source": 1, "target": 2},
            ],
            "multigraph": True,
            "nodes": [
                {"data": {"nodeLabel": "node=0"}, "id": 0},
                {"data": {"nodeLabel": "node=1"}, "id": 1},
                {"data": {"nodeLabel": "node=2"}, "id": 2},
            ],
        }
        self.assertEqual(json.loads(res), expected)

    def test_path_graph_edge_attr(self):
        graph = rustworkx.generators.path_graph(3)
        for edge, (source, target, _weight) in graph.edge_index_map().items():
            graph.update_edge_by_index(edge, {"edgeLabel": f"{source}->{target}"})

        res = rustworkx.node_link_json(graph, edge_attrs=dict)
        expected = {
            "attrs": None,
            "directed": False,
            "links": [
                {"data": {"edgeLabel": "0->1"}, "id": 0, "source": 0, "target": 1},
                {"data": {"edgeLabel": "1->2"}, "id": 1, "source": 1, "target": 2},
            ],
            "multigraph": True,
            "nodes": [{"data": None, "id": 0}, {"data": None, "id": 1}, {"data": None, "id": 2}],
        }
        self.assertEqual(json.loads(res), expected)

    def test_path_graph_attr(self):
        graph = rustworkx.PyGraph(attrs="label")
        res = rustworkx.node_link_json(graph, graph_attrs=lambda x: {"label": x})
        expected = {
            "attrs": {"label": "label"},
            "directed": False,
            "links": [],
            "multigraph": True,
            "nodes": [],
        }
        self.assertEqual(json.loads(res), expected)

    def test_file_output(self):
        graph = rustworkx.generators.path_graph(3)
        graph.attrs = "path_graph"
        for node in graph.node_indices():
            graph[node] = {"nodeLabel": f"node={node}"}
        for edge, (source, target, _weight) in graph.edge_index_map().items():
            graph.update_edge_by_index(edge, {"edgeLabel": f"{source}->{target}"})
        expected = {
            "attrs": {"label": "path_graph"},
            "directed": False,
            "links": [
                {"data": {"edgeLabel": "0->1"}, "id": 0, "source": 0, "target": 1},
                {"data": {"edgeLabel": "1->2"}, "id": 1, "source": 1, "target": 2},
            ],
            "multigraph": True,
            "nodes": [
                {"data": {"nodeLabel": "node=0"}, "id": 0},
                {"data": {"nodeLabel": "node=1"}, "id": 1},
                {"data": {"nodeLabel": "node=2"}, "id": 2},
            ],
        }
        with tempfile.NamedTemporaryFile() as fd:
            res = rustworkx.node_link_json(
                graph,
                path=fd.name,
                graph_attrs=lambda x: {"label": x},
                node_attrs=dict,
                edge_attrs=dict,
            )
            self.assertIsNone(res)
            json_dict = json.load(fd)
            self.assertEqual(json_dict, expected)

    def test_invalid_path_dir(self):
        nonexistent_path = f"{tempfile.gettempdir()}/{uuid.uuid4()}/graph.rustworkx.json"
        graph = rustworkx.PyGraph()
        with self.assertRaises(FileNotFoundError):
            rustworkx.node_link_json(graph, path=nonexistent_path)

    def test_attr_callback_invalid_type(self):
        graph = rustworkx.PyGraph()
        with self.assertRaises(TypeError):
            rustworkx.node_link_json(graph, graph_attrs=lambda _: "attrs_field")

    def test_not_multigraph(self):
        graph = rustworkx.PyGraph(multigraph=False)
        res = rustworkx.node_link_json(graph)
        expected = {"attrs": None, "directed": False, "links": [], "multigraph": False, "nodes": []}
        self.assertEqual(json.loads(res), expected)

    def test_round_trip(self):
        graph = rustworkx.generators.path_graph(3)
        res = rustworkx.node_link_json(graph)
        new = rustworkx.parse_node_link_json(res)
        self.assertIsInstance(new, type(graph))
        self.assertEqual(new.nodes(), graph.nodes())
        self.assertEqual(new.weighted_edge_list(), graph.weighted_edge_list())
        self.assertEqual(new.attrs, graph.attrs)

    def test_round_trip_with_file(self):
        graph = rustworkx.generators.path_graph(3)
        graph.attrs = "path_graph"
        for node in graph.node_indices():
            graph[node] = {"nodeLabel": f"node={node}"}
        for edge, (source, target, _weight) in graph.edge_index_map().items():
            graph.update_edge_by_index(edge, {"edgeLabel": f"{source}->{target}"})
        with tempfile.NamedTemporaryFile() as fd:
            rustworkx.node_link_json(
                graph,
                path=fd.name,
                graph_attrs=lambda x: {"label": x},
                node_attrs=dict,
                edge_attrs=dict,
            )
            new = rustworkx.from_node_link_json_file(fd.name, graph_attrs=lambda x: x["label"])
        self.assertIsInstance(new, type(graph))
        self.assertEqual(new.nodes(), graph.nodes())
        self.assertEqual(new.weighted_edge_list(), graph.weighted_edge_list())
        self.assertEqual(new.attrs, graph.attrs)

    def test_round_trip_networkx(self):
        graph = nx.generators.path_graph(5)
        try:
            node_link_str = json.dumps(nx.node_link_data(graph, edges="links"))
        except TypeError:
            # TODO: Remove this once we no longer need to support Python 3.9
            node_link_str = json.dumps(nx.node_link_data(graph))
        new = rustworkx.parse_node_link_json(node_link_str)
        self.assertIsInstance(new, rustworkx.PyGraph)
        self.assertEqual(new.num_nodes(), graph.number_of_nodes())
        self.assertEqual(new.edge_list(), list(graph.edges()))

    def test_round_trip_with_file_no_graph_attr(self):
        graph = rustworkx.generators.path_graph(3)
        graph.attrs = "path_graph"
        for node in graph.node_indices():
            graph[node] = {"nodeLabel": f"node={node}"}
        for edge, (source, target, _weight) in graph.edge_index_map().items():
            graph.update_edge_by_index(edge, {"edgeLabel": f"{source}->{target}"})
        with tempfile.NamedTemporaryFile() as fd:
            rustworkx.node_link_json(
                graph,
                path=fd.name,
                graph_attrs=lambda x: {"label": x},
                node_attrs=dict,
                edge_attrs=dict,
            )
            new = rustworkx.from_node_link_json_file(fd.name)
        self.assertIsInstance(new, type(graph))
        self.assertEqual(new.nodes(), graph.nodes())
        self.assertEqual(new.weighted_edge_list(), graph.weighted_edge_list())
        self.assertEqual(new.attrs, {"label": graph.attrs})

    def test_node_indices_preserved_with_deletion(self):
        """Test that node indices are preserved after deletion (related to issue #1503)"""
        graph = rustworkx.PyGraph()
        graph.add_node(None)  # 0
        graph.add_node(None)  # 1
        graph.add_node(None)  # 2
        graph.add_edge(0, 2, None)
        graph.remove_node(1)  # Remove middle node

        # Verify original has gaps in indices
        self.assertEqual([0, 2], graph.node_indices())

        # Round-trip through JSON
        json_str = rustworkx.node_link_json(graph)
        restored = rustworkx.parse_node_link_json(json_str)

        # Verify indices are preserved
        self.assertEqual(graph.node_indices(), restored.node_indices())
        self.assertEqual(graph.edge_list(), restored.edge_list())

    def test_node_indices_preserved_with_contraction(self):
        """Test that node indices are preserved after contraction (issue #1503)"""
        graph = rustworkx.PyGraph()
        graph.add_node(None)  # 0
        graph.add_node(None)  # 1
        graph.add_node(None)  # 2

        # Contract nodes 0 and 1
        contracted_idx = graph.contract_nodes([0, 1], None)
        graph.add_edge(2, contracted_idx, None)

        # Verify original has non-consecutive indices
        self.assertEqual([2, contracted_idx], graph.node_indices())

        # Round-trip through JSON
        json_str = rustworkx.node_link_json(graph)
        restored = rustworkx.parse_node_link_json(json_str)

        # Verify indices are preserved
        self.assertEqual(graph.node_indices(), restored.node_indices())
        self.assertEqual(graph.edge_list(), restored.edge_list())

    def test_node_indices_preserved_complex(self):
        """Test index preservation with multiple deletions and edges"""
        graph = rustworkx.PyGraph()
        for i in range(6):
            graph.add_node(None)

        graph.add_edge(0, 1, None)
        graph.add_edge(2, 3, None)
        graph.add_edge(4, 5, None)

        # Remove nodes 1 and 4
        graph.remove_node(1)
        graph.remove_node(4)

        # Verify gaps exist
        self.assertEqual([0, 2, 3, 5], graph.node_indices())

        # Round-trip through JSON
        json_str = rustworkx.node_link_json(graph)
        restored = rustworkx.parse_node_link_json(json_str)

        # Verify complete state is preserved
        self.assertEqual(graph.node_indices(), restored.node_indices())
        self.assertEqual(graph.edge_list(), restored.edge_list())
