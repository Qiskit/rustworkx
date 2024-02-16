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


class TestNodeLinkJSON(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.node_link_json(graph)
        expected = {"attrs": None, "directed": True, "links": [], "multigraph": True, "nodes": []}
        self.assertEqual(json.loads(res), expected)

    def test_directed_path_graph(self):
        graph = rustworkx.generators.directed_path_graph(3)
        res = rustworkx.node_link_json(graph)
        expected = {
            "attrs": None,
            "directed": True,
            "links": [
                {"data": None, "id": 0, "source": 0, "target": 1},
                {"data": None, "id": 1, "source": 1, "target": 2},
            ],
            "multigraph": True,
            "nodes": [{"data": None, "id": 0}, {"data": None, "id": 1}, {"data": None, "id": 2}],
        }
        self.assertEqual(json.loads(res), expected)

    def test_directed_path_graph_node_attrs(self):
        graph = rustworkx.generators.directed_path_graph(3)
        for node in graph.node_indices():
            graph[node] = {"nodeLabel": f"node={node}"}
        res = rustworkx.node_link_json(graph, node_attrs=dict)
        expected = {
            "attrs": None,
            "directed": True,
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

    def test_directed_path_graph_edge_attr(self):
        graph = rustworkx.generators.directed_path_graph(3)
        for edge, (source, target, _weight) in graph.edge_index_map().items():
            graph.update_edge_by_index(edge, {"edgeLabel": f"{source}->{target}"})

        res = rustworkx.node_link_json(graph, edge_attrs=dict)
        expected = {
            "attrs": None,
            "directed": True,
            "links": [
                {"data": {"edgeLabel": "0->1"}, "id": 0, "source": 0, "target": 1},
                {"data": {"edgeLabel": "1->2"}, "id": 1, "source": 1, "target": 2},
            ],
            "multigraph": True,
            "nodes": [{"data": None, "id": 0}, {"data": None, "id": 1}, {"data": None, "id": 2}],
        }
        self.assertEqual(json.loads(res), expected)

    def test_directed_path_graph_attr(self):
        graph = rustworkx.PyDiGraph(attrs="label")
        res = rustworkx.node_link_json(graph, graph_attrs=lambda x: {"label": x})
        expected = {
            "attrs": {"label": "label"},
            "directed": True,
            "links": [],
            "multigraph": True,
            "nodes": [],
        }
        self.assertEqual(json.loads(res), expected)

    def test_file_output(self):
        graph = rustworkx.generators.directed_path_graph(3)
        graph.attrs = "directed_path_graph"
        for node in graph.node_indices():
            graph[node] = {"nodeLabel": f"node={node}"}
        for edge, (source, target, _weight) in graph.edge_index_map().items():
            graph.update_edge_by_index(edge, {"edgeLabel": f"{source}->{target}"})
        expected = {
            "attrs": {"label": "directed_path_graph"},
            "directed": True,
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
        nonexistent_path = tempfile.gettempdir() + "/" + str(uuid.uuid4()) + "/graph.rustworkx.json"
        graph = rustworkx.PyDiGraph()
        with self.assertRaises(FileNotFoundError):
            rustworkx.node_link_json(graph, path=nonexistent_path)

    def test_attr_callback_invalid_type(self):
        graph = rustworkx.PyDiGraph()
        with self.assertRaises(TypeError):
            rustworkx.node_link_json(graph, graph_attrs=lambda _: "attrs_field")

    def test_not_multigraph(self):
        graph = rustworkx.PyDiGraph(multigraph=False)
        res = rustworkx.node_link_json(graph)
        expected = {"attrs": None, "directed": True, "links": [], "multigraph": False, "nodes": []}
        self.assertEqual(json.loads(res), expected)

    def test_round_trip(self):
        graph = rustworkx.generators.directed_path_graph(123)
        res = rustworkx.node_link_json(graph)
        new = rustworkx.parse_node_link_json_str(res)
        self.assertIsInstance(new, type(graph))
        self.assertEqual(new.nodes(), graph.nodes())
        self.assertEquals(new.weighted_edge_list(), graph.weighted_edge_list())
        self.assertEqual(new.attrs, graph.attrs)

    def test_round_trip_file(self):
        graph = rustworkx.generators.directed_heavy_hex_graph(19)
        graph.attrs = "directed_heavy_hex_graph"
        for node in graph.node_indices():
            graph[node] = {"nodeLabel": f"node={node}"}
        for edge, (source, target, _weight) in graph.edge_index_map().items():
            graph.update_edge_by_index(edge, {"edgeLabel": f"{source}-          >{target}"})
        with tempfile.NamedTemporaryFile() as fd:
            res = rustworkx.node_link_json(
                graph,
                path=fd.name,
                graph_attrs=lambda x: {"label": x},
                node_attrs=dict,
                edge_attrs=dict,
            )
            new = rustworkx.parse_node_link_json_file(fd.name, graph_attrs=lambda x: x["label"])
        self.assertIsInstance(new, type(graph))
        self.assertEqual(new.nodes(), graph.nodes())
        self.assertEquals(new.weighted_edge_list(), graph.weighted_edge_list())
        self.assertEqual(new.attrs, graph.attrs)

    def test_round_trip_file_no_callback(self):
        graph = rustworkx.generators.directed_heavy_hex_graph(19)
        graph.attrs = "directed_heavy_hex_graph"
        for node in graph.node_indices():
            graph[node] = {"nodeLabel": f"node={node}"}
        for edge, (source, target, _weight) in graph.edge_index_map().items():
            graph.update_edge_by_index(edge, {"edgeLabel": f"{source}-          >{target}"})
        with tempfile.NamedTemporaryFile() as fd:
            res = rustworkx.node_link_json(
                graph,
                path=fd.name,
                graph_attrs=lambda x: {"label": x},
                node_attrs=dict,
                edge_attrs=dict,
            )
            new = rustworkx.parse_node_link_json_file(fd.name)
        self.assertIsInstance(new, type(graph))
        self.assertEqual(new.nodes(), graph.nodes())
        self.assertEquals(new.weighted_edge_list(), graph.weighted_edge_list())
        self.assertEqual(new.attrs, {"label": graph.attrs})
