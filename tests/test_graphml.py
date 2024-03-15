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
import tempfile
import numpy

import rustworkx


class TestGraphML(unittest.TestCase):
    HEADER = """
        <?xml version="1.0" encoding="UTF-8"?>
        <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
                http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
        {}
        </graphml>
    """

    def assertDictPayloadEqual(self, xs, ys):
        self.assertEqual(len(xs), len(ys))
        for key, va in xs.items():
            vb = ys.get(key, None)
            self.assertTrue(
                (isinstance(va, float) and isinstance(vb, float) and numpy.allclose(va, vb))
                or (va == vb)
            )

    def assertGraphEqual(self, graph, nodes, edges, directed=True, attrs={}):
        self.assertTrue(isinstance(graph, rustworkx.PyDiGraph if directed else rustworkx.PyGraph))
        self.assertEqual(len(graph), len(nodes))
        self.assertEqual(graph.attrs, attrs)
        for node_a, node_b in zip(graph.nodes(), nodes):
            self.assertDictPayloadEqual(node_a, node_b)

        for ((s, t, data), edge) in zip(graph.weighted_edge_list(), edges):
            self.assertEqual((graph[s]["id"], graph[t]["id"]), (edge[0], edge[1]))
            self.assertDictPayloadEqual(data, edge[2])

    def assertGraphMLRaises(self, graph_xml):
        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            with self.assertRaises(Exception):
                rustworkx.read_graphml(fd.name)

    def test_simple(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="color" attr.type="string">
            <default>yellow</default>
            </key>
            <key id="d1" for="edge" attr.name="fidelity" attr.type="float">
            <default>0.95</default>
            </key>
            <graph id="G" edgedefault="undirected">
            <node id="n0">
                <data key="d0">blue</data>
            </node>
            <node id="n1"/>
            <node id="n2">
                <data key="d0">green</data>
            </node>
            <edge source="n0" target="n1">
                <data key="d1">0.98</data>
            </edge>
            <edge source="n0" target="n2"/>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "color": "blue"},
                {"id": "n1", "color": "yellow"},
                {"id": "n2", "color": "green"},
            ]
            edges = [
                ("n0", "n1", {"fidelity": 0.98}),
                ("n0", "n2", {"fidelity": 0.95}),
            ]
            self.assertGraphEqual(graph, nodes, edges, directed=False)

    def test_multiple_graphs_in_single_file(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="color" attr.type="string">
            <default>yellow</default>
            </key>
            <key id="d1" for="edge" attr.name="fidelity" attr.type="float">
            <default>0.95</default>
            </key>
            <graph id="G" edgedefault="undirected">
            <node id="n0">
                <data key="d0">blue</data>
            </node>
            <node id="n1"/>
            <edge id="e01" source="n0" target="n1">
                <data key="d1">0.98</data>
            </edge>
            </graph>
            <graph id="H" edgedefault="directed">
            <node id="n0">
                <data key="d0">red</data>
            </node>
            <node id="n1"/>
            <edge id="e01" source="n0" target="n1"/>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            self.assertEqual(len(graphml), 2)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "color": "blue"},
                {"id": "n1", "color": "yellow"},
            ]
            edges = [
                ("n0", "n1", {"id": "e01", "fidelity": 0.98}),
            ]
            self.assertGraphEqual(graph, nodes, edges, directed=False)
            graph = graphml[1]
            nodes = [
                {"id": "n0", "color": "red"},
                {"id": "n1", "color": "yellow"},
            ]
            edges = [
                ("n0", "n1", {"id": "e01", "fidelity": 0.95}),
            ]
            self.assertGraphEqual(graph, nodes, edges, directed=True)

    def test_key_for_graph(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="graph" attr.name="test" attr.type="boolean"/>
            <graph id="G" edgedefault="directed">
            <data key="d0">true</data>
            <node id="n0"/>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [{"id": "n0"}]
            edges = []
            self.assertGraphEqual(graph, nodes, edges, directed=True, attrs={"test": True})

    def test_key_for_all(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="all" attr.name="test" attr.type="string"/>
            <graph id="G" edgedefault="directed">
            <data key="d0">I'm a graph.</data>
            <node id="n0">
                <data key="d0">I'm a node.</data>
            </node>
            <node id="n1">
                <data key="d0">I'm a node.</data>
            </node>
            <edge source="n0" target="n1">
                <data key="d0">I'm an edge.</data>
            </edge>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "test": "I'm a node."},
                {"id": "n1", "test": "I'm a node."},
            ]
            edges = [("n0", "n1", {"test": "I'm an edge."})]
            self.assertGraphEqual(
                graph, nodes, edges, directed=True, attrs={"test": "I'm a graph."}
            )

    def test_key_default_undefined(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="boolean"/>
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <data key="d0">true</data>
            </node>
            <node id="n1"/>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "test": True},
                {"id": "n1", "test": None},
            ]
            edges = []
            self.assertGraphEqual(graph, nodes, edges, directed=True)

    def test_bool(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="boolean">
            <default>false</default>
            </key>
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <data key="d0">true</data>
            </node>
            <node id="n1"/>
            <node id="n2">
                <data key="d0">false</data>
            </node>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "test": True},
                {"id": "n1", "test": False},
                {"id": "n2", "test": False},
            ]
            edges = []
            self.assertGraphEqual(graph, nodes, edges, directed=True)

    def test_int(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="int">
            <default>42</default>
            </key>
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <data key="d0">8</data>
            </node>
            <node id="n1"/>
            <node id="n2">
                <data key="d0">42</data>
            </node>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "test": 8},
                {"id": "n1", "test": 42},
                {"id": "n2", "test": 42},
            ]
            edges = []
            self.assertGraphEqual(graph, nodes, edges, directed=True)

    def test_float(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="float">
            <default>4.2</default>
            </key>
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <data key="d0">1.8</data>
            </node>
            <node id="n1"/>
            <node id="n2">
                <data key="d0">4.2</data>
            </node>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "test": 1.8},
                {"id": "n1", "test": 4.2},
                {"id": "n2", "test": 4.2},
            ]
            edges = []
            self.assertGraphEqual(graph, nodes, edges, directed=True)

    def test_double(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="double">
            <default>4.2</default>
            </key>
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <data key="d0">1.8</data>
            </node>
            <node id="n1"/>
            <node id="n2">
                <data key="d0">4.2</data>
            </node>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "test": 1.8},
                {"id": "n1", "test": 4.2},
                {"id": "n2", "test": 4.2},
            ]
            edges = []
            self.assertGraphEqual(graph, nodes, edges, directed=True)

    def test_string(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="string">
            <default>yellow</default>
            </key>
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <data key="d0">blue</data>
            </node>
            <node id="n1"/>
            <node id="n2">
                <data key="d0">yellow</data>
            </node>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "test": "blue"},
                {"id": "n1", "test": "yellow"},
                {"id": "n2", "test": "yellow"},
            ]
            edges = []
            self.assertGraphEqual(graph, nodes, edges, directed=True)

    def test_long(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="long">
            <default>42</default>
            </key>
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <data key="d0">8</data>
            </node>
            <node id="n1"/>
            <node id="n2">
                <data key="d0">42</data>
            </node>
            </graph>
            """
        )

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(graph_xml)
            fd.flush()
            graphml = rustworkx.read_graphml(fd.name)
            graph = graphml[0]
            nodes = [
                {"id": "n0", "test": 8},
                {"id": "n1", "test": 42},
                {"id": "n2", "test": 42},
            ]
            edges = []
            self.assertGraphEqual(graph, nodes, edges, directed=True)

    def test_convert_error(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="{}">
            <default>blah</default>
            </key>
            """
        )

        for type in ["boolean", "int", "float", "double", "long"]:
            self.assertGraphMLRaises(graph_xml=graph_xml.format(type))

    def test_invalid_xml(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="string">
            </default>
            </key>
            """
        )
        self.assertGraphMLRaises(graph_xml)

    def test_invalid_edgedefault(self):
        graph_xml = self.HEADER.format(
            """
            <graph id="G" edgedefault="UnDir">
            <node id="n0"/>
            </graph>
            """
        )
        self.assertGraphMLRaises(graph_xml)

    def test_missing_node_id(self):
        graph_xml = self.HEADER.format(
            """
            <graph id="G" edgedefault="directed">
            <node/>
            </graph>
            """
        )
        self.assertGraphMLRaises(graph_xml)

    def test_missing_key_for_node(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="edge" attr.name="color" attr.type="string"/>
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <data key="d0">blue</data>
            </node>
            </graph>
            """
        )
        self.assertGraphMLRaises(graph_xml)

    def test_invalid_key_type(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="node" attr.name="test" attr.type="List[int]"/>
            """
        )
        self.assertGraphMLRaises(graph_xml)

    def test_unsupported_key_domain(self):
        graph_xml = self.HEADER.format(
            """
            <key id="d0" for="bad" attr.name="test" attr.type="int"/>
            """
        )
        self.assertGraphMLRaises(graph_xml)

    def test_unsupported_nested_graphs(self):
        graph_xml = self.HEADER.format(
            """
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <graph id="n0:" edgedefault="undirected">
                    <node id="n0::n0"/>
                    <node id="n0::n1"/>
                    <edge id="en" source="n0::n0" target="n0::n1"/>
                </graph>
            </node>
            </graph>
            """
        )
        self.assertGraphMLRaises(graph_xml)

    def test_unsupported_hyperedges(self):
        graph_xml = self.HEADER.format(
            """
            <graph id="G" edgedefault="directed">
            <node id="n0"/>
            <node id="n1"/>
            <node id="n2"/>
            <hyperedge>
                <endpoint node="n0"/>
                <endpoint node="n1"/>
                <endpoint node="n2"/>
            </hyperedge>
            </graph>
            """
        )
        self.assertGraphMLRaises(graph_xml)

    def test_unsupported_ports(self):
        graph_xml = self.HEADER.format(
            """
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <port name="North"/>
            </node>
            </graph>
            """
        )
        self.assertGraphMLRaises(graph_xml)

    def test_unsupported_nested_ports(self):
        graph_xml = self.HEADER.format(
            """
            <graph id="G" edgedefault="directed">
            <node id="n0">
                <port name="North">
                    <port name="Snow"/>
                </port>
            </node>
            </graph>
            """
        )
        self.assertGraphMLRaises(graph_xml)
