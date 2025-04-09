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

import rustworkx as rx


class TestKarate(unittest.TestCase):
    def test_isomorphic_to_networkx(self):
        def node_matcher(a, b):
            if isinstance(a, dict):
                (
                    a,
                    b,
                ) = (
                    b,
                    a,
                )
            return a == b["club"]

        def edge_matcher(a, b):
            if isinstance(a, dict):
                (
                    a,
                    b,
                ) = (
                    b,
                    a,
                )
            return a == b["weight"]

        with tempfile.NamedTemporaryFile("wt") as fd:
            fd.write(karate_xml)
            fd.flush()
            expected = rx.read_graphml(fd.name)[0]

        graph = rx.generators.karate_club_graph()

        self.assertTrue(
            rx.is_isomorphic(graph, expected, node_matcher=node_matcher, edge_matcher=edge_matcher)
        )


# ruff: noqa: E501
# Output of
# import networkx as nx
# nx.write_graphml_lxml(nx.karate_club_graph(), open("karate.xml", "w"))
karate_xml = """
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"><key id="d2" for="edge" attr.name="weight" attr.type="long"/>
<key id="d1" for="node" attr.name="club" attr.type="string"/>
<key id="d0" for="graph" attr.name="name" attr.type="string"/>
<graph edgedefault="undirected"><data key="d0">Zachary's Karate Club</data>
<node id="0">
  <data key="d1">Mr. Hi</data>
</node>
<node id="1">
  <data key="d1">Mr. Hi</data>
</node>
<node id="2">
  <data key="d1">Mr. Hi</data>
</node>
<node id="3">
  <data key="d1">Mr. Hi</data>
</node>
<node id="4">
  <data key="d1">Mr. Hi</data>
</node>
<node id="5">
  <data key="d1">Mr. Hi</data>
</node>
<node id="6">
  <data key="d1">Mr. Hi</data>
</node>
<node id="7">
  <data key="d1">Mr. Hi</data>
</node>
<node id="8">
  <data key="d1">Mr. Hi</data>
</node>
<node id="9">
  <data key="d1">Officer</data>
</node>
<node id="10">
  <data key="d1">Mr. Hi</data>
</node>
<node id="11">
  <data key="d1">Mr. Hi</data>
</node>
<node id="12">
  <data key="d1">Mr. Hi</data>
</node>
<node id="13">
  <data key="d1">Mr. Hi</data>
</node>
<node id="14">
  <data key="d1">Officer</data>
</node>
<node id="15">
  <data key="d1">Officer</data>
</node>
<node id="16">
  <data key="d1">Mr. Hi</data>
</node>
<node id="17">
  <data key="d1">Mr. Hi</data>
</node>
<node id="18">
  <data key="d1">Officer</data>
</node>
<node id="19">
  <data key="d1">Mr. Hi</data>
</node>
<node id="20">
  <data key="d1">Officer</data>
</node>
<node id="21">
  <data key="d1">Mr. Hi</data>
</node>
<node id="22">
  <data key="d1">Officer</data>
</node>
<node id="23">
  <data key="d1">Officer</data>
</node>
<node id="24">
  <data key="d1">Officer</data>
</node>
<node id="25">
  <data key="d1">Officer</data>
</node>
<node id="26">
  <data key="d1">Officer</data>
</node>
<node id="27">
  <data key="d1">Officer</data>
</node>
<node id="28">
  <data key="d1">Officer</data>
</node>
<node id="29">
  <data key="d1">Officer</data>
</node>
<node id="30">
  <data key="d1">Officer</data>
</node>
<node id="31">
  <data key="d1">Officer</data>
</node>
<node id="32">
  <data key="d1">Officer</data>
</node>
<node id="33">
  <data key="d1">Officer</data>
</node>
<edge source="0" target="1">
  <data key="d2">4</data>
</edge>
<edge source="0" target="2">
  <data key="d2">5</data>
</edge>
<edge source="0" target="3">
  <data key="d2">3</data>
</edge>
<edge source="0" target="4">
  <data key="d2">3</data>
</edge>
<edge source="0" target="5">
  <data key="d2">3</data>
</edge>
<edge source="0" target="6">
  <data key="d2">3</data>
</edge>
<edge source="0" target="7">
  <data key="d2">2</data>
</edge>
<edge source="0" target="8">
  <data key="d2">2</data>
</edge>
<edge source="0" target="10">
  <data key="d2">2</data>
</edge>
<edge source="0" target="11">
  <data key="d2">3</data>
</edge>
<edge source="0" target="12">
  <data key="d2">1</data>
</edge>
<edge source="0" target="13">
  <data key="d2">3</data>
</edge>
<edge source="0" target="17">
  <data key="d2">2</data>
</edge>
<edge source="0" target="19">
  <data key="d2">2</data>
</edge>
<edge source="0" target="21">
  <data key="d2">2</data>
</edge>
<edge source="0" target="31">
  <data key="d2">2</data>
</edge>
<edge source="1" target="2">
  <data key="d2">6</data>
</edge>
<edge source="1" target="3">
  <data key="d2">3</data>
</edge>
<edge source="1" target="7">
  <data key="d2">4</data>
</edge>
<edge source="1" target="13">
  <data key="d2">5</data>
</edge>
<edge source="1" target="17">
  <data key="d2">1</data>
</edge>
<edge source="1" target="19">
  <data key="d2">2</data>
</edge>
<edge source="1" target="21">
  <data key="d2">2</data>
</edge>
<edge source="1" target="30">
  <data key="d2">2</data>
</edge>
<edge source="2" target="3">
  <data key="d2">3</data>
</edge>
<edge source="2" target="7">
  <data key="d2">4</data>
</edge>
<edge source="2" target="8">
  <data key="d2">5</data>
</edge>
<edge source="2" target="9">
  <data key="d2">1</data>
</edge>
<edge source="2" target="13">
  <data key="d2">3</data>
</edge>
<edge source="2" target="27">
  <data key="d2">2</data>
</edge>
<edge source="2" target="28">
  <data key="d2">2</data>
</edge>
<edge source="2" target="32">
  <data key="d2">2</data>
</edge>
<edge source="3" target="7">
  <data key="d2">3</data>
</edge>
<edge source="3" target="12">
  <data key="d2">3</data>
</edge>
<edge source="3" target="13">
  <data key="d2">3</data>
</edge>
<edge source="4" target="6">
  <data key="d2">2</data>
</edge>
<edge source="4" target="10">
  <data key="d2">3</data>
</edge>
<edge source="5" target="6">
  <data key="d2">5</data>
</edge>
<edge source="5" target="10">
  <data key="d2">3</data>
</edge>
<edge source="5" target="16">
  <data key="d2">3</data>
</edge>
<edge source="6" target="16">
  <data key="d2">3</data>
</edge>
<edge source="8" target="30">
  <data key="d2">3</data>
</edge>
<edge source="8" target="32">
  <data key="d2">3</data>
</edge>
<edge source="8" target="33">
  <data key="d2">4</data>
</edge>
<edge source="9" target="33">
  <data key="d2">2</data>
</edge>
<edge source="13" target="33">
  <data key="d2">3</data>
</edge>
<edge source="14" target="32">
  <data key="d2">3</data>
</edge>
<edge source="14" target="33">
  <data key="d2">2</data>
</edge>
<edge source="15" target="32">
  <data key="d2">3</data>
</edge>
<edge source="15" target="33">
  <data key="d2">4</data>
</edge>
<edge source="18" target="32">
  <data key="d2">1</data>
</edge>
<edge source="18" target="33">
  <data key="d2">2</data>
</edge>
<edge source="19" target="33">
  <data key="d2">1</data>
</edge>
<edge source="20" target="32">
  <data key="d2">3</data>
</edge>
<edge source="20" target="33">
  <data key="d2">1</data>
</edge>
<edge source="22" target="32">
  <data key="d2">2</data>
</edge>
<edge source="22" target="33">
  <data key="d2">3</data>
</edge>
<edge source="23" target="25">
  <data key="d2">5</data>
</edge>
<edge source="23" target="27">
  <data key="d2">4</data>
</edge>
<edge source="23" target="29">
  <data key="d2">3</data>
</edge>
<edge source="23" target="32">
  <data key="d2">5</data>
</edge>
<edge source="23" target="33">
  <data key="d2">4</data>
</edge>
<edge source="24" target="25">
  <data key="d2">2</data>
</edge>
<edge source="24" target="27">
  <data key="d2">3</data>
</edge>
<edge source="24" target="31">
  <data key="d2">2</data>
</edge>
<edge source="25" target="31">
  <data key="d2">7</data>
</edge>
<edge source="26" target="29">
  <data key="d2">4</data>
</edge>
<edge source="26" target="33">
  <data key="d2">2</data>
</edge>
<edge source="27" target="33">
  <data key="d2">4</data>
</edge>
<edge source="28" target="31">
  <data key="d2">2</data>
</edge>
<edge source="28" target="33">
  <data key="d2">2</data>
</edge>
<edge source="29" target="32">
  <data key="d2">4</data>
</edge>
<edge source="29" target="33">
  <data key="d2">2</data>
</edge>
<edge source="30" target="32">
  <data key="d2">3</data>
</edge>
<edge source="30" target="33">
  <data key="d2">3</data>
</edge>
<edge source="31" target="32">
  <data key="d2">4</data>
</edge>
<edge source="31" target="33">
  <data key="d2">4</data>
</edge>
<edge source="32" target="33">
  <data key="d2">5</data>
</edge>
</graph></graphml>
"""
