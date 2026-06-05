// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

use crate::{StablePyGraph, graph};

use hashbrown::HashMap;

use petgraph::Undirected;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use rustworkx_core::dictmap::*;
use rustworkx_core::line_graph::line_graph;

use pyo3::Python;
use pyo3::prelude::*;

/// Constructs the line graph of a :class:`~.PyGraph` object.
///
/// The line graph `L(G)` of a graph `G` represents the adjacencies between edges of G.
/// `L(G)` contains a vertex for every edge in `G`, and `L(G)` contains an edge between two
/// vertices if the corresponding edges in `G` have a vertex in common.
///
/// :param PyGraph: The input PyGraph object
///
/// :returns: A new PyGraph object that is the line graph of ``graph``, and the dictionary
///     where the keys are indices of edges in``graph`` and the values are the corresponding
///     indices of nodes in the linear graph.
/// :rtype: Tuple[:class:`~rustworkx.PyGraph`, dict]
///
/// .. jupyter-execute::
///
///   import rustworkx as rx
///
///   graph = rx.PyGraph()
///   node_a = graph.add_node("a")
///   node_b = graph.add_node("b")
///   node_c = graph.add_node("c")
///   node_d = graph.add_node("d")
///   edge_ab = graph.add_edge(node_a, node_b, 1)
///   edge_ac = graph.add_edge(node_a, node_c, 1)
///   edge_bc = graph.add_edge(node_b, node_c, 1)
///   edge_ad = graph.add_edge(node_a, node_d, 1)
///
///   out_graph, out_edge_map = rx.graph_line_graph(graph)
///   assert out_graph.node_indices() == [0, 1, 2, 3]
///   assert out_graph.edge_list() == [(3, 1), (3, 0), (1, 0), (2, 0), (2, 1)]
///   assert out_edge_map == {edge_ab: 0, edge_ac: 1, edge_bc: 2, edge_ad: 3}
///
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_line_graph(
    py: Python,
    graph: &graph::PyGraph,
) -> (graph::PyGraph, DictMap<usize, usize>) {
    let default_fn = || py.None();

    let (output_graph, output_edge_to_node_map): (
        StablePyGraph<Undirected>,
        HashMap<EdgeIndex, NodeIndex>,
    ) = line_graph(&graph.graph, default_fn, default_fn);

    let output_graph_py = graph::PyGraph {
        graph: output_graph,
        node_removed: false,
        multigraph: false,
        attrs: py.None(),
    };

    let mut output_edge_to_node_map_py: DictMap<usize, usize> = DictMap::new();

    for edge in graph.graph.edge_references() {
        let edge_id = edge.id();
        let node_id = output_edge_to_node_map.get(&edge_id).unwrap();
        output_edge_to_node_map_py.insert(edge_id.index(), node_id.index());
    }

    (output_graph_py, output_edge_to_node_map_py)
}
