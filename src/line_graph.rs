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

use crate::graph;
use crate::StablePyGraph;
use hashbrown::HashMap;
use petgraph::Undirected;
use rustworkx_core::dictmap::*;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences};
use petgraph::EdgeType;
use pyo3::pyclass::boolean_struct::True;
use rustworkx_core::coloring::greedy_node_color;
use rustworkx_core::line_graph::line_graph;

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
