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
use rustworkx_core::dictmap::*;
use hashbrown::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{IntoEdgeReferences, IntoNodeReferences};
use petgraph::EdgeType;
use rustworkx_core::coloring::greedy_color;


/// Color a PyGraph using a largest_first strategy greedy graph coloring.
///
/// :param PyGraph: The input PyGraph object to color
///
/// :returns: A dictionary where keys are node indices and the value is
///     the color
/// :rtype: dict
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_greedy_color(py: Python, graph: &graph::PyGraph) -> PyResult<PyObject> {
    let colors = greedy_color(&graph.graph);

    let out_dict = PyDict::new(py);
    for (index, color) in colors {
        out_dict.set_item(index.index(), color)?;
    }
    Ok(out_dict.into())
}


fn line_graph<Ty: EdgeType>(
    py: Python,
    graph: &StablePyGraph<Ty>
) -> (StablePyGraph<Ty>, HashMap<EdgeIndex, NodeIndex>) {

    let mut out_graph = StablePyGraph::<Ty>::with_capacity(graph.edge_count(), 0);
    let mut out_edge_map = HashMap::<EdgeIndex, NodeIndex>::with_capacity(graph.edge_count());

    for edge in graph.edge_references() {
        let e0 = edge.id();
        let n0 = out_graph.add_node(py.None());
        out_edge_map.insert(e0, n0);
    }

    // There must be a better way to iterate over all pairs of edges, but I can't get
    // combinations() to work.
    for node in graph.node_references() {
        for edge0 in graph.edges(node.0) {
            for edge1 in graph.edges(node.0) {
                if edge0.id().index() < edge1.id().index() {
                    let node0 = out_edge_map.get(&edge0.id()).unwrap();
                    let node1 = out_edge_map.get(&edge1.id()).unwrap();
                    out_graph.add_edge(*node0, *node1, py.None());
                }
            }
        }
    }

    (out_graph, out_edge_map)
}


fn greedy_edge_color<Ty: EdgeType>(
    py: Python,
    graph: &StablePyGraph<Ty>
) -> DictMap<usize, usize> {
    let (line_graph, edge_to_node_map) = line_graph(py, &graph);
    let colors = greedy_color(&line_graph);

    let mut edge_colors: DictMap<usize, usize> = DictMap::new();

    for edge in graph.edge_references() {
        let e0 = edge.id();
        let n0 = edge_to_node_map.get(&e0).unwrap();
        let c0 = colors.get(n0).unwrap();
        edge_colors.insert(e0.index(), *c0);
    }
    edge_colors
}

#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_greedy_edge_color(py: Python, graph: &graph::PyGraph) -> PyResult<PyObject> {
    let edge_colors = greedy_edge_color(py, &graph.graph);

    let out_dict = PyDict::new(py);
    for (index, color) in edge_colors {
        out_dict.set_item(index, color)?;
    }
    Ok(out_dict.into())
}


