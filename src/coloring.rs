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

/// Color a :class:`~.PyGraph` object using a greedy graph coloring algorithm.
///
/// This function uses a `largest-first` strategy as described in [1]_ and colors
/// the nodes with higher degree first.
///
/// .. note::
///
///     The coloring problem is NP-hard and this is a heuristic algorithm which
///     may not return an optimal solution.
///
/// :param PyGraph: The input PyGraph object to color
///
/// :returns: A dictionary where keys are node indices and the value is
///     the color
/// :rtype: dict
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///     from rustworkx.visualization import mpl_draw
///
///     graph = rx.generators.generalized_petersen_graph(5, 2)
///     coloring = rx.graph_greedy_color(graph)
///     colors = [coloring[node] for node in graph.node_indices()]
///
///     # Draw colored graph
///     layout = rx.shell_layout(graph, nlist=[[0, 1, 2, 3, 4],[6, 7, 8, 9, 5]])
///     mpl_draw(graph, node_color=colors, pos=layout)
///
///
/// .. [1] Adrian Kosowski, and Krzysztof Manuszewski, Classical Coloring of Graphs,
///     Graph Colorings, 2-19, 2004. ISBN 0-8218-3458-4.
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_greedy_color(py: Python, graph: &graph::PyGraph) -> PyResult<PyObject> {
    let colors = greedy_node_color(&graph.graph);
    let out_dict = PyDict::new(py);
    for (node, color) in colors {
        out_dict.set_item(node.index(), color)?;
    }
    Ok(out_dict.into())
}

//graph::PyGraph
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

fn line_graph_tmp<Ty: EdgeType>(
    py: Python,
    graph: &StablePyGraph<Ty>,
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

fn greedy_edge_color<Ty: EdgeType>(py: Python, graph: &StablePyGraph<Ty>) -> DictMap<usize, usize> {
    let (line_graph, edge_to_node_map) = line_graph_tmp(py, graph);
    let colors = greedy_node_color(&line_graph);

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
    println!("Running graph_greedy_edge_color....");
    let edge_colors = greedy_edge_color(py, &graph.graph);

    let out_dict = PyDict::new(py);
    for (index, color) in edge_colors {
        out_dict.set_item(index, color)?;
    }
    Ok(out_dict.into())
}
