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

use crate::{digraph, graph};

use rustworkx_core::coloring::{greedy_edge_color, greedy_node_color, two_color};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;

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

/// Color edges of a :class:`~.PyGraph` object using a greedy approach.
///
/// This function works by greedily coloring the line graph of the given graph.
///
/// :param PyGraph: The input PyGraph object to edge-color
///
/// :returns: A dictionary where keys are edge indices and the value is the color
/// :rtype: dict
///
/// .. jupyter-execute::
///
///   import rustworkx as rx
///
///   graph = rx.generators.cycle_graph(7)
///   edge_colors = rx.graph_greedy_edge_color(graph)
///   assert edge_colors == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 2}
///
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_greedy_edge_color(py: Python, graph: &graph::PyGraph) -> PyResult<PyObject> {
    let colors = greedy_edge_color(&graph.graph);
    let out_dict = PyDict::new(py);
    for (node, color) in colors {
        out_dict.set_item(node.index(), color)?;
    }
    Ok(out_dict.into())
}

/// Compute a two-coloring of a graph
///
/// If a two coloring is not possible for the input graph (meaning it is not
/// bipartite), ``None`` is returned.
///
/// :param PyGraph graph: The graph to find the coloring for
///
/// :returns: If a coloring is possible return a dictionary of node indices to the color as an
/// integer (0 or 1)
/// :rtype: dict
#[pyfunction]
pub fn graph_two_color(py: Python, graph: &graph::PyGraph) -> PyResult<Option<PyObject>> {
    match two_color(&graph.graph) {
        Some(colors) => {
            let out_dict = PyDict::new(py);
            for (node, color) in colors {
                out_dict.set_item(node.index(), color)?;
            }
            Ok(Some(out_dict.into()))
        }
        None => Ok(None),
    }
}

/// Compute a two-coloring of a directed graph
///
/// If a two coloring is not possible for the input graph (meaning it is not
/// bipartite), ``None`` is returned.
///
/// :param PyDiGraph graph: The graph to find the coloring for
///
/// :returns: If a coloring is possible return a dictionary of node indices to the color as an
/// integer (0 or 1)
/// :rtype: dict
#[pyfunction]
pub fn digraph_two_color(py: Python, graph: &digraph::PyDiGraph) -> PyResult<Option<PyObject>> {
    match two_color(&graph.graph) {
        Some(colors) => {
            let out_dict = PyDict::new(py);
            for (node, color) in colors {
                out_dict.set_item(node.index(), color)?;
            }
            Ok(Some(out_dict.into()))
        }
        None => Ok(None),
    }
}
