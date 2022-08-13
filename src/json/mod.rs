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

mod node_link_data;

use crate::{digraph, graph};

use pyo3::prelude::*;
use pyo3::Python;

/// Generate a JSON object representing a :class:`~.PyDiGraph` in a node-link format
///
/// :param PyDiGraph graph: The graph to generate the JSON for
/// :param str path: An optional path to write the JSON output to. If specified
///     the function will not return anything and instead will write the JSON
///     to the file specified.
/// :param graph_attrs: An optional callable that will be passed the
///     :attr:`~.PyDiGraph.attrs` attribute of the graph and is expected to
///     return a dictionary of string keys to string values representing the
///     graph attributes. This dictionary will be included as attributes in
///     the output JSON. If anything other than a dictionary with string keys
///     and string values is returned an exception will be raised.
/// :param node_attrs: An optional callable that will be passed the node data
///     payload for each node in the graph and is expected to return a
///     dictionary of string keys to string values representing the data payload.
///     This dictionary will be used as the ``data`` field for each node.
/// :param edge_attrs:  An optional callable that will be passed the edge data
///     payload for each node in the graph and is expected to return a
///     dictionary of string keys to string values representing the data payload.
///     This dictionary will be used as the ``data`` field for each edge.
///
/// :returns: Either the JSON string for the payload or ``None`` if ``path`` is specified
/// :rtype: str
#[pyfunction]
#[pyo3(
    text_signature = "(graph, /, path=None, graph_attrs=None, node_attrs=None, edge_attrs=None)"
)]
pub fn digraph_node_link_json(
    py: Python,
    graph: &digraph::PyDiGraph,
    path: Option<String>,
    graph_attrs: Option<PyObject>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<Option<String>> {
    node_link_data::node_link_data(
        py,
        &graph.graph,
        graph.multigraph,
        &graph.attrs,
        path,
        graph_attrs,
        node_attrs,
        edge_attrs,
    )
}

/// Generate a JSON object representing a :class:`~.PyGraph` in a node-link format
///
/// :param PyGraph graph: The graph to generate the JSON for
/// :param str path: An optional path to write the JSON output to. If specified
///     the function will not return anything and instead will write the JSON
///     to the file specified.
/// :param graph_attrs: An optional callable that will be passed the
///     :attr:`~.PyGraph.attrs` attribute of the graph and is expected to
///     return a dictionary of string keys to string values representing the
///     graph attributes. This dictionary will be included as attributes in
///     the output JSON. If anything other than a dictionary with string keys
///     and string values is returned an exception will be raised.
/// :param node_attrs: An optional callable that will be passed the node data
///     payload for each node in the graph and is expected to return a
///     dictionary of string keys to string values representing the data payload.
///     This dictionary will be used as the ``data`` field for each node.
/// :param edge_attrs:  An optional callable that will be passed the edge data
///     payload for each node in the graph and is expected to return a
///     dictionary of string keys to string values representing the data payload.
///     This dictionary will be used as the ``data`` field for each edge.
///
/// :returns: Either the JSON string for the payload or ``None`` if ``path`` is specified
/// :rtype: str
#[pyfunction]
#[pyo3(
    text_signature = "(graph, /, path=None, graph_attrs=None, node_attrs=None, edge_attrs=None)"
)]
pub fn graph_node_link_json(
    py: Python,
    graph: &graph::PyGraph,
    path: Option<String>,
    graph_attrs: Option<PyObject>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<Option<String>> {
    node_link_data::node_link_data(
        py,
        &graph.graph,
        graph.multigraph,
        &graph.attrs,
        path,
        graph_attrs,
        node_attrs,
        edge_attrs,
    )
}
