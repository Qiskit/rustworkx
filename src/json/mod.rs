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

use std::fs::File;
use std::io::BufReader;

use crate::{digraph, graph, JSONDeserializationError, StablePyGraph};
use petgraph::{algo, Directed, Undirected};

use pyo3::prelude::*;
use pyo3::Python;

/// Parse a node-link format JSON file to generate a graph
///
/// :param path str: The path to the JSON file to load
/// :param graph_attrs: An optional callable that will be passed a dictionary
///     with string keys and string values and is expected to return a Python
///     object to use for :attr:`~.PyGraph.attrs` attribute of the output graph.
///     If not specified the dictionary with string keys and string values will
///     be used as the value for ``attrs``.
/// :param node_attrs: An optional callable that will be passed a dictionary with
///     string keys and string values representing the data payload
///     for each node in the graph and is expected to return a Python object to
///     use for the data payload of the node. If not specified the dictionary with
///     string keys and string values will be used for the nodes' data payload.
/// :param edge_attrs:  An optional callable that will be passed a dictionary with
///     string keys and string values representing the data payload
///     for each edge in the graph and is expected to return a Python object to
///     use for the data payload of the node. If not specified the dictionary with
///     string keys and string values will be used for the edge' data payload.
///
/// :returns: The graph represented by the node link JSON
/// :rtype: PyGraph | PyDiGraph
#[pyfunction]
#[pyo3(signature = (path, graph_attrs=None, node_attrs=None, edge_attrs=None))]
pub fn from_node_link_json_file(
    py: Python,
    path: &str,
    graph_attrs: Option<PyObject>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<PyObject> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let graph: node_link_data::GraphInput = match serde_json::from_reader(reader) {
        Ok(v) => v,
        Err(e) => {
            return Err(JSONDeserializationError::new_err(format!(
                "JSON Deserialization Error {}",
                e
            )));
        }
    };
    let attrs: PyObject = match graph.attrs {
        Some(ref attrs) => match graph_attrs {
            Some(ref callback) => callback.call1(py, (attrs.clone(),))?,
            None => attrs.to_object(py),
        },
        None => py.None(),
    };
    let multigraph = graph.multigraph;

    Ok(if graph.directed {
        let mut inner_graph: StablePyGraph<Directed> =
            StablePyGraph::with_capacity(graph.nodes.len(), graph.links.len());
        node_link_data::parse_node_link_data(&py, graph, &mut inner_graph, node_attrs, edge_attrs)?;
        digraph::PyDiGraph {
            graph: inner_graph,
            cycle_state: algo::DfsSpace::default(),
            check_cycle: false,
            node_removed: false,
            multigraph,
            attrs,
        }
        .into_py(py)
    } else {
        let mut inner_graph: StablePyGraph<Undirected> =
            StablePyGraph::with_capacity(graph.nodes.len(), graph.links.len());
        node_link_data::parse_node_link_data(&py, graph, &mut inner_graph, node_attrs, edge_attrs)?;

        graph::PyGraph {
            graph: inner_graph,
            node_removed: false,
            multigraph,
            attrs,
        }
        .into_py(py)
    })
}

/// Parse a node-link format JSON str to generate a graph
///
/// :param data str: The JSON str to parse
/// :param graph_attrs: An optional callable that will be passed a dictionary
///     with string keys and string values and is expected to return a Python
///     object to use for :attr:`~.PyGraph.attrs` attribute of the output graph.
///     If not specified the dictionary with string keys and string values will
///     be used as the value for ``attrs``.
/// :param node_attrs: An optional callable that will be passed a dictionary with
///     string keys and string values representing the data payload
///     for each node in the graph and is expected to return a Python object to
///     use for the data payload of the node. If not specified the dictionary with
///     string keys and string values will be used for the nodes' data payload.
/// :param edge_attrs:  An optional callable that will be passed a dictionary with
///     string keys and string values representing the data payload
///     for each edge in the graph and is expected to return a Python object to
///     use for the data payload of the node. If not specified the dictionary with
///     string keys and string values will be used for the edge' data payload.
///
/// :returns: The graph represented by the node link JSON
/// :rtype: PyGraph | PyDiGraph
#[pyfunction]
#[pyo3(signature = (data, graph_attrs=None, node_attrs=None, edge_attrs=None))]
pub fn parse_node_link_json(
    py: Python,
    data: &str,
    graph_attrs: Option<PyObject>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<PyObject> {
    let graph: node_link_data::GraphInput = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            return Err(JSONDeserializationError::new_err(format!(
                "JSON Deserialization Error {}",
                e
            )));
        }
    };
    let attrs: PyObject = match graph.attrs {
        Some(ref attrs) => match graph_attrs {
            Some(ref callback) => callback.call1(py, (attrs.clone(),))?,
            None => attrs.to_object(py),
        },
        None => py.None(),
    };
    let multigraph = graph.multigraph;
    Ok(if graph.directed {
        let mut inner_graph: StablePyGraph<Directed> =
            StablePyGraph::with_capacity(graph.nodes.len(), graph.links.len());
        node_link_data::parse_node_link_data(&py, graph, &mut inner_graph, node_attrs, edge_attrs)?;
        digraph::PyDiGraph {
            graph: inner_graph,
            cycle_state: algo::DfsSpace::default(),
            check_cycle: false,
            node_removed: false,
            multigraph,
            attrs,
        }
        .into_py(py)
    } else {
        let mut inner_graph: StablePyGraph<Undirected> =
            StablePyGraph::with_capacity(graph.nodes.len(), graph.links.len());
        node_link_data::parse_node_link_data(&py, graph, &mut inner_graph, node_attrs, edge_attrs)?;
        graph::PyGraph {
            graph: inner_graph,
            node_removed: false,
            multigraph,
            attrs,
        }
        .into_py(py)
    })
}

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
    text_signature = "(graph, /, path=None, graph_attrs=None, node_attrs=None, edge_attrs=None)",
    signature = (graph, path=None, graph_attrs=None, node_attrs=None, edge_attrs=None)
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
    text_signature = "(graph, /, path=None, graph_attrs=None, node_attrs=None, edge_attrs=None)",
    signature = (graph, path=None, graph_attrs=None, node_attrs=None, edge_attrs=None)
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
