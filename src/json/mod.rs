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

#[pyfunction]
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

#[pyfunction]
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
