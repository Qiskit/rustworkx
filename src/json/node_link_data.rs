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

use std::collections::BTreeMap;
use std::fs::File;

use serde::{Deserialize, Serialize};

use pyo3::prelude::*;
use pyo3::Python;

use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use petgraph::EdgeType;

use crate::JSONSerializationError;
use crate::StablePyGraph;

#[derive(Serialize, Deserialize)]
struct Graph {
    directed: bool,
    multigraph: bool,
    attrs: Option<BTreeMap<String, String>>,
    nodes: Vec<Node>,
    links: Vec<Link>,
}

#[derive(Serialize, Deserialize)]
struct Node {
    id: usize,
    data: Option<BTreeMap<String, String>>,
}

#[derive(Serialize, Deserialize)]
struct Link {
    source: usize,
    target: usize,
    id: usize,
    data: Option<BTreeMap<String, String>>,
}

#[allow(clippy::too_many_arguments)]
pub fn node_link_data<Ty: EdgeType>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    multigraph: bool,
    attrs: &PyObject,
    path: Option<String>,
    graph_attrs: Option<PyObject>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<Option<String>> {
    let attr_callable = |attrs: &PyObject, obj: &PyObject| -> PyResult<BTreeMap<String, String>> {
        let res = attrs.call1(py, (obj,))?;
        res.extract(py)
    };
    let mut nodes: Vec<Node> = Vec::with_capacity(graph.node_count());
    for n in graph.node_indices() {
        let data = match node_attrs {
            Some(ref callback) => Some(attr_callable(callback, &graph[n])?),
            None => None,
        };
        nodes.push(Node {
            id: n.index(),
            data,
        });
    }
    let mut links: Vec<Link> = Vec::with_capacity(graph.edge_count());
    for e in graph.edge_references() {
        let data = match edge_attrs {
            Some(ref callback) => Some(attr_callable(callback, e.weight())?),
            None => None,
        };
        links.push(Link {
            source: e.source().index(),
            target: e.target().index(),
            id: e.id().index(),
            data,
        });
    }

    let graph_attrs = match graph_attrs {
        Some(ref callback) => Some(attr_callable(callback, attrs)?),
        None => None,
    };

    let output_struct = Graph {
        directed: graph.is_directed(),
        multigraph,
        attrs: graph_attrs,
        nodes,
        links,
    };
    match path {
        None => match serde_json::to_string(&output_struct) {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(JSONSerializationError::new_err(format!(
                "JSON Error: {}",
                e
            ))),
        },
        Some(filename) => {
            let file = File::create(filename)?;
            match serde_json::to_writer(file, &output_struct) {
                Ok(_) => Ok(None),
                Err(e) => Err(JSONSerializationError::new_err(format!(
                    "JSON Error: {}",
                    e
                ))),
            }
        }
    }
}
