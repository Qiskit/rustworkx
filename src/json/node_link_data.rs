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

use hashbrown::HashMap;

use serde::{Deserialize, Serialize};

use pyo3::prelude::*;
use pyo3::Python;

use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use petgraph::EdgeType;

use crate::JSONSerializationError;
use crate::NodeIndex;
use crate::StablePyGraph;

#[derive(Serialize)]
pub struct Graph {
    pub directed: bool,
    pub multigraph: bool,
    pub attrs: Option<BTreeMap<String, String>>,
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
}

#[derive(Deserialize)]
pub struct GraphInput {
    pub directed: bool,
    pub multigraph: bool,
    pub attrs: Option<BTreeMap<String, String>>,
    pub nodes: Vec<NodeInput>,
    pub links: Vec<LinkInput>,
}

#[derive(Serialize)]
pub struct Node {
    id: usize,
    data: Option<BTreeMap<String, String>>,
}

#[derive(Deserialize)]
pub struct NodeInput {
    id: Option<usize>,
    data: Option<BTreeMap<String, String>>,
}

#[derive(Deserialize)]
pub struct LinkInput {
    source: usize,
    target: usize,
    #[allow(dead_code)]
    id: Option<usize>,
    data: Option<BTreeMap<String, String>>,
}

#[derive(Serialize)]
pub struct Link {
    source: usize,
    target: usize,
    id: usize,
    data: Option<BTreeMap<String, String>>,
}

#[allow(clippy::too_many_arguments)]
pub fn parse_node_link_data<Ty: EdgeType>(
    py: &Python,
    graph: GraphInput,
    out_graph: &mut StablePyGraph<Ty>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<()> {
    let mut id_mapping: HashMap<usize, NodeIndex> = HashMap::with_capacity(graph.nodes.len());
    for node in graph.nodes {
        let payload = match node.data {
            Some(data) => match node_attrs {
                Some(ref callback) => callback.call1(*py, (data,))?,
                None => data.to_object(*py),
            },
            None => py.None(),
        };
        let id = out_graph.add_node(payload);
        match node.id {
            Some(input_id) => id_mapping.insert(input_id, id),
            None => id_mapping.insert(id.index(), id),
        };
    }
    for edge in graph.links {
        let data = match edge.data {
            Some(data) => match edge_attrs {
                Some(ref callback) => callback.call1(*py, (data,))?,
                None => data.to_object(*py),
            },
            None => py.None(),
        };
        out_graph.add_edge(id_mapping[&edge.source], id_mapping[&edge.target], data);
    }
    Ok(())
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
