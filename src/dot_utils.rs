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
use std::io::prelude::*;

use petgraph::visit::{
    Data, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences,
    IntoNodeReferences, NodeIndexable, NodeRef,
};
use pyo3::prelude::*;

static TYPE: [&str; 2] = ["graph", "digraph"];
static EDGE: [&str; 2] = ["--", "->"];

pub fn build_dot<G, N, E, T>(
    graph: G,
    file: &mut T,
    graph_attrs: Option<BTreeMap<String, String>>,
    mut node_attrs: N,
    mut edge_attrs: E,
) -> PyResult<()>
where
    T: Write,
    G: GraphBase
        + IntoEdgeReferences
        + IntoNodeReferences
        + NodeIndexable
        + GraphProp,
    G: Data<NodeWeight = PyObject, EdgeWeight = PyObject>,
    N: FnMut(&PyObject) -> PyResult<BTreeMap<String, String>>,
    E: FnMut(&PyObject) -> PyResult<BTreeMap<String, String>>,
{
    writeln!(file, "{} {{", TYPE[graph.is_directed() as usize])?;
    if let Some(graph_attr_map) = graph_attrs {
        for (key, value) in graph_attr_map.iter() {
            writeln!(file, "{}={} ;", key, value)?;
        }
    }

    for node in graph.node_references() {
        let node_attr_map = node_attrs(&node.weight())?;
        writeln!(
            file,
            "{} {};",
            graph.to_index(node.id()),
            attr_map_to_string(node_attr_map)
        )?;
    }
    for edge in graph.edge_references() {
        let edge_attr_map = edge_attrs(&edge.weight())?;
        writeln!(
            file,
            "{} {} {} {};",
            graph.to_index(edge.source()),
            EDGE[graph.is_directed() as usize],
            graph.to_index(edge.target()),
            attr_map_to_string(edge_attr_map)
        )?;
    }
    writeln!(file, "}}")?;
    Ok(())
}

/// Convert an attr map to an output string
fn attr_map_to_string(attrs: BTreeMap<String, String>) -> String {
    if attrs.is_empty() {
        return "".to_string();
    }
    let attr_string = attrs
        .iter()
        .map(|(key, value)| format!("{}={}", key, value))
        .collect::<Vec<String>>()
        .join(", ");
    format!("[{}]", attr_string)
}
