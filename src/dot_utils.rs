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
    Data, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, IntoNodeReferences, NodeIndexable,
    NodeRef,
};
use pyo3::prelude::*;

static TYPE: [&str; 2] = ["graph", "digraph"];
static EDGE: [&str; 2] = ["--", "->"];

pub fn build_dot<G, T>(
    py: Python,
    graph: G,
    file: &mut T,
    graph_attrs: Option<BTreeMap<String, String>>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<()>
where
    T: Write,
    G: GraphBase + IntoEdgeReferences + IntoNodeReferences + NodeIndexable + GraphProp,
    G: Data<NodeWeight = PyObject, EdgeWeight = PyObject>,
{
    writeln!(file, "{} {{", TYPE[graph.is_directed() as usize])?;
    if let Some(graph_attr_map) = graph_attrs {
        for (key, value) in graph_attr_map.iter() {
            writeln!(file, "{}={} ;", key, value)?;
        }
    }

    for node in graph.node_references() {
        writeln!(
            file,
            "{} {};",
            graph.to_index(node.id()),
            attr_map_to_string(py, node_attrs.as_ref(), node.weight())?
        )?;
    }
    for edge in graph.edge_references() {
        writeln!(
            file,
            "{} {} {} {};",
            graph.to_index(edge.source()),
            EDGE[graph.is_directed() as usize],
            graph.to_index(edge.target()),
            attr_map_to_string(py, edge_attrs.as_ref(), edge.weight())?
        )?;
    }
    writeln!(file, "}}")?;
    Ok(())
}

/// Convert an attr map to an output string
fn attr_map_to_string<'a>(
    py: Python,
    attrs: Option<&'a PyObject>,
    weight: &'a PyObject,
) -> PyResult<String> {
    if attrs.is_none() {
        return Ok("".to_string());
    }
    let attr_callable = |node: &'a PyObject| -> PyResult<BTreeMap<String, String>> {
        let res = attrs.unwrap().call1(py, (node,))?;
        res.extract(py)
    };

    let attrs = attr_callable(weight)?;
    if attrs.is_empty() {
        return Ok("".to_string());
    }

    let attr_string = attrs
        .iter()
        .map(|(key, value)| {
            if key == "label" {
                format!("{}=\"{}\"", key, value)
            } else {
                format!("{}={}", key, value)
            }
        })
        .collect::<Vec<String>>()
        .join(", ");
    Ok(format!("[{}]", attr_string))
}
