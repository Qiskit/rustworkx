use pest::Parser;
use pest_derive::Parser;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::digraph::PyDiGraph;
use crate::graph::PyGraph;
use crate::StablePyGraph;

use rustworkx_core::petgraph::prelude::{Directed, NodeIndex, Undirected};
use std::collections::HashMap;

#[derive(Parser)]
#[grammar = "dot_parser/dot.pest"]
pub struct DotParser;

/// Unquote a quoted string
fn unquote_str(s: &str) -> String {
    let t = s.trim();
    if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
        t[1..t.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
    } else {
        t.to_string()
    }
}

/// Parse an `attr_list` pair into a Rust HashMap<String,String>
fn parse_attr_list_to_map(pair: pest::iterators::Pair<Rule>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for a_list in pair.into_inner() {
        if a_list.as_rule() != Rule::a_list {
            continue;
        }
        let tokens: Vec<_> = a_list.into_inner().collect();
        let mut i = 0usize;
        while i < tokens.len() {
            // tokens are (id) (maybe "=") (maybe id), but pyparsing/pest flattening depends on grammar.
            // The simple approach here: take token i as key; if token i+1 exists and is not a comma (we filtered commas out in grammar),
            // treat it as value.
            let key = tokens[i].as_str().trim().to_string();
            if i + 1 < tokens.len() {
                let val = tokens[i + 1].as_str().trim().to_string();
                map.insert(key, unquote_str(&val));
                i += 2;
            } else {
                map.insert(key, String::new());
                i += 1;
            }
        }
    }
    map
}

/// Extract the first inner token of node_id
fn node_id_to_string(pair: pest::iterators::Pair<Rule>) -> String {
    if let Some(child) = pair.into_inner().next() {
        return unquote_str(child.as_str().trim());
    }
    String::new()
}

#[pyfunction]
pub fn from_dot(py: Python<'_>, dot_str: &str) -> PyResult<PyObject> {
    let pairs = DotParser::parse(Rule::graph_file, dot_str).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("DOT parse error: {}", e))
    })?;

    let mut is_directed = false;
    let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

    let mut di_graph: Option<PyDiGraph> = None;
    let mut undi_graph: Option<PyGraph> = None;

    let graph_attrs = PyDict::new(py);

    let mut default_node_attrs: HashMap<String, String> = HashMap::new();
    let mut default_edge_attrs: HashMap<String, String> = HashMap::new();

    let mut node_attrs_map: HashMap<String, PyObject> = HashMap::new();

    for pair in pairs {
        if pair.as_rule() != Rule::graph_file {
            continue;
        }

        let mut inner = pair.into_inner();

        // handle graph_type
        let first = inner.next().unwrap();
        let graph_type_str = if first.as_rule() == Rule::strict {
            inner.next().unwrap().as_str()
        } else {
            first.as_str()
        };
        is_directed = graph_type_str == "digraph";

        if is_directed {
            let graph = StablePyGraph::<Directed>::with_capacity(0, 0);
            di_graph = Some(PyDiGraph {
                graph,
                cycle_state: rustworkx_core::petgraph::algo::DfsSpace::default(),
                check_cycle: false,
                node_removed: false,
                multigraph: true,
                attrs: graph_attrs.clone().into(),
            });
        } else {
            let graph = StablePyGraph::<Undirected>::with_capacity(0, 0);
            undi_graph = Some(PyGraph {
                graph,
                node_removed: false,
                multigraph: true,
                attrs: graph_attrs.clone().into(),
            });
        }

        // handle stmt_list
        for rest in inner {
            if rest.as_rule() != Rule::stmt_list {
                continue;
            }

            for stmt in rest.into_inner() {
                match stmt.as_rule() {
                    Rule::node_stmt => {
                        let mut it = stmt.into_inner();
                        let nid = it.next().unwrap();
                        let name = node_id_to_string(nid);
                        // create python node object
                        let py_node_obj = PyString::new(py, &name).into();

                        // add node to graph
                        let idx = if is_directed {
                            di_graph.as_mut().unwrap().graph.add_node(py_node_obj)
                        } else {
                            undi_graph.as_mut().unwrap().graph.add_node(py_node_obj)
                        };
                        node_map.insert(name.clone(), idx);

                        // produce merged attrs
                        let merged = PyDict::new(py);
                        for (k, v) in default_node_attrs.iter() {
                            merged.set_item(k, v)?;
                        }
                        for maybe_attr in it {
                            if maybe_attr.as_rule() == Rule::attr_list {
                                let map = parse_attr_list_to_map(maybe_attr);
                                for (k, v) in map {
                                    merged.set_item(k, v)?;
                                }
                            }
                        }
                        node_attrs_map.insert(name.clone(), merged.into());
                    }

                    Rule::edge_stmt => {
                        // edge_stmt
                        let mut endpoints: Vec<String> = Vec::new();
                        let mut operators: Vec<String> = Vec::new();

                        // start collected edge attrs from defaults
                        let collected = PyDict::new(py);
                        for (k, v) in default_edge_attrs.iter() {
                            collected.set_item(k, v)?;
                        }

                        for child in stmt.into_inner() {
                            match child.as_rule() {
                                Rule::edge_point => {
                                    for ep_child in child.into_inner() {
                                        if ep_child.as_rule() == Rule::node_id {
                                            let n = node_id_to_string(ep_child);
                                            endpoints.push(n);
                                        }
                                    }
                                }
                                Rule::edge_op => {
                                    operators.push(child.as_str().trim().to_string());
                                }
                                Rule::attr_list => {
                                    let map = parse_attr_list_to_map(child);
                                    for (k, v) in map {
                                        collected.set_item(k, v)?;
                                    }
                                }
                                _ => {}
                            }
                        }

                        // create pairwise edges and add them to the graph
                        for i in 0..endpoints.len().saturating_sub(1) {
                            let src = endpoints[i].clone();
                            let dst = endpoints[i + 1].clone();

                            let src_idx = *node_map.entry(src.clone()).or_insert_with(|| {
                                let py_node = PyString::new(py, &src).into();
                                if is_directed {
                                    di_graph.as_mut().unwrap().graph.add_node(py_node)
                                } else {
                                    undi_graph.as_mut().unwrap().graph.add_node(py_node)
                                }
                            });

                            let dst_idx = *node_map.entry(dst.clone()).or_insert_with(|| {
                                let py_node = PyString::new(py, &dst).into();
                                if is_directed {
                                    di_graph.as_mut().unwrap().graph.add_node(py_node)
                                } else {
                                    undi_graph.as_mut().unwrap().graph.add_node(py_node)
                                }
                            });

                            let edge_attrs_obj: PyObject = collected.clone().into();

                            if is_directed {
                                di_graph.as_mut().unwrap().graph.add_edge(
                                    src_idx,
                                    dst_idx,
                                    edge_attrs_obj.clone(),
                                );
                            } else {
                                undi_graph.as_mut().unwrap().graph.add_edge(
                                    src_idx,
                                    dst_idx,
                                    edge_attrs_obj.clone(),
                                );
                            }
                        }
                    }

                    Rule::attr_stmt => {
                        let mut it = stmt.into_inner();
                        if let Some(target_pair) = it.next() {
                            let target = target_pair.as_str();
                            for rest in it {
                                if rest.as_rule() == Rule::attr_list {
                                    let map = parse_attr_list_to_map(rest);
                                    match target {
                                        "graph" => {
                                            for (k, v) in map {
                                                graph_attrs.set_item(k, v)?;
                                            }
                                            if let Some(dg) = di_graph.as_mut() {
                                                dg.attrs = graph_attrs.clone().into();
                                            }
                                            if let Some(ug) = undi_graph.as_mut() {
                                                ug.attrs = graph_attrs.clone().into();
                                            }
                                        }
                                        "node" => {
                                            for (k, v) in map {
                                                default_node_attrs.insert(k, v);
                                            }
                                        }
                                        "edge" => {
                                            for (k, v) in map {
                                                default_edge_attrs.insert(k, v);
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }

                    Rule::assignment => {
                        let mut parts = stmt.into_inner();
                        let key = parts.next().map(|p| p.as_str()).unwrap_or("");
                        let val = parts.next().map(|p| p.as_str()).unwrap_or("");
                        graph_attrs.set_item(key, val)?;
                    }

                    Rule::subgraph => {
                        // TODO: subgraph handling
                    }

                    _ => {}
                }
            }
        }
    }

    if is_directed {
        let dg = di_graph
            .take()
            .expect("directed graph was created but now missing");
        let py_obj = Py::new(py, dg)?.into();
        Ok(py_obj)
    } else {
        let ug = undi_graph
            .take()
            .expect("undirected graph was created but now missing");
        let py_obj = Py::new(py, ug)?.into();
        Ok(py_obj)
    }
}
