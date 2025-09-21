use pest::Parser;
use pest_derive::Parser;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::StablePyGraph;
use crate::digraph::PyDiGraph;
use crate::graph::PyGraph;

use hashbrown::HashMap;
use rustworkx_core::petgraph::prelude::{Directed, NodeIndex, Undirected};

#[derive(Parser)]
#[grammar = "dot_parser/dot.pest"]
pub struct DotParser;

/// Keep a single graph value that can be either directed or undirected. This avoids generic return-type mismatches.
enum DotGraph {
    Directed(StablePyGraph<Directed>),
    Undirected(StablePyGraph<Undirected>),
}

impl DotGraph {
    fn new_directed() -> Self {
        DotGraph::Directed(StablePyGraph::<Directed>::with_capacity(0, 0))
    }
    fn new_undirected() -> Self {
        DotGraph::Undirected(StablePyGraph::<Undirected>::with_capacity(0, 0))
    }
    fn add_node(&mut self, w: PyObject) -> NodeIndex {
        match self {
            DotGraph::Directed(g) => g.add_node(w),
            DotGraph::Undirected(g) => g.add_node(w),
        }
    }
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, w: PyObject) {
        match self {
            DotGraph::Directed(g) => {
                g.add_edge(a, b, w);
            }
            DotGraph::Undirected(g) => {
                g.add_edge(a, b, w);
            }
        }
    }

    #[allow(dead_code)]
    fn is_directed(&self) -> bool {
        matches!(self, DotGraph::Directed(_))
    }

    fn into_inner(self) -> Result<StablePyGraph<Directed>, StablePyGraph<Undirected>> {
        match self {
            DotGraph::Directed(g) => Ok(g),
            DotGraph::Undirected(g) => Err(g),
        }
    }
}

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

    // Detect directedness from a clone of the iterator so we don't consume it.
    let mut is_directed = false;
    for pair in pairs.clone() {
        if pair.as_rule() != Rule::graph_file {
            continue;
        }
        let mut inner = pair.into_inner();
        let first = inner.next().unwrap();
        let graph_type_str = if first.as_rule() == Rule::strict {
            inner.next().unwrap().as_str()
        } else {
            first.as_str()
        };
        is_directed = graph_type_str == "digraph";
        break;
    }

    build_graph_enum(py, pairs, is_directed)
}

fn build_graph_enum(
    py: Python<'_>,
    pairs: pest::iterators::Pairs<Rule>,
    is_directed: bool,
) -> PyResult<PyObject> {
    let mut node_map: HashMap<String, NodeIndex> = HashMap::new();
    let graph_attrs = PyDict::new(py);

    let mut default_node_attrs: HashMap<String, String> = HashMap::new();
    let mut default_edge_attrs: HashMap<String, String> = HashMap::new();

    let mut node_attrs_map: HashMap<String, PyObject> = HashMap::new();

    let mut graph = if is_directed {
        DotGraph::new_directed()
    } else {
        DotGraph::new_undirected()
    };

    for pair in pairs {
        if pair.as_rule() != Rule::graph_file {
            continue;
        }
        let mut inner = pair.into_inner();
        let first = inner.next().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing graph type in DOT")
        })?;
        if first.as_rule() == Rule::strict {
            inner.next();
        }

        for rest in inner {
            if rest.as_rule() != Rule::stmt_list {
                continue;
            }

            for stmt in rest.into_inner() {
                match stmt.as_rule() {
                    Rule::node_stmt => {
                        let mut it = stmt.into_inner();
                        let nid = it.next().ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Missing node id in DOT",
                            )
                        })?;
                        let name = node_id_to_string(nid);
                        let py_node_obj: PyObject = PyString::new(py, &name).into();

                        let idx = graph.add_node(py_node_obj);
                        node_map.insert(name.clone(), idx);

                        // Merge default node attrs + node's attr_list
                        let merged = PyDict::new(py);
                        for (k, v) in default_node_attrs.iter() {
                            merged.set_item(k.as_str(), v.as_str())?;
                        }
                        for maybe_attr in it {
                            if maybe_attr.as_rule() == Rule::attr_list {
                                let map = parse_attr_list_to_map(maybe_attr);
                                for (k, v) in map {
                                    merged.set_item(k.as_str(), v.as_str())?;
                                }
                            }
                        }
                        node_attrs_map.insert(name.clone(), merged.into());
                    }

                    Rule::edge_stmt => {
                        let mut endpoints: Vec<String> = Vec::new();

                        // Start collected edge attrs from defaults
                        let collected = PyDict::new(py);
                        for (k, v) in default_edge_attrs.iter() {
                            collected.set_item(k.as_str(), v.as_str())?;
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
                                    // we already know directedness
                                }
                                Rule::attr_list => {
                                    let map = parse_attr_list_to_map(child);
                                    for (k, v) in map {
                                        collected.set_item(k.as_str(), v.as_str())?;
                                    }
                                }
                                _ => {}
                            }
                        }

                        // Pairwise edges along the chain
                        for i in 0..endpoints.len().saturating_sub(1) {
                            let src = endpoints[i].clone();
                            let dst = endpoints[i + 1].clone();

                            let src_idx = *node_map.entry(src.clone()).or_insert_with(|| {
                                let py_node: PyObject = PyString::new(py, &src).into();
                                graph.add_node(py_node)
                            });

                            let dst_idx = *node_map.entry(dst.clone()).or_insert_with(|| {
                                let py_node: PyObject = PyString::new(py, &dst).into();
                                graph.add_node(py_node)
                            });

                            let edge_attrs_obj: PyObject = collected.clone().into();
                            graph.add_edge(src_idx, dst_idx, edge_attrs_obj);
                        }
                    }

                    Rule::attr_stmt => {
                        // attr_stmt = ("graph" | "node" | "edge") ~ attr_list+
                        let mut it = stmt.into_inner();
                        if let Some(target_pair) = it.next() {
                            let target = target_pair.as_str();
                            for rest in it {
                                if rest.as_rule() == Rule::attr_list {
                                    let map = parse_attr_list_to_map(rest);
                                    match target {
                                        "graph" => {
                                            for (k, v) in map {
                                                graph_attrs.set_item(k.as_str(), v.as_str())?;
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
                        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                            "subgraph parsing is not supported",
                        ));
                    }

                    _ => {}
                }
            }
        }
    }

    // Wrap into the a Python class
    match graph.into_inner() {
        Ok(directed_graph) => {
            let dg = PyDiGraph {
                graph: directed_graph,
                cycle_state: rustworkx_core::petgraph::algo::DfsSpace::default(),
                check_cycle: false,
                node_removed: false,
                multigraph: true,
                attrs: graph_attrs.clone().into(),
            };
            Ok(Py::new(py, dg)?.into())
        }
        Err(undirected_graph) => {
            let ug = PyGraph {
                graph: undirected_graph,
                node_removed: false,
                multigraph: true,
                attrs: graph_attrs.clone().into(),
            };
            Ok(Py::new(py, ug)?.into())
        }
    }
}
