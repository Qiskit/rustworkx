use pest::Parser;
use pest_derive::Parser;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::IntoPy;

use crate::StablePyGraph;
use crate::digraph::PyDiGraph;
use crate::graph::PyGraph;

use rustworkx_core::petgraph::prelude::{NodeIndex, Directed, Undirected};
use std::collections::HashMap;

#[derive(Parser)]
#[grammar = "dot_parser/dot.pest"]
pub struct DotParser;

#[pyfunction]
pub fn from_dot(py: Python<'_>, dot_str: &str) -> PyResult<PyObject> {
    let pairs = DotParser::parse(Rule::graph_file, dot_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("DOT parse error: {}", e)))?;

    let mut is_directed = false;
    let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

    let mut di_graph: Option<PyDiGraph> = None;
    let mut undi_graph: Option<PyGraph> = None;

    // Initialize an empty dict of attributes for the graph
    let mut graph_attrs = PyDict::new(py).into_py(py);

    for pair in pairs {
        if pair.as_rule() != Rule::graph_file {
            continue;
        }

        let mut inner = pair.into_inner();
        // Handle optional 'strict' keyword
        let first = inner.next().unwrap();
        let graph_type = if first.as_rule() == Rule::strict {
            inner.next().unwrap().as_str()
        } else {
            first.as_str()
        };

        is_directed = graph_type == "digraph";

        // Create appropriate graph wrapper
        if is_directed {
            let graph = StablePyGraph::<Directed>::with_capacity(0, 0);
            di_graph = Some(PyDiGraph {
                graph,
                cycle_state: rustworkx_core::petgraph::algo::DfsSpace::default(),
                check_cycle: false,
                node_removed: false,
                multigraph: true,
                attrs: graph_attrs.clone_ref(py),
            });
        } else {
            let graph = StablePyGraph::<Undirected>::with_capacity(0, 0);
            undi_graph = Some(PyGraph {
                graph,
                node_removed: false,
                multigraph: true,
                attrs: graph_attrs.clone_ref(py),
            });
        }

        for stmt_pair in inner {
            if stmt_pair.as_rule() != Rule::stmt_list {
                continue;
            }

            for stmt in stmt_pair.into_inner() {
                match stmt.as_rule() {
                    Rule::node_stmt => {
                        let mut parts = stmt.into_inner();
                        let name = parts.next().unwrap().as_str().to_string();
                        let py_node = PyString::new(py, &name).into_py(py);
                        let idx = if is_directed {
                            di_graph.as_mut().unwrap().graph.add_node(py_node)
                        } else {
                            undi_graph.as_mut().unwrap().graph.add_node(py_node)
                        };
                        node_map.insert(name.clone(), idx);

                        if let Some(attr_list) = parts.next() {
                            let _attrs = parse_attrs(py, attr_list.into_inner());
                            // Future: Use or store node attributes here
                        }
                    }

                    Rule::edge_stmt => {
                        let mut parts = stmt.into_inner();
                        let mut endpoints = vec![];

                        // Collect node identifiers
                        while let Some(token) = parts.peek() {
                            match token.as_rule() {
                                Rule::edge_op => { parts.next(); }
                                Rule::node_id | Rule::subgraph => {
                                    endpoints.push(parts.next().unwrap());
                                }
                                _ => break,
                            }
                        }

                        // Edge attributes or default empty dict
                        let attrs = parts
                            .next()
                            .map(|p| parse_attrs(py, p.into_inner()))
                            .unwrap_or_else(|| PyDict::new(py).into_py(py));

                        for pair in endpoints.windows(2) {
                            let src = pair[0].as_str().to_string();
                            let dst = pair[1].as_str().to_string();

                            let src_idx = *node_map.entry(src.clone()).or_insert_with(|| {
                                let py_node = PyString::new(py, &src).into_py(py);
                                if is_directed {
                                    di_graph.as_mut().unwrap().graph.add_node(py_node)
                                } else {
                                    undi_graph.as_mut().unwrap().graph.add_node(py_node)
                                }
                            });

                            let dst_idx = *node_map.entry(dst.clone()).or_insert_with(|| {
                                let py_node = PyString::new(py, &dst).into_py(py);
                                if is_directed {
                                    di_graph.as_mut().unwrap().graph.add_node(py_node)
                                } else {
                                    undi_graph.as_mut().unwrap().graph.add_node(py_node)
                                }
                            });

                            if is_directed {
                                di_graph.as_mut().unwrap().graph.add_edge(src_idx, dst_idx, attrs.clone_ref(py));
                            } else {
                                undi_graph.as_mut().unwrap().graph.add_edge(src_idx, dst_idx, attrs.clone_ref(py));
                            }
                        }
                    }

                    Rule::attr_stmt => {
                        let mut parts = stmt.into_inner();
                        let target = parts.next().unwrap().as_str();
                        let attrs = parse_attrs(py, parts.flat_map(|p| p.into_inner()));
                        if target == "graph" {
                            graph_attrs = attrs;
                        }
                    }

                    Rule::assignment => {
                        let mut parts = stmt.into_inner();
                        let key = parts.next().unwrap().as_str();
                        let val = parts.next().unwrap().as_str();

                        let any = graph_attrs.bind(py);                         // Returns Bound<'_, PyAny>
                        let dict = any.downcast::<PyDict>()?;                   // Downcast to PyDict

                        dict.set_item(key, val)?;                               // Set key-value in dict
                    }


                    Rule::subgraph => {
                        // TODO: Add subgraph handling if needed
                    }

                    _ => {}
                }
            }
        }
    }

    if is_directed {
        Ok(di_graph.unwrap().into_py(py))
    } else {
        Ok(undi_graph.unwrap().into_py(py))
    }
}

fn parse_attrs<'a>(
    py: Python<'a>,
    pairs: impl Iterator<Item = pest::iterators::Pair<'a, Rule>>,
) -> PyObject {
    let dict = PyDict::new(py);
    for pair in pairs {
        let mut kv = pair.into_inner();
        if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
            dict.set_item(k.as_str(), v.as_str()).unwrap();
        }
    }
    dict.into_py(py)
}
