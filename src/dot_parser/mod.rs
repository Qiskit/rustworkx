use pest::Parser;
use pest_derive::Parser;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::StablePyGraph;
use crate::digraph::PyDiGraph;
use crate::graph::PyGraph;
// use crate::dot_parser::Rule;

use rustworkx_core::petgraph::Directed;
use rustworkx_core::petgraph::Undirected;

#[derive(Parser)]
#[grammar = "dot_parser/dot.pest"]
pub struct DotParser;

#[pyfunction]
pub fn from_dot(py: Python<'_>, dot_str: &str) -> PyResult<PyObject> {
    let pairs = DotParser::parse(Rule::graph_file, dot_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("DOT parse error: {}", e)))?;

    let mut is_directed = false;
    let mut node_map = std::collections::HashMap::new();

    // We'll populate these after we know graph type
    let mut di_graph: Option<PyDiGraph> = None;
    let mut undi_graph: Option<PyGraph> = None;

    for pair in pairs {
        if pair.as_rule() != Rule::graph_file {
            continue;
        }
        let mut inner = pair.into_inner();
        let gtype = inner.next().unwrap().as_str();
        is_directed = gtype == "digraph";

        if is_directed {
            // Create a directed StablePyGraph with some initial capacity
            let mut graph = StablePyGraph::<Directed>::with_capacity(0, 0);
            di_graph = Some(PyDiGraph {
                graph,
                cycle_state: rustworkx_core::petgraph::algo::DfsSpace::default(),
                check_cycle: false,
                node_removed: false,
                multigraph: true,
                attrs: py.None()
            });
        } else {
            // Create an undirected StablePyGraph with some initial capacity
            let mut graph = StablePyGraph::<Undirected>::with_capacity(0, 0);
            undi_graph = Some(PyGraph {
                graph,
                node_removed: false,
                multigraph: true,
                attrs: py.None()
            });
        }

        // Now parse statements
        for stmt in inner {
            if stmt.as_rule() != Rule::stmt_list {
                continue;
            }
            for s in stmt.into_inner() {
                match s.as_rule() {
                    Rule::node_stmt => {
                        let mut tokens = s.into_inner();
                        let name = tokens.next().unwrap().as_str().to_string();
                        let py_node = PyString::new(py, &name).to_object(py);

                        let idx = if is_directed {
                            di_graph.as_mut().unwrap().graph.add_node(py_node)
                        } else {
                            undi_graph.as_mut().unwrap().graph.add_node(py_node)
                        };
                        node_map.insert(name, idx);
                    }
                    Rule::edge_stmt => {
                        let mut tokens = s.into_inner();
                        let source = tokens.next().unwrap().as_str().to_string();
                        let _arrow = tokens.next().unwrap();
                        let target = tokens.next().unwrap().as_str().to_string();

                        let source_idx = *node_map.entry(source.clone()).or_insert_with(|| {
                            let py_node = PyString::new(py, &source).to_object(py);
                            if is_directed {
                                di_graph.as_mut().unwrap().graph.add_node(py_node)
                            } else {
                                undi_graph.as_mut().unwrap().graph.add_node(py_node)
                            }
                        });
                        let target_idx = *node_map.entry(target.clone()).or_insert_with(|| {
                            let py_node = PyString::new(py, &target).to_object(py);
                            if is_directed {
                                di_graph.as_mut().unwrap().graph.add_node(py_node)
                            } else {
                                undi_graph.as_mut().unwrap().graph.add_node(py_node)
                            }
                        });

                        let edge_weight = PyString::new(py, "1").to_object(py);
                        if is_directed {
                            di_graph.as_mut().unwrap().graph.add_edge(source_idx, target_idx, edge_weight);
                        } else {
                            undi_graph.as_mut().unwrap().graph.add_edge(source_idx, target_idx, edge_weight);
                        }
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
