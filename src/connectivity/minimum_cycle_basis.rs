use rustworkx_core::connectivity::minimal_cycle_basis;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::EdgeIndexable;
use petgraph::EdgeType;

use crate::{CostFn, StablePyGraph};

pub fn minimum_cycle_basis_map<Ty: EdgeType + Sync>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    edge_cost_fn: PyObject,
) -> PyResult<Vec<Vec<NodeIndex>>> {
    if graph.node_count() == 0 {
        return Ok(vec![]);
    } else if graph.edge_count() == 0 {
        return Ok(vec![]);
    }
    let edge_cost_callable = CostFn::from(edge_cost_fn);
    let mut edge_weights: Vec<Option<f64>> = Vec::with_capacity(graph.edge_bound());
    for index in 0..=graph.edge_bound() {
        let raw_weight = graph.edge_weight(EdgeIndex::new(index));
        match raw_weight {
            Some(weight) => edge_weights.push(Some(edge_cost_callable.call(py, weight)?)),
            None => edge_weights.push(None),
        };
    }
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };
    let cycle_basis = minimal_cycle_basis(graph, |e| edge_cost(e.id())).unwrap();
    Ok(cycle_basis)
}
