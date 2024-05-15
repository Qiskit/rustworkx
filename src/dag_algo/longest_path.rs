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

use crate::{digraph, DAGHasCycle};
use rustworkx_core::dag_algo::longest_path as core_longest_path;

use petgraph::stable_graph::EdgeReference;
use petgraph::visit::EdgeRef;

use pyo3::prelude::*;

use num_traits::{Num, Zero};

/// Calculate the longest path in a directed acyclic graph (DAG).
///
/// This function interfaces with the Python `PyDiGraph` object to compute the longest path
/// using the provided weight function.
///
/// # Arguments
/// * `graph`: Reference to a `PyDiGraph` object.
/// * `weight_fn`: A callable that takes the source node index, target node index, and the weight
///   object and returns the weight of the edge as a `PyResult<T>`.
///
/// # Type Parameters
/// * `F`: Type of the weight function.
/// * `T`: The type of the edge weight. Must implement `Num`, `Zero`, `PartialOrd`, and `Copy`.
///
/// # Returns
/// * `PyResult<(Vec<usize>, T)>` representing the longest path as a sequence of node indices and its total weight.
pub fn longest_path<F, T>(graph: &digraph::PyDiGraph, mut weight_fn: F) -> PyResult<(Vec<usize>, T)>
where
    F: FnMut(usize, usize, &PyObject) -> PyResult<T>,
    T: Num + Zero + PartialOrd + Copy,
{
    let dag = &graph.graph;

    // Create a new weight function that matches the required signature
    let edge_cost = |edge_ref: EdgeReference<'_, PyObject>| -> Result<T, PyErr> {
        let source = edge_ref.source().index();
        let target = edge_ref.target().index();
        let weight = edge_ref.weight();
        weight_fn(source, target, weight)
    };

    let (path, path_weight) = match core_longest_path(dag, edge_cost) {
        Ok(Some(result)) => result,
        Ok(None) => return Err(DAGHasCycle::new_err("The graph contains a cycle")),
        Err(e) => return Err(e),
    };

    Ok((path, path_weight))
}
