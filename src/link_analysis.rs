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

use pyo3::prelude::*;
use pyo3::Python;

use crate::digraph::PyDiGraph;
use crate::iterators::CentralityMapping;
use crate::FailedToConverge;

use hashbrown::HashMap;
use ndarray::prelude::*;
use ndarray_stats::DeviationExt;
use petgraph::prelude::*;
use petgraph::visit::NodeCount;
use petgraph::visit::NodeIndexable;
use rustworkx_core::dictmap::*;
use sprs::{CsMat, TriMat};

#[pyfunction]
#[pyo3(
    text_signature = "(graph, alpha=0.85, weight_fn=None, personalization=None, tol=1e-6, max_iter=100 /)"
)]
pub fn pagerank(
    py: Python,
    graph: &PyDiGraph,
    alpha: f64,
    weight_fn: PyObject,
    personalization: Option<HashMap<usize, f64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<CentralityMapping> {
    // we use the node bound to make the code work if nodes were removed
    let n = graph.graph.node_count();
    let mat_size = graph.graph.node_bound();

    // Create sparse Google Matrix that describes the Markov Chain process
    let mut a = TriMat::new((mat_size, mat_size));
    a.add_triplet(0, 0, 3.0_f64);
    //a.add_triplet(1, 2, 2.0);
    //a.add_triplet(3, 0, -2.0);
    let a: CsMat<_> = a.to_csr();

    // Vector with probabilities
    // TODO: loop over nodes
    let mut popularity = Array1::<f64>::zeros(mat_size);
    let mut damping = Array1::<f64>::zeros(mat_size);
    let default_pop = (n as f64).recip();
    let default_damp = (1.0 - alpha) * default_pop;

    for node_index in graph.graph.node_indices() {
        let i = node_index.index();
        popularity[i] = default_pop;
        damping[i] = default_damp;
    }

    // Power Method iteration for the Google Matrix
    let mut has_converged = false;
    for _ in 0..max_iter {
        let new_popularity = alpha * (&a * &popularity) + &damping;
        let norm: f64 = new_popularity.l1_dist(&popularity).unwrap();
        if norm < (n as f64) * tol {
            has_converged = true;
            break;
        } else {
            popularity = new_popularity;
        }
    }

    // Convert to custom return type
    if !has_converged {
        return Err(FailedToConverge::new_err(format!(
            "Function failed to converge on a solution in {} iterations",
            max_iter
        )));
    }

    let out_map: DictMap<usize, f64> = graph
        .graph
        .node_indices()
        .map(|x| (x.index(), popularity[x.index()]))
        .collect();

    Ok(CentralityMapping {
        centralities: out_map,
    })
}
