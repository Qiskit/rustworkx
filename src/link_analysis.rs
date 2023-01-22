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
use hashbrown::HashMap;
use indexmap::IndexMap;
use ndarray::prelude::*;
use sprs::{CsMat, TriMat};

#[pyfunction]
#[pyo3(text_signature = "(graph, alpha=0.85, weight_fn=None, personalization=None, tol=1e-6, max_iter=100 /)")]
pub fn pagerank(
    graph: &graph::PyDiGraph,
    alpha: f64,
    weight_fn: PyObject,
    personalization: Option<HashMap<usize, f64>>,
    tol: f64,
    max_iter: usize
) -> PyResult<IndexMap<usize,f64>> {
    let n = graph.node_count();
    // we use the node bound to make the code work if nodes were removed
    let mat_size = graph.graph.node_bound();

    // Create sprase matrix
    let mut a = TriMat::new((mat_size, mat_size));
    //a.add_triplet(0, 0, 3.0_f64);
    //a.add_triplet(1, 2, 2.0);
    //a.add_triplet(3, 0, -2.0);
    let a: CsMat<_> = a.to_csr();

    // Vector with probabilities
    // TODO: loop over nodes
    let mut popularity = Array1<f64>::zeros(mat_size);
    let mut damping = Array1<f64>::zeros(mat_size);

    for _ in 0..max_iter {
        let mut new_popularity = alpha * (a * &popularity) + damping;
        // TODO calculate norms
        let norm;
        if norm < tol {

        }
        else {

        }
    }

    // TODO: loop to put probs into an array

}