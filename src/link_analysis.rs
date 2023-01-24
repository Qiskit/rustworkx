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

// PageRank has many possible personalizations, so we accept them all
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use pyo3::Python;

use crate::digraph::PyDiGraph;
use crate::iterators::CentralityMapping;
use crate::{weight_callable, FailedToConverge};

use hashbrown::HashMap;
use ndarray::prelude::*;
use ndarray_stats::DeviationExt;
use petgraph::prelude::*;
use petgraph::visit::IntoEdgeReferences;
use petgraph::visit::NodeIndexable;
use rustworkx_core::dictmap::*;
use sprs::{CsMat, TriMat};

#[pyfunction(
    alpha = "0.85",
    weight_fn = "None",
    nstart = "None",
    personalization = "None",
    tol = "1.0e-6",
    max_iter = "100",
    dangling = "None"
)]
#[pyo3(
    text_signature = "(graph, /, alpha=0.85, weight_fn=None, nstart=None, personalization=None, tol=1.0e-6, max_iter=100)"
)]
pub fn pagerank(
    py: Python,
    graph: &PyDiGraph,
    alpha: f64,
    weight_fn: Option<PyObject>,
    nstart: Option<HashMap<usize, f64>>,
    personalization: Option<HashMap<usize, f64>>,
    tol: f64,
    max_iter: usize,
    dangling: Option<HashMap<usize, f64>>,
) -> PyResult<CentralityMapping> {
    // we use the node bound to make the code work if nodes were removed
    let n = graph.graph.node_count();
    let mat_size = graph.graph.node_bound();
    let node_indices: Vec<usize> = graph.graph.node_indices().map(|x| x.index()).collect();

    // Handle empty case
    if n == 0 {
        return Ok(CentralityMapping {
            centralities: DictMap::new(),
        });
    }

    // Grab the graph weights from Python to Rust
    let mut in_weights: HashMap<(usize, usize), f64> =
        HashMap::with_capacity(graph.graph.edge_count());
    let mut out_weights: Vec<f64> = vec![0.0; mat_size];
    let default_weight: f64 = 1.0;

    for edge in graph.graph.edge_references() {
        let i = NodeIndexable::to_index(&graph.graph, edge.source());
        let j = NodeIndexable::to_index(&graph.graph, edge.target());
        let weight = edge.weight().clone();

        let edge_weight = weight_callable(py, &weight_fn, &weight, default_weight)?;
        out_weights[i] += edge_weight;
        *in_weights.entry((i, j)).or_insert(0.0) += edge_weight;
    }

    // Create sparse Google Matrix that describes the Markov Chain process
    let mut a = TriMat::new((mat_size, mat_size));
    for ((i, j), weight) in in_weights.into_iter() {
        a.add_triplet(j, i, weight / out_weights[i]);
    }
    let a: CsMat<_> = a.to_csr();

    // Vector with probabilities for the Markov Chain process
    let mut popularity = Array1::<f64>::zeros(mat_size);
    let mut personalized_array = Array1::<f64>::zeros(mat_size);
    let default_pop = (n as f64).recip();

    // Handle custom start
    if let Some(nstart) = nstart {
        for i in &node_indices {
            popularity[*i] = *nstart.get(&i).unwrap_or(&0.0);
        }
        let pop_sum = popularity.sum();
        popularity /= pop_sum;
    } else {
        for i in &node_indices {
            popularity[*i] = default_pop;
        }
    }

    // Handle personalization
    if let Some(personalization) = personalization {
        for i in &node_indices {
            personalized_array[*i] = *personalization.get(&i).unwrap_or(&0.0);
        }
        let p_sum = personalized_array.sum();
        personalized_array /= p_sum;
    } else {
        for i in &node_indices {
            personalized_array[*i] = default_pop;
        }
    }
    let damping = (1.0 - alpha) * &personalized_array;

    // Handle dangling nodes i.e. nodes that point nowhere
    let mut is_dangling: Vec<bool> = vec![false; mat_size];
    let mut dangling_weights = Array1::<f64>::zeros(mat_size);

    for i in &node_indices {
        if out_weights[*i] == 0.0 {
            is_dangling[*i] = true;
        }
    }

    if let Some(dangling) = dangling {
        for i in &node_indices {
            dangling_weights[*i] = *dangling.get(&i).unwrap_or(&0.0);
        }
        let d_sum = dangling_weights.sum();
        dangling_weights /= d_sum;
    } else {
        dangling_weights = personalized_array.clone();
    }

    // Power Method iteration for the Google Matrix
    let mut has_converged = false;
    for _ in 0..max_iter {
        let dangling_sum: f64 = is_dangling
            .iter()
            .zip(popularity.iter())
            .map(|(cond, pop)| if *cond { *pop } else { 0.0 })
            .sum();
        let new_popularity =
            alpha * ((&a * &popularity) + (dangling_sum * &dangling_weights)) + &damping;
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
