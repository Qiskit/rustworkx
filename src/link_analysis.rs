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
use ndarray_stats::{DeviationExt, QuantileExt};
use petgraph::prelude::*;
use petgraph::visit::IntoEdgeReferences;
use petgraph::visit::NodeIndexable;
use rustworkx_core::dictmap::*;
use sprs::{CsMat, TriMat};

/// Computes the PageRank of the nodes in a :class:`~PyDiGraph`.
///
/// For details on the PageRank, refer to:
///
/// L. Page, S. Brin, R. Motwani, and T. Winograd. “The PageRank Citation Ranking: Bringing order to the Web”.
/// Stanford Digital Library Technologies Project, (1998).
/// <http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf>
///
/// This function uses a power iteration method to compute the PageRank
/// and convergence is not guaranteed. The function will stop when `max_iter`
/// iterations is reached or when the computed vector between two iterations
/// is smaller than the error tolerance multiplied by the number of nodes.
/// The implementation of this algorithm tries to match NetworkX's
/// `pagerank() <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html>`__
/// implementation.
///
/// In the case of multigraphs the weights of any parallel edges will be
/// summed when computing the PageRank.
///
/// :param PyDiGraph graph: The graph object to run the algorithm on
/// :param float alpha: Damping parameter for PageRank, default=0.85.
/// :param weight_fn: An optional input callable that will be passed the edge's
///     payload object and is expected to return a `float` weight for that edge.
///     If this is not specified ``default_weight`` will be used as the weight
///     for every edge in ``graph``
/// :param dict nstart: Optional starting value of PageRank iteration for each node.
/// :param dict personalization: An optional dictionary representing the personalization
///     vector for a subset of nodes. At least one personalization entry must be non-zero.
///     If not specified, a nodes personalization value will be zero. By default,
///     a uniform distribution is used.
/// :param float tol: The error tolerance used when checking for convergence in the
///     power method. If this is not specified default value of 1e-6 is used.
/// :param int max_iter: The maximum number of iterations in the power method. If
///     not specified a default value of 100 is used.
/// :param dict dangling: An optional dictionary for the outedges to be assigned to any "dangling" nodes,
///     i.e., nodes without any outedges.  The dict key is the node the outedge points to and the dict
///     value is the weight of that outedge. By default, dangling nodes are given outedges according to
///     the personalization vector (uniform if not specified). This must be selected to result in an irreducible
///     transition matrix. It may be common to have the dangling dict to be the same as the personalization dict.
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      PageRank score for that node.
/// :rtype: CentralityMapping
#[pyfunction(
    signature = (
        graph,
        alpha=0.85,
        weight_fn=None,
        nstart=None,
        personalization=None,
        tol=1e-6,
        max_iter=100,
        dangling=None,
    )
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
    let default_pop = (n as f64).recip();

    // Handle custom start
    if let Some(nstart) = nstart {
        for i in &node_indices {
            popularity[*i] = *nstart.get(i).unwrap_or(&0.0);
        }
        let pop_sum = popularity.sum();
        popularity /= pop_sum;
    } else {
        for i in &node_indices {
            popularity[*i] = default_pop;
        }
    }

    // Handle personalization
    let personalized_array: Array1<f64> = match personalization {
        Some(personalization) => {
            let mut personalized_array = Array1::<f64>::zeros(mat_size);
            for i in &node_indices {
                personalized_array[*i] = *personalization.get(i).unwrap_or(&0.0);
            }
            let p_sum = personalized_array.sum();
            personalized_array /= p_sum;
            personalized_array
        }
        None => {
            let mut personalized_array = Array1::<f64>::zeros(mat_size);
            for i in &node_indices {
                personalized_array[*i] = default_pop;
            }
            personalized_array
        }
    };
    let damping = (1.0 - alpha) * &personalized_array;

    // Handle dangling nodes i.e. nodes that point nowhere
    let is_dangling = (0..mat_size)
        .map(|i| out_weights[i] == 0.0)
        .collect::<Vec<_>>();
    let dangling_weights: Array1<f64> = match dangling {
        Some(dangling) => {
            let mut dangling_weights = Array1::<f64>::zeros(mat_size);
            for i in &node_indices {
                dangling_weights[*i] = *dangling.get(i).unwrap_or(&0.0);
            }
            let d_sum = dangling_weights.sum();
            dangling_weights /= d_sum;
            dangling_weights
        }
        None => personalized_array,
    };

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

/// Computes the hubs and authorities in a :class:`~PyDiGraph`.
///
/// For details on the HITS algorithm, refer to:
///
/// J.  Kleinberg. “Authoritative Sources in a Hyperlinked Environment”.
/// Journal of the ACM, 46 (5), (1999).
/// <http://www.cs.cornell.edu/home/kleinber/auth.pdf>
///
/// This function uses a power iteration method to compute the hubs and authorities
/// and convergence is not guaranteed. The function will stop when `max_iter`
/// iterations is reached or when the computed vector between two iterations
/// is smaller than the error tolerance multiplied by the number of nodes.
///
/// In the case of multigraphs the weights of any parallel edges will be
/// summed when computing the hubs and authorities.
///
/// :param PyDiGraph graph: The graph object to run the algorithm on
/// :param weight_fn: An optional input callable that will be passed the edge's
///     payload object and is expected to return a `float` weight for that edge.
///     If this is not specified 1.0 will be used as the weight
///     for every edge in ``graph``
/// :param dict nstart: Optional starting value for the power iteration for each node.
/// :param float tol: The error tolerance used when checking for convergence in the
///     power method. If this is not specified default value of 1e-8 is used.
/// :param int max_iter: The maximum number of iterations in the power method. If
///     not specified a default value of 100 is used.
/// :param boolean normalized: If the scores should be normalized (defaults to True).
///
/// :returns: a tuple of read-only dict-like object whose keys are the node indices. The first value in the tuple
///      contain the hubs scores. The second value contains the authority scores.
/// :rtype: tuple[CentralityMapping, CentralityMapping]
#[pyfunction(
    signature = (
        graph,
        weight_fn=None,
        nstart=None,
        tol=1e-6,
        max_iter=100,
        normalized=true,
    )
)]
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, nstart=None, tol=1.0e-8, max_iter=100, normalized=True)"
)]
pub fn hits(
    py: Python,
    graph: &PyDiGraph,
    weight_fn: Option<PyObject>,
    nstart: Option<HashMap<usize, f64>>,
    tol: f64,
    max_iter: usize,
    normalized: bool,
) -> PyResult<(CentralityMapping, CentralityMapping)> {
    // we use the node bound to make the code work if nodes were removed
    let n = graph.graph.node_count();
    let mat_size = graph.graph.node_bound();
    let node_indices: Vec<usize> = graph.graph.node_indices().map(|x| x.index()).collect();

    // Handle empty case
    if n == 0 {
        return Ok((
            CentralityMapping {
                centralities: DictMap::new(),
            },
            CentralityMapping {
                centralities: DictMap::new(),
            },
        ));
    }

    // Grab the graph weights from Python to Rust
    let mut adjacent: HashMap<(usize, usize), f64> =
        HashMap::with_capacity(graph.graph.edge_count());
    let default_weight: f64 = 1.0;

    for edge in graph.graph.edge_references() {
        let i = NodeIndexable::to_index(&graph.graph, edge.source());
        let j = NodeIndexable::to_index(&graph.graph, edge.target());
        let weight = edge.weight().clone_ref(py);

        let edge_weight = weight_callable(py, &weight_fn, &weight, default_weight)?;

        *adjacent.entry((i, j)).or_insert(0.0) += edge_weight;
    }

    // Create sparse adjacency matrix and transpose
    let mut a = TriMat::new((mat_size, mat_size));
    let mut a_t = TriMat::new((mat_size, mat_size));
    for ((i, j), weight) in adjacent.into_iter() {
        a.add_triplet(i, j, weight);
        a_t.add_triplet(j, i, weight);
    }
    let a: CsMat<_> = a.to_csr();
    let a_t: CsMat<_> = a_t.to_csr();

    // Initial guess of eigenvector of A^T @ A
    let mut authority = Array1::<f64>::zeros(mat_size);
    let default_auth = (n as f64).recip();

    // Handle custom start
    if let Some(nstart) = nstart {
        for i in &node_indices {
            authority[*i] = *nstart.get(i).unwrap_or(&0.0);
        }
        let a_sum = authority.sum();
        authority /= a_sum;
    } else {
        for i in &node_indices {
            authority[*i] = default_auth;
        }
    }

    // Power Method iteration for A^T @ A
    let mut has_converged = false;
    for _ in 0..max_iter {
        // Instead of evaluating A^T @ A, which might not be sparse
        // we prefer to calculate A^T (A @ x); A @ x is a vector hence
        // we don't have to worry about sparsity
        let temp_hub = &a * &authority;
        let mut new_authority = &a_t * &temp_hub;
        new_authority /= *new_authority.max_skipnan();
        let norm: f64 = new_authority.l1_dist(&authority).unwrap();
        if norm < tol {
            has_converged = true;
            break;
        } else {
            authority = new_authority;
        }
    }

    // Convert to custom return type
    if !has_converged {
        return Err(FailedToConverge::new_err(format!(
            "Function failed to converge on a solution in {} iterations",
            max_iter
        )));
    }

    let mut hubs = &a * &authority;

    if normalized {
        hubs /= hubs.sum();
        authority /= authority.sum();
    }

    let hubs_map: DictMap<usize, f64> = node_indices.iter().map(|x| (*x, hubs[*x])).collect();
    let auth_map: DictMap<usize, f64> = node_indices.iter().map(|x| (*x, authority[*x])).collect();

    Ok((
        CentralityMapping {
            centralities: hubs_map,
        },
        CentralityMapping {
            centralities: auth_map,
        },
    ))
}
