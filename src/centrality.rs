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

use std::convert::TryFrom;

use crate::digraph;
use crate::graph;
use crate::iterators::CentralityMapping;
use crate::CostFn;
use crate::FailedToConverge;

use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeIndexable;
use petgraph::visit::EdgeRef;
use pyo3::prelude::*;
use rustworkx_core::centrality;

/// Compute the betweenness centrality of all nodes in a PyGraph.
///
/// Betweenness centrality of a node :math:`v` is the sum of the
/// fraction of all-pairs shortest paths that pass through :math`v`
///
/// .. math::
///
///    c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}
///
/// where :math:`V` is the set of nodes, :math:`\sigma(s, t)` is the number of
/// shortest :math:`(s, t)` paths, and :math:`\sigma(s, t|v)` is the number of
/// those paths  passing through some  node :math:`v` other than :math:`s, t`.
/// If :math:`s = t`, :math:`\sigma(s, t) = 1`, and if :math:`v \in {s, t}`,
/// :math:`\sigma(s, t|v) = 0`
///
/// The algorithm used in this function is based on:
///
/// Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
/// Journal of Mathematical Sociology 25(2):163-177, 2001.
///
/// This function is multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold`` (it
/// defaults to 50). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// :param PyGraph graph: The input graph
/// :param bool normalized: Whether to normalize the betweenness scores by the number of distinct
///    paths between all pairs of nodes.
/// :param bool endpoints: Whether to include the endpoints of paths in pathlengths used to
///    compute the betweenness.
/// :param int parallel_threshold: The number of nodes to calculate the
///     the betweenness centrality in parallel at if the number of nodes in
///     the graph is less than this value it will run in a single thread. The
///     default value is 50
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      betweenness score for each node.
/// :rtype: CentralityMapping
#[pyfunction(normalized = "true", endpoints = "false", parallel_threshold = "50")]
#[pyo3(text_signature = "(graph, /, normalized=True, endpoints=False, parallel_threshold=50)")]
pub fn graph_betweenness_centrality(
    graph: &graph::PyGraph,
    normalized: bool,
    endpoints: bool,
    parallel_threshold: usize,
) -> CentralityMapping {
    let betweenness =
        centrality::betweenness_centrality(&graph.graph, endpoints, normalized, parallel_threshold);
    CentralityMapping {
        centralities: betweenness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    }
}

/// Compute the betweenness centrality of all nodes in a PyDiGraph.
///
/// Betweenness centrality of a node :math:`v` is the sum of the
/// fraction of all-pairs shortest paths that pass through :math`v`
///
/// .. math::
///
///    c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}
///
/// where :math:`V` is the set of nodes, :math:`\sigma(s, t)` is the number of
/// shortest :math`(s, t)` paths, and :math:`\sigma(s, t|v)` is the number of
/// those paths  passing through some  node :math:`v` other than :math:`s, t`.
/// If :math:`s = t`, :math:`\sigma(s, t) = 1`, and if :math:`v \in {s, t}`,
/// :math:`\sigma(s, t|v) = 0`
///
/// The algorithm used in this function is based on:
///
/// Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
/// Journal of Mathematical Sociology 25(2):163-177, 2001.
///
/// This function is multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold`` (it
/// defaults to 50). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// :param PyDiGraph graph: The input graph
/// :param bool normalized: Whether to normalize the betweenness scores by the number of distinct
///    paths between all pairs of nodes.
/// :param bool endpoints: Whether to include the endpoints of paths in pathlengths used to
///    compute the betweenness.
/// :param int parallel_threshold: The number of nodes to calculate the
///     the betweenness centrality in parallel at if the number of nodes in
///     the graph is less than this value it will run in a single thread. The
///     default value is 50
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      betweenness score for each node.
/// :rtype: CentralityMapping
#[pyfunction(normalized = "true", endpoints = "false", parallel_threshold = "50")]
#[pyo3(text_signature = "(graph, /, normalized=True, endpoints=False, parallel_threshold=50)")]
pub fn digraph_betweenness_centrality(
    graph: &digraph::PyDiGraph,
    normalized: bool,
    endpoints: bool,
    parallel_threshold: usize,
) -> CentralityMapping {
    let betweenness =
        centrality::betweenness_centrality(&graph.graph, endpoints, normalized, parallel_threshold);
    CentralityMapping {
        centralities: betweenness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    }
}

/// Compute the eigenvector centrality of a :class:`~PyGraph`.
///
/// For details on the eigenvector centrality refer to:
///
/// Phillip Bonacich. “Power and Centrality: A Family of Measures.”
/// American Journal of Sociology 92(5):1170–1182, 1986
/// <https://doi.org/10.1086/228631>
///
/// This function uses a power iteration method to compute the eigenvector
/// and convergence is not guaranteed. The function will stop when `max_iter`
/// iterations is reached or when the computed vector between two iterations
/// is smaller than the error tolerance multiplied by the number of nodes.
/// The implementation of this algorithm is based on the NetworkX
/// `eigenvector_centrality() <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.eigenvector_centrality.html>`__
/// function.
///
/// In the case of multigraphs the weights of any parallel edges will be
/// summed when computing the eigenvector centrality.
///
/// :param PyGraph graph: The graph object to run the algorithm on
/// :param weight_fn: An optional input callable that will be passed the edge's
///     payload object and is expected to return a `float` weight for that edge.
///     If this is not specified ``default_weight`` will be used as the weight
///     for every edge in ``graph``
/// :param float default_weight: If ``weight_fn`` is not set the default weight
///     value to use for the weight of all edges
/// :param int max_iter: The maximum number of iterations in the power method. If
///     not specified a default value of 100 is used.
/// :param float tol: The error tolerance used when checking for convergence in the
///     power method. If this is not specified default value of 1e-6 is used.
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      centrality score for that node.
/// :rtype: CentralityMapping
#[pyfunction(default_weight = "1.0", max_iter = "100", tol = "1e-6")]
#[pyo3(text_signature = "(graph, /, weight_fn=None, default_weight=1.0, max_iter=100, tol=1e-6)")]
pub fn graph_eigenvector_centrality(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<CentralityMapping> {
    let mut edge_weights = vec![default_weight; graph.graph.edge_bound()];
    if weight_fn.is_some() {
        let cost_fn = CostFn::try_from((weight_fn, default_weight))?;
        for edge in graph.graph.edge_indices() {
            edge_weights[edge.index()] =
                cost_fn.call(py, graph.graph.edge_weight(edge).unwrap())?;
        }
    }
    let ev_centrality = centrality::eigenvector_centrality(
        &graph.graph,
        |e| -> PyResult<f64> { Ok(edge_weights[e.id().index()]) },
        Some(max_iter),
        Some(tol),
    )?;
    match ev_centrality {
        Some(centrality) => Ok(CentralityMapping {
            centralities: centrality
                .iter()
                .enumerate()
                .filter_map(|(k, v)| {
                    if graph.graph.contains_node(NodeIndex::new(k)) {
                        Some((k, *v))
                    } else {
                        None
                    }
                })
                .collect(),
        }),
        None => Err(FailedToConverge::new_err(format!(
            "Function failed to converge on a solution in {} iterations",
            max_iter
        ))),
    }
}

/// Compute the eigenvector centrality of a :class:`~PyDiGraph`.
///
/// For details on the eigenvector centrality refer to:
///
/// Phillip Bonacich. “Power and Centrality: A Family of Measures.”
/// American Journal of Sociology 92(5):1170–1182, 1986
/// <https://doi.org/10.1086/228631>
///
/// This function uses a power iteration method to compute the eigenvector
/// and convergence is not guaranteed. The function will stop when `max_iter`
/// iterations is reached or when the computed vector between two iterations
/// is smaller than the error tolerance multiplied by the number of nodes.
/// The implementation of this algorithm is based on the NetworkX
/// `eigenvector_centrality() <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.eigenvector_centrality.html>`__
/// function.
///
/// In the case of multigraphs the weights of any parallel edges will be
/// summed when computing the eigenvector centrality.
///
/// :param PyDiGraph graph: The graph object to run the algorithm on
/// :param weight_fn: An optional input callable that will be passed the edge's
///     payload object and is expected to return a `float` weight for that edge.
///     If this is not specified ``default_weight`` will be used as the weight
///     for every edge in ``graph``
/// :param float default_weight: If ``weight_fn`` is not set the default weight
///     value to use for the weight of all edges
/// :param int max_iter: The maximum number of iterations in the power method. If
///     not specified a default value of 100 is used.
/// :param float tol: The error tolerance used when checking for convergence in the
///     power method. If this is not specified default value of 1e-6 is used.
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      centrality score for that node.
/// :rtype: CentralityMapping
#[pyfunction(default_weight = "1.0", max_iter = "100", tol = "1e-6")]
#[pyo3(text_signature = "(graph, /, weight_fn=None, default_weight=1.0, max_iter=100, tol=1e-6)")]
pub fn digraph_eigenvector_centrality(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<CentralityMapping> {
    let mut edge_weights = vec![default_weight; graph.graph.edge_bound()];
    if weight_fn.is_some() {
        let cost_fn = CostFn::try_from((weight_fn, default_weight))?;
        for edge in graph.graph.edge_indices() {
            edge_weights[edge.index()] =
                cost_fn.call(py, graph.graph.edge_weight(edge).unwrap())?;
        }
    }
    let ev_centrality = centrality::eigenvector_centrality(
        &graph.graph,
        |e| -> PyResult<f64> { Ok(edge_weights[e.id().index()]) },
        Some(max_iter),
        Some(tol),
    )?;

    match ev_centrality {
        Some(centrality) => Ok(CentralityMapping {
            centralities: centrality
                .iter()
                .enumerate()
                .filter_map(|(k, v)| {
                    if graph.graph.contains_node(NodeIndex::new(k)) {
                        Some((k, *v))
                    } else {
                        None
                    }
                })
                .collect(),
        }),
        None => Err(FailedToConverge::new_err(format!(
            "Function failed to converge on a solution in {} iterations",
            max_iter
        ))),
    }
}
