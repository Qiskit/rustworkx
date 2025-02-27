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

#![allow(clippy::too_many_arguments)]

use std::convert::TryFrom;

use crate::digraph;
use crate::graph;
use crate::iterators::{CentralityMapping, EdgeCentralityMapping};
use crate::CostFn;
use crate::FailedToConverge;

use hashbrown::HashMap;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeIndexable;
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoNodeIdentifiers;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustworkx_core::centrality;

/// Compute the betweenness centrality of all nodes in a PyGraph.
///
/// Betweenness centrality of a node :math:`v` is the sum of the
/// fraction of all-pairs shortest paths that pass through :math:`v`
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
/// See Also
/// --------
/// :func:`~rustworkx.graph_edge_betweenness_centrality`
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
#[pyfunction(
    signature = (
        graph,
        normalized=true,
        endpoints=false,
        parallel_threshold=50
    )
)]
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
/// fraction of all-pairs shortest paths that pass through :math:`v`
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
/// See Also
/// --------
/// :func:`~rustworkx.digraph_edge_betweenness_centrality`
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
#[pyfunction(
    signature = (
        graph,
        normalized=true,
        endpoints=false,
        parallel_threshold=50
    )
)]
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

/// Compute the degree centrality for nodes in a PyGraph.
///
/// Degree centrality assigns an importance score based simply on the number of edges held by each node.
///
/// :param PyGraph graph: The input graph
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      centrality score for each node.
/// :rtype: CentralityMapping
#[pyfunction(signature = (graph,))]
#[pyo3(text_signature = "(graph, /,)")]
pub fn graph_degree_centrality(graph: &graph::PyGraph) -> PyResult<CentralityMapping> {
    let centrality = centrality::degree_centrality(&graph.graph, None);

    Ok(CentralityMapping {
        centralities: graph
            .graph
            .node_indices()
            .map(|i| (i.index(), centrality[i.index()]))
            .collect(),
    })
}

/// Compute the degree centrality for nodes in a PyDiGraph.
///
/// Degree centrality assigns an importance score based simply on the number of edges held by each node.
/// This function computes the TOTAL (in + out) degree centrality.
///
/// :param PyDiGraph graph: The input graph
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      centrality score for each node.
/// :rtype: CentralityMapping
#[pyfunction(signature = (graph,))]
#[pyo3(text_signature = "(graph, /,)")]
pub fn digraph_degree_centrality(graph: &digraph::PyDiGraph) -> PyResult<CentralityMapping> {
    let centrality = centrality::degree_centrality(&graph.graph, None);

    Ok(CentralityMapping {
        centralities: graph
            .graph
            .node_indices()
            .map(|i| (i.index(), centrality[i.index()]))
            .collect(),
    })
}
/// Compute the in-degree centrality for nodes in a PyDiGraph.
///
/// In-degree centrality assigns an importance score based on the number of incoming edges
/// to each node.
///
/// :param PyDiGraph graph: The input graph
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      centrality score for each node.
/// :rtype: CentralityMapping
#[pyfunction(signature = (graph,))]
#[pyo3(text_signature = "(graph, /)")]
pub fn in_degree_centrality(graph: &digraph::PyDiGraph) -> PyResult<CentralityMapping> {
    let centrality =
        centrality::degree_centrality(&graph.graph, Some(petgraph::Direction::Incoming));

    Ok(CentralityMapping {
        centralities: graph
            .graph
            .node_indices()
            .map(|i| (i.index(), centrality[i.index()]))
            .collect(),
    })
}

/// Compute the out-degree centrality for nodes in a PyDiGraph.
///
/// Out-degree centrality assigns an importance score based on the number of outgoing edges
/// from each node.
///
/// :param PyDiGraph graph: The input graph
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      centrality score for each node.
/// :rtype: CentralityMapping
#[pyfunction(signature = (graph,))]
#[pyo3(text_signature = "(graph, /,)")]
pub fn out_degree_centrality(graph: &digraph::PyDiGraph) -> PyResult<CentralityMapping> {
    let centrality =
        centrality::degree_centrality(&graph.graph, Some(petgraph::Direction::Outgoing));

    Ok(CentralityMapping {
        centralities: graph
            .graph
            .node_indices()
            .map(|i| (i.index(), centrality[i.index()]))
            .collect(),
    })
}

/// Compute the closeness centrality of each node in a :class:`~.PyGraph` object.
///
/// The closeness centrality of a node :math:`u` is defined as the
/// reciprocal of the average shortest path distance to :math:`u` over all
/// :math:`n-1` reachable nodes in the graph. In it's general form this can
/// be expressed as:
///
/// .. math::
///
///     C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},
///
/// where:
///
///   * :math:`d(v, u)` - the shortest-path distance between :math:`v` and
///     :math:`u`
///   * :math:`n` - the number of nodes that can reach :math:`u`.
///
/// In the case of a graphs with more than one connected component there is
/// an alternative improved formula that calculates the closeness centrality
/// as "a ratio of the fraction of actors in the group who are reachable, to
/// the average distance" [WF]_. This can be expressed as
///
/// .. math::
///
///     C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},
///
/// where :math:`N` is the number of nodes in the graph. This alternative
/// formula can be used with the ``wf_improved`` argument.
///
/// :param PyGraph graph: The input graph. Can either be a
///     :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
/// :param bool wf_improved: This is optional; the default is True. If True,
///     scale by the fraction of nodes reachable.
/// :param int parallel_threshold: The number of nodes to calculate the
///     the betweenness centrality in parallel at if the number of nodes in
///     the graph is less than this value it will run in a single thread. The
///     default value is 50
///
/// :returns: A dictionary mapping each node index to its closeness centrality.
/// :rtype: CentralityMapping
#[pyfunction(signature = (graph, wf_improved=true, parallel_threshold=50))]
pub fn graph_closeness_centrality(
    graph: &graph::PyGraph,
    wf_improved: bool,
    parallel_threshold: usize,
) -> CentralityMapping {
    let closeness = centrality::closeness_centrality(&graph.graph, wf_improved, parallel_threshold);
    CentralityMapping {
        centralities: closeness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    }
}

/// Compute the closeness centrality of each node in a :class:`~.PyDiGraph` object.
///
/// The closeness centrality of a node :math:`u` is defined as the
/// reciprocal of the average shortest path distance to :math:`u` over all
/// :math:`n-1` reachable nodes in the graph. In it's general form this can
/// be expressed as:
///
/// .. math::
///
///     C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},
///
/// where:
///
///   * :math:`d(v, u)` - the shortest-path distance between :math:`v` and
///     :math:`u`
///   * :math:`n` - the number of nodes that can reach :math:`u`.
///
/// In the case of a graphs with more than one connected component there is
/// an alternative improved formula that calculates the closeness centrality
/// as "a ratio of the fraction of actors in the group who are reachable, to
/// the average distance" [WF]_. This can be expressed as
///
/// .. math::
///
///     C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},
///
/// where :math:`N` is the number of nodes in the graph. This alternative
/// formula can be used with the ``wf_improved`` argument.
///
/// :param PyDiGraph graph: The input graph. Can either be a
///     :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
/// :param bool wf_improved: This is optional; the default is True. If True,
///     scale by the fraction of nodes reachable.
/// :param int parallel_threshold: The number of nodes to calculate the
///     the betweenness centrality in parallel at if the number of nodes in
///     the graph is less than this value it will run in a single thread. The
///     default value is 50
///
/// :returns: A dictionary mapping each node index to its closeness centrality.
/// :rtype: CentralityMapping
#[pyfunction(signature = (graph, wf_improved=true, parallel_threshold=50))]
pub fn digraph_closeness_centrality(
    graph: &digraph::PyDiGraph,
    wf_improved: bool,
    parallel_threshold: usize,
) -> CentralityMapping {
    let closeness = centrality::closeness_centrality(&graph.graph, wf_improved, parallel_threshold);
    CentralityMapping {
        centralities: closeness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    }
}

/// Compute the weighted closeness centrality of each node in the graph.
///
/// The weighted closeness centrality is an extension of the standard closeness
/// centrality measure where edge weights represent connection strength rather
/// than distance. To properly compute shortest paths, weights are inverted
/// so that stronger connections correspond to shorter effective distances.
/// The algorithm follows the method described by Newman (2001) in analyzing
/// weighted graphs.[Newman]
///
/// The edges originally represent connection strength between nodes.
/// The idea is that if two nodes have a strong connection, the computed
/// distance between them should be small (shorter), and vice versa.
/// Note that this assume that the graph is modelling a measure of
/// connection strength (e.g. trust, collaboration, or similarity).
/// If the graph is not modelling a measure of connection strength,
/// the function `weight_fn` should invert the weights before calling this
/// function, if not it is considered as a logical error.
///
/// In the case of a graphs with more than one connected component there is
/// an alternative improved formula that calculates the closeness centrality
/// as "a ratio of the fraction of actors in the group who are reachable, to
/// the average distance".[WF]
/// You can enable this by setting `wf_improved` to `true`.
///
/// :param PyGraph graph: The input graph. Can either be a
///     :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
/// :param weight_fn: An optional input callable that will be passed the edge's
///    payload object and is expected to return a `float` weight for that edge.
///    If this is not specified ``default_weight`` will be used as the weight
///    for every edge in ``graph``
/// :param bool wf_improved: This is optional; the default is True. If True,
///   scale by the fraction of nodes reachable.
/// :param float default_weight: If ``weight_fn`` is not set the default weight
///     value to use for the weight of all edges
///
/// :returns: A dictionary mapping each node index to its closeness centrality.
/// :rtype: CentralityMapping
#[pyfunction(signature = (graph, weight_fn=None, wf_improved=true, default_weight = 1.0, parallel_threshold=50))]
pub fn graph_newman_weighted_closeness_centrality(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    wf_improved: bool,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<CentralityMapping> {
    let mut edge_weights = vec![default_weight; graph.graph.edge_bound()];
    if weight_fn.is_some() {
        let cost_fn = CostFn::try_from((weight_fn, default_weight))?;
        for edge in graph.graph.edge_indices() {
            edge_weights[edge.index()] =
                cost_fn.call(py, graph.graph.edge_weight(edge).unwrap())?;
        }
    }

    let closeness = centrality::newman_weighted_closeness_centrality(
        &graph.graph,
        wf_improved,
        |e| edge_weights[e.id().index()],
        parallel_threshold,
    );

    Ok(CentralityMapping {
        centralities: closeness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    })
}

/// Compute the weighted closeness centrality of each node in the graph.
///
/// The weighted closeness centrality is an extension of the standard closeness
/// centrality measure where edge weights represent connection strength rather
/// than distance. To properly compute shortest paths, weights are inverted
/// so that stronger connections correspond to shorter effective distances.
/// The algorithm follows the method described by Newman (2001) in analyzing
/// weighted graphs.[Newman]
///
/// The edges originally represent connection strength between nodes.
/// The idea is that if two nodes have a strong connection, the computed
/// distance between them should be small (shorter), and vice versa.
/// Note that this assume that the graph is modelling a measure of
/// connection strength (e.g. trust, collaboration, or similarity).
/// If the graph is not modelling a measure of connection strength,
/// the function `weight_fn` should invert the weights before calling this
/// function, if not it is considered as a logical error.
///
/// In the case of a graphs with more than one connected component there is
/// an alternative improved formula that calculates the closeness centrality
/// as "a ratio of the fraction of actors in the group who are reachable, to
/// the average distance".[WF]
/// You can enable this by setting `wf_improved` to `true`.
///
/// :param PyDiGraph graph: The input graph. Can either be a
///     :class:`~rustworkx.PyGraph` or :class:`~rustworkx.PyDiGraph`.
/// :param weight_fn: An optional input callable that will be passed the edge's
///    payload object and is expected to return a `float` weight for that edge.
///    If this is not specified ``default_weight`` will be used as the weight
///    for every edge in ``graph``
/// :param bool wf_improved: This is optional; the default is True. If True,
///   scale by the fraction of nodes reachable.
/// :param float default_weight: If ``weight_fn`` is not set the default weight
///     value to use for the weight of all edges
///
/// :returns: A dictionary mapping each node index to its closeness centrality.
/// :rtype: CentralityMapping
#[pyfunction(signature = (graph, weight_fn=None, wf_improved=true, default_weight = 1.0, parallel_threshold=50))]
pub fn digraph_newman_weighted_closeness_centrality(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    wf_improved: bool,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<CentralityMapping> {
    let mut edge_weights = vec![default_weight; graph.graph.edge_bound()];
    if weight_fn.is_some() {
        let cost_fn = CostFn::try_from((weight_fn, default_weight))?;
        for edge in graph.graph.edge_indices() {
            edge_weights[edge.index()] =
                cost_fn.call(py, graph.graph.edge_weight(edge).unwrap())?;
        }
    }

    let closeness = centrality::newman_weighted_closeness_centrality(
        &graph.graph,
        wf_improved,
        |e| edge_weights[e.id().index()],
        parallel_threshold,
    );

    Ok(CentralityMapping {
        centralities: closeness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    })
}

/// Compute the edge betweenness centrality of all edges in a :class:`~PyGraph`.
///
/// Edge betweenness centrality of an edge :math:`e` is the sum of the
/// fraction of all-pairs shortest paths that pass through :math:`e`
///
/// .. math::
///
///   c_B(e) =\sum_{s,t \in V} \frac{\sigma(s, t|e)}{\sigma(s, t)}
///
/// where :math:`V` is the set of nodes, :math:`\sigma(s, t)` is the
/// number of shortest :math:`(s, t)`-paths, and :math:`\sigma(s, t|e)` is
/// the number of those paths passing through edge :math:`e`.
///
/// The above definition and the algorithm used in this function is based on:
///
/// Ulrik Brandes, On Variants of Shortest-Path Betweenness Centrality
/// and their Generic Computation. Social Networks 30(2):136-145, 2008.
///
/// This function is multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold`` (it
/// defaults to 50). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// See Also
/// --------
/// :func:`~rustworkx.graph_betweenness_centrality`
///
/// :param PyGraph graph: The input graph
/// :param bool normalized: Whether to normalize the betweenness scores by the number of distinct
///    paths between all pairs of nodes.
/// :param int parallel_threshold: The number of nodes to calculate the
///     the betweenness centrality in parallel at if the number of nodes in
///     the graph is less than this value it will run in a single thread. The
///     default value is 50
///
/// :returns: a read-only dict-like object whose keys are the edge indices and values are the
///      betweenness score for each edge.
/// :rtype: EdgeCentralityMapping
#[pyfunction(
    signature = (
        graph,
        normalized=true,
        parallel_threshold=50
    )
)]
#[pyo3(text_signature = "(graph, /, normalized=True, parallel_threshold=50)")]
pub fn graph_edge_betweenness_centrality(
    graph: &graph::PyGraph,
    normalized: bool,
    parallel_threshold: usize,
) -> PyResult<EdgeCentralityMapping> {
    let betweenness =
        centrality::edge_betweenness_centrality(&graph.graph, normalized, parallel_threshold);
    Ok(EdgeCentralityMapping {
        centralities: betweenness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    })
}

/// Compute the edge betweenness centrality of all edges in a :class:`~PyDiGraph`.
///
/// Edge betweenness centrality of an edge :math:`e` is the sum of the
/// fraction of all-pairs shortest paths that pass through :math:`e`
///
/// .. math::
///
///   c_B(e) =\sum_{s,t \in V} \frac{\sigma(s, t|e)}{\sigma(s, t)}
///
/// where :math:`V` is the set of nodes, :math:`\sigma(s, t)` is the
/// number of shortest :math:`(s, t)`-paths, and :math:`\sigma(s, t|e)` is
/// the number of those paths passing through edge :math:`e`.
///
/// The above definition and the algorithm used in this function is based on:
///
/// Ulrik Brandes, On Variants of Shortest-Path Betweenness Centrality
/// and their Generic Computation. Social Networks 30(2):136-145, 2008.
///
/// This function is multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold`` (it
/// defaults to 50). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// See Also
/// --------
/// :func:`~rustworkx.digraph_betweenness_centrality`
///
/// :param PyGraph graph: The input graph
/// :param bool normalized: Whether to normalize the betweenness scores by the number of distinct
///    paths between all pairs of nodes.
/// :param int parallel_threshold: The number of nodes to calculate the
///     the betweenness centrality in parallel at if the number of nodes in
///     the graph is less than this value it will run in a single thread. The
///     default value is 50
///
/// :returns: a read-only dict-like object whose keys are edges and values are the
///      betweenness score for each node.
/// :rtype: EdgeCentralityMapping
#[pyfunction(
    signature = (
        graph,
        normalized=true,
        parallel_threshold=50
    )
)]
#[pyo3(text_signature = "(graph, /, normalized=True, parallel_threshold=50)")]
pub fn digraph_edge_betweenness_centrality(
    graph: &digraph::PyDiGraph,
    normalized: bool,
    parallel_threshold: usize,
) -> PyResult<EdgeCentralityMapping> {
    let betweenness =
        centrality::edge_betweenness_centrality(&graph.graph, normalized, parallel_threshold);
    Ok(EdgeCentralityMapping {
        centralities: betweenness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    })
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
#[pyfunction(
    signature = (
        graph,
        weight_fn=None,
        default_weight=1.0,
        max_iter=100,
        tol=1e-6
    )
)]
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
#[pyfunction(
    signature = (
        graph,
        weight_fn=None,
        default_weight=1.0,
        max_iter=100,
        tol=1e-6
    )
)]
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

/// Compute the Katz centrality of a :class:`~PyGraph`.
///
/// For details on the Katz centrality refer to:
///
/// Leo Katz. “A New Status Index Derived from Sociometric Index.”
/// Psychometrika 18(1):39–43, 1953
/// <https://link.springer.com/content/pdf/10.1007/BF02289026.pdf>
///
/// This function uses a power iteration method to compute the eigenvector
/// and convergence is not guaranteed. The function will stop when `max_iter`
/// iterations is reached or when the computed vector between two iterations
/// is smaller than the error tolerance multiplied by the number of nodes.
/// The implementation of this algorithm is based on the NetworkX
/// `katz_centrality() <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.katz_centrality.html>`__
/// function.
///
/// In the case of multigraphs the weights of any parallel edges will be
/// summed when computing the Katz centrality.
///
/// :param PyGraph graph: The graph object to run the algorithm on
/// :param float alpha: Attenuation factor. If this is not specified default value of 0.1 is used.
/// :param float | dict beta: Immediate neighbourhood weights. If a float is provided, the neighbourhood
///     weight is used for all nodes. If a dictionary is provided, it must contain all node indices.
///     If beta is not specified, a default value of 1.0 is used.
/// :param weight_fn: An optional input callable that will be passed the edge's
///     payload object and is expected to return a `float` weight for that edge.
///     If this is not specified ``default_weight`` will be used as the weight
///     for every edge in ``graph``
/// :param float default_weight: If ``weight_fn`` is not set the default weight
///     value to use for the weight of all edges
/// :param int max_iter: The maximum number of iterations in the power method. If
///     not specified a default value of 1000 is used.
/// :param float tol: The error tolerance used when checking for convergence in the
///     power method. If this is not specified default value of 1e-6 is used.
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      centrality score for that node.
/// :rtype: CentralityMapping
#[pyfunction(
    signature = (
        graph,
        alpha=0.1,
        beta=None,
        weight_fn=None,
        default_weight=1.0,
        max_iter=1000,
        tol=1e-6
    )
)]
#[pyo3(
    text_signature = "(graph, /, alpha=0.1, beta=None, weight_fn=None, default_weight=1.0, max_iter=1000, tol=1e-6)"
)]
pub fn graph_katz_centrality(
    py: Python,
    graph: &graph::PyGraph,
    alpha: f64,
    beta: Option<PyObject>,
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

    let mut beta_map: HashMap<usize, f64> = HashMap::new();

    if let Some(beta) = beta {
        match beta.extract::<f64>(py) {
            Ok(beta_scalar) => {
                // User provided a scalar, populate beta_map with the value
                for node_index in graph.graph.node_identifiers() {
                    beta_map.insert(node_index.index(), beta_scalar);
                }
            }
            Err(_) => {
                beta_map = beta.extract::<HashMap<usize, f64>>(py)?;

                for node_index in graph.graph.node_identifiers() {
                    if !beta_map.contains_key(&node_index.index()) {
                        return Err(PyValueError::new_err(
                            "Beta does not contain all node indices",
                        ));
                    }
                }
            }
        }
    } else {
        // Populate with 1.0
        for node_index in graph.graph.node_identifiers() {
            beta_map.insert(node_index.index(), 1.0);
        }
    }

    let ev_centrality = centrality::katz_centrality(
        &graph.graph,
        |e| -> PyResult<f64> { Ok(edge_weights[e.id().index()]) },
        Some(alpha),
        Some(beta_map),
        None,
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

/// Compute the Katz centrality of a :class:`~PyDiGraph`.
///
/// For details on the Katz centrality refer to:
///
/// Leo Katz. “A New Status Index Derived from Sociometric Index.”
/// Psychometrika 18(1):39–43, 1953
/// <https://link.springer.com/content/pdf/10.1007/BF02289026.pdf>
///
/// This function uses a power iteration method to compute the eigenvector
/// and convergence is not guaranteed. The function will stop when `max_iter`
/// iterations is reached or when the computed vector between two iterations
/// is smaller than the error tolerance multiplied by the number of nodes.
/// The implementation of this algorithm is based on the NetworkX
/// `katz_centrality() <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.katz_centrality.html>`__
/// function.
///
/// In the case of multigraphs the weights of any parallel edges will be
/// summed when computing the Katz centrality.
///
/// :param PyDiGraph graph: The graph object to run the algorithm on
/// :param float alpha: Attenuation factor. If this is not specified default value of 0.1 is used.
/// :param float | dict beta: Immediate neighbourhood weights. If a float is provided, the neighbourhood
///     weight is used for all nodes. If a dictionary is provided, it must contain all node indices.
///     If beta is not specified, a default value of 1.0 is used.
/// :param weight_fn: An optional input callable that will be passed the edge's
///     payload object and is expected to return a `float` weight for that edge.
///     If this is not specified ``default_weight`` will be used as the weight
///     for every edge in ``graph``
/// :param float default_weight: If ``weight_fn`` is not set the default weight
///     value to use for the weight of all edges
/// :param int max_iter: The maximum number of iterations in the power method. If
///     not specified a default value of 1000 is used.
/// :param float tol: The error tolerance used when checking for convergence in the
///     power method. If this is not specified default value of 1e-6 is used.
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are the
///      centrality score for that node.
/// :rtype: CentralityMapping
#[pyfunction(
    signature = (
        graph,
        alpha=0.1,
        beta=None,
        weight_fn=None,
        default_weight=1.0,
        max_iter=1000,
        tol=1e-6
    )
)]
#[pyo3(
    text_signature = "(graph, /, alpha=0.1, beta=None, weight_fn=None, default_weight=1.0, max_iter=1000, tol=1e-6)"
)]
pub fn digraph_katz_centrality(
    py: Python,
    graph: &digraph::PyDiGraph,
    alpha: f64,
    beta: Option<PyObject>,
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

    let mut beta_map: HashMap<usize, f64> = HashMap::new();

    if let Some(beta) = beta {
        match beta.extract::<f64>(py) {
            Ok(beta_scalar) => {
                // User provided a scalar, populate beta_map with the value
                for node_index in graph.graph.node_identifiers() {
                    beta_map.insert(node_index.index(), beta_scalar);
                }
            }
            Err(_) => {
                beta_map = beta.extract::<HashMap<usize, f64>>(py)?;

                for node_index in graph.graph.node_identifiers() {
                    if !beta_map.contains_key(&node_index.index()) {
                        return Err(PyValueError::new_err(
                            "Beta does not contain all node indices",
                        ));
                    }
                }
            }
        }
    } else {
        // Populate with 1.0
        for node_index in graph.graph.node_identifiers() {
            beta_map.insert(node_index.index(), 1.0);
        }
    }

    let ev_centrality = centrality::katz_centrality(
        &graph.graph,
        |e| -> PyResult<f64> { Ok(edge_weights[e.id().index()]) },
        Some(alpha),
        Some(beta_map),
        None,
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
