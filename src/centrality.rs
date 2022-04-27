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

use crate::iterators::CentralityMapping;

use crate::digraph;
use crate::graph;

use pyo3::prelude::*;

use retworkx_core::centrality;

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

/// Compute the closeness centrality of all nodes in a PyGraph.
///
/// :param PyGraph graph: The input graph
/// :param bool wf_improved: If True, scale by the fraction of nodes reachable.
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are its
///     closeness centrality score for each node.
/// :rtype: CentralityMapping
#[pyfunction(wf_improved = "true")]
#[pyo3(text_signature = "(graph, /, wf_improved=True)")]
pub fn graph_closeness_centrality(graph: &graph::PyGraph, wf_improved: bool) -> CentralityMapping {
    let betweenness = centrality::closeness_centrality(&graph.graph, wf_improved);
    CentralityMapping {
        centralities: betweenness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    }
}

/// Compute the closeness centrality of all nodes in a PyDiGraph.
///
/// :param PyDiGraph graph: The input digraph
/// :param bool wf_improved: If True, scale by the fraction of nodes reachable.
///
/// :returns: a read-only dict-like object whose keys are the node indices and values are its
///     closeness centrality score for each node.
/// :rtype: CentralityMapping
#[pyfunction(wf_improved = "true")]
#[pyo3(text_signature = "(graph, /, wf_improved=True)")]
pub fn digraph_closeness_centrality(
    graph: &digraph::PyDiGraph,
    wf_improved: bool,
) -> CentralityMapping {
    let betweenness = centrality::closeness_centrality(&graph.graph, wf_improved);
    CentralityMapping {
        centralities: betweenness
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|x| (i, x)))
            .collect(),
    }
}
