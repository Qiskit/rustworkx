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

use crate::graph;
use rustworkx_core::max_weight_matching as mwm;

use hashbrown::HashSet;

use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::IntoEdgeReferences;

use crate::weight_callable;

/// Compute a maximum-weighted matching for a :class:`~rustworkx.PyGraph`
///
/// A matching is a subset of edges in which no node occurs more than once.
/// The weight of a matching is the sum of the weights of its edges.
/// A maximal matching cannot add more edges and still be a matching.
/// The cardinality of a matching is the number of matched edges.
///
/// This function takes time :math:`O(n^3)` where ``n`` is the number of nodes
/// in the graph.
///
/// This method is based on the "blossom" method for finding augmenting
/// paths and the "primal-dual" method for finding a matching of maximum
/// weight, both methods invented by Jack Edmonds [1]_.
///
/// :param PyGraph graph: The undirected graph to compute the max weight
///     matching for. Expects to have no parallel edges (multigraphs are
///     untested currently).
/// :param bool max_cardinality: If True, compute the maximum-cardinality
///     matching with maximum weight among all maximum-cardinality matchings.
///     Defaults False.
/// :param callable weight_fn: An optional callable that will be passed a
///     single argument the edge object for each edge in the graph. It is
///     expected to return an ``int`` weight for that edge. For example,
///     if the weights are all integers you can use: ``lambda x: x``. If not
///     specified the value for ``default_weight`` will be used for all
///     edge weights.
/// :param int default_weight: The ``int`` value to use for all edge weights
///     in the graph if ``weight_fn`` is not specified. Defaults to ``1``.
/// :param bool verify_optimum: A boolean flag to run a check that the found
///     solution is optimum. If set to true an exception will be raised if
///     the found solution is not optimum. This is mostly useful for testing.
///
/// :returns: A set of tuples ofthe matching, Note that only a single
///     direction will be listed in the output, for example:
///     ``{(0, 1),}``.
/// :rtype: set
///
/// .. [1] "Efficient Algorithms for Finding Maximum Matching in Graphs",
///     Zvi Galil, ACM Computing Surveys, 1986.
///
#[pyfunction(
    max_cardinality = "false",
    default_weight = 1,
    verify_optimum = "false"
)]
#[pyo3(
    text_signature = "(graph, /, max_cardinality=False, weight_fn=None, default_weight=1, verify_optimum=False)"
)]
pub fn max_weight_matching(
    py: Python,
    graph: &graph::PyGraph,
    max_cardinality: bool,
    weight_fn: Option<PyObject>,
    default_weight: i128,
    verify_optimum: bool,
) -> PyResult<HashSet<(usize, usize)>> {
    mwm::max_weight_matching(
        &graph.graph,
        max_cardinality,
        |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
        verify_optimum,
    )
}

fn _inner_is_matching(graph: &graph::PyGraph, matching: &HashSet<(usize, usize)>) -> bool {
    let has_edge = |e: &(usize, usize)| -> bool {
        graph
            .graph
            .contains_edge(NodeIndex::new(e.0), NodeIndex::new(e.1))
    };

    if !matching.iter().all(|e| has_edge(e)) {
        return false;
    }
    let mut found: HashSet<usize> = HashSet::with_capacity(2 * matching.len());
    for (v1, v2) in matching {
        if found.contains(v1) || found.contains(v2) {
            return false;
        }
        found.insert(*v1);
        found.insert(*v2);
    }
    true
}

/// Check if matching is valid for graph
///
/// A *matching* in a graph is a set of edges in which no two distinct
/// edges share a common endpoint.
///
/// :param PyDiGraph graph: The graph to check if the matching is valid for
/// :param set matching: A set of node index tuples for each edge in the
///     matching.
///
/// :returns: Whether the provided matching is a valid matching for the graph
/// :rtype: bool
#[pyfunction]
#[pyo3(text_signature = "(graph, matching, /)")]
pub fn is_matching(graph: &graph::PyGraph, matching: HashSet<(usize, usize)>) -> bool {
    _inner_is_matching(graph, &matching)
}

/// Check if a matching is a maximal (**not** maximum) matching for a graph
///
/// A *maximal matching* in a graph is a matching in which adding any
/// edge would cause the set to no longer be a valid matching.
///
/// .. note::
///
///   This is not checking for a *maximum* (globally optimal) matching, but
///   a *maximal* (locally optimal) matching.
///
/// :param PyDiGraph graph: The graph to check if the matching is maximal for.
/// :param set matching: A set of node index tuples for each edge in the
///     matching.
///
/// :returns: Whether the provided matching is a valid matching and whether it
///     is maximal or not.
/// :rtype: bool
#[pyfunction]
#[pyo3(text_signature = "(graph, matching, /)")]
pub fn is_maximal_matching(graph: &graph::PyGraph, matching: HashSet<(usize, usize)>) -> bool {
    if !_inner_is_matching(graph, &matching) {
        return false;
    }
    let edge_list: HashSet<[usize; 2]> = graph
        .graph
        .edge_references()
        .map(|edge| {
            let mut tmp_array = [edge.source().index(), edge.target().index()];
            tmp_array.sort_unstable();
            tmp_array
        })
        .collect();
    let matched_edges: HashSet<[usize; 2]> = matching
        .iter()
        .map(|edge| {
            let mut tmp_array = [edge.0, edge.1];
            tmp_array.sort_unstable();
            tmp_array
        })
        .collect();
    let mut unmatched_edges = edge_list.difference(&matched_edges);
    unmatched_edges.all(|e| {
        let mut tmp_set = matching.clone();
        tmp_set.insert((e[0], e[1]));
        !_inner_is_matching(graph, &tmp_set)
    })
}
