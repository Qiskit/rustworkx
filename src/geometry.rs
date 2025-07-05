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

use petgraph::visit::NodeIndexable;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::graph;
use rustworkx_core::geometry as core_geometry;

fn dist<G>(graph: &G, pos: &[Vec<f64>], u: G::NodeId, v: G::NodeId) -> f64
where
    G: petgraph::visit::NodeIndexable,
{
    core_geometry::hyperboloid_hyperbolic_distance(&pos[graph.to_index(u)], &pos[graph.to_index(v)])
        .unwrap()
}

/// Performs the greedy routing algorithm from ``source`` to ``destination`` using the hyperbolic
/// distance.
///
/// The greedy routing algorithm attempts to find the shortest path between ``source`` and
/// ``destination`` by starting from ``source`` and continuously going to the neighbor closest to
/// ``destination``. The closest neighbor of :math:`u` is the node :math:`v` that minimizes the
/// hyperbolic distance
///
/// .. math::
///
///     d(u,v) = \text{arccosh}\left[x_0(u) x_0(v) - \sum_{j=1}^D x_j(u) x_j(v) \right],
///
/// where :math:`D` is the dimension of the hyperbolic space and :math:`x_d(u)` is the
/// :math:`d` th-dimension coordinate of node :math:`u` in the hyperboloid model. The distance is
/// computed using ``pos``, an array of shape (n, D) in which the :math:`i`-th row is the position
/// of the :math:`i` -th indexed vertex in the graph. The dimension is inferred from the
/// coordinates ``pos`` and the 0-dimension "time" coordinate is inferred from the other
/// coordinates. Note that the hyperbolic space is at least 2 dimensional.
///
/// .. note::
///
///     Node indices are constant across removal of nodes. Since ``pos`` maps rows to node indices,
///     the unused rows should not be deleted.
///
/// This algorithm has a time complexity of :math:`O(m)` for :math:`m` edges.
///
/// :param PyGraph graph: The input graph.
/// :param list[list[float]] pos: Hyperboloid coordinates of the nodes
///     [[:math:`x_1(1)`, ..., :math:`x_D(1)`], [:math:`x_1(2)`, ..., :math:`x_D(2)`], ...].
///     The "time" coordinate :math:`x_0` is inferred from the other coordinates.
/// :param int source: source of the greedy routing path.
/// :param int destination: destination of the greedy routing path.
///
/// :returns: A tuple ``(path, length)`` where ``path`` is the path followed and ``length`` is the
///     sum of the distance between nodes on the path. Returns ``None`` if the greedy algorithm fails.
/// to reach ``destination``.
/// :rtype: Optional[tuple[list[int], float]]
#[pyfunction]
#[pyo3(
    signature=(graph, pos, source, destination),
)]
pub fn hyperbolic_greedy_routing(
    graph: &graph::PyGraph,
    pos: Vec<Vec<f64>>,
    source: usize,
    destination: usize,
) -> PyResult<Option<(Vec<usize>, f64)>> {
    if pos.len() < graph.graph.node_count() {
        return Err(PyValueError::new_err(
            "Graph contains more nodes than there are positions.",
        ));
    }
    if !graph.has_node(source) || !graph.has_node(destination) {
        return Err(PyValueError::new_err(
            "The graph doesn't contain the source or destination node.",
        ));
    }
    let dim = pos[0].len();
    if dim < 2 || pos.iter().any(|x| x.len() != dim) {
        return Err(PyValueError::new_err("Each node must have the same number of coordinates and must have at least 2 coordinates."));
    }

    Ok(core_geometry::greedy_routing(
        &graph.graph,
        NodeIndexable::from_index(&graph.graph, source),
        NodeIndexable::from_index(&graph.graph, destination),
        |u, v| dist(&graph.graph, &pos, u, v),
    )
    .map(|(path, length)| {
        (
            path.iter()
                .map(|&x| NodeIndexable::to_index(&graph.graph, x))
                .collect(),
            length,
        )
    })
    .ok())
}

/// Returns the proportion of successful hyperbolic greedy routing paths between all pairs of
/// nodes. See :func:`.hyperbolic_greedy_routing` for details on the greedy routing algorithm.
///
/// This algorithm has a time complexity of :math:`O(n^2 m)` for :math:`n` nodes and :math:`m`
/// edges.
///
/// .. note::
///
///     Node indices are constant across removal of nodes. Since ``pos`` maps rows to node indices,
///     the unused rows should not be deleted.
///
/// :param PyGraph graph: The input graph.
/// :param list[list[float]] pos: Hyperboloid coordinates of the nodes
///     [[:math:`x_1(1)`, ..., :math:`x_D(1)`], [:math:`x_1(2)`, ..., :math:`x_D(2)`], ...].
///     The "time" coordinate :math:`x_0` is inferred from the other coordinates.
///
/// :returns: Proportion of successful greedy paths.
/// :rtype: float
#[pyfunction]
#[pyo3(
    signature=(graph, pos),
)]
pub fn hyperbolic_greedy_success_rate(graph: &graph::PyGraph, pos: Vec<Vec<f64>>) -> PyResult<f64> {
    if pos.len() < graph.graph.node_count() {
        return Err(PyValueError::new_err(
            "Graph contains more nodes than there are positions.",
        ));
    }
    if pos.is_empty() {
        return Ok(f64::NAN);
    }
    let dim = pos[0].len();
    if dim < 2 || pos.iter().any(|x| x.len() != dim) {
        return Err(PyValueError::new_err("Each node must have the same number of coordinates and must have at least 2 coordinates."));
    }

    Ok(core_geometry::greedy_routing_success_rate(
        &graph.graph,
        |u, v| dist(&graph.graph, &pos, u, v),
    ))
}
