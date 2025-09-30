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

use std::hash::Hash;

use petgraph::algo::Measure;
use petgraph::visit::{
    IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable, VisitMap, Visitable,
};

/// Error returned when a greedy routing fails.
#[derive(Debug, PartialEq, Eq)]
pub struct NodeNotReachedError;

/// Returns the proportion of successful greedy routing paths between all pairs of nodes.
///
/// See [`greedy_routing`] for details on the greedy routing algorithm.
///
/// # Example
/// ```rust
///
/// use rustworkx_core::petgraph;
/// use rustworkx_core::geometry::{euclidean_distance, greedy_routing_success_rate};
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2), (0, 3), (3, 4), (1, 4), (4, 5)
/// ]);
///
/// fn distance2d<G>(graph: &G, positions: &Vec<[f64; 2]>, u: G::NodeId, v: G::NodeId) -> f64 where G: petgraph::visit::NodeIndexable {
///     euclidean_distance(&positions[graph.to_index(u)], &positions[graph.to_index(v)]).unwrap()
/// }
///
/// let positions = vec![[1., 0.] , [2., 3.], [3., 0.], [1., -1.], [2., -1.], [3., -1.]];
/// let success_rate = greedy_routing_success_rate(&g, |u, v| {distance2d(&g, &positions, u, v)});
/// assert!( (success_rate - 26./30.).abs() < 1e-15);
/// ```
pub fn greedy_routing_success_rate<G, F, K>(graph: G, distance: F) -> f64
where
    G: IntoEdges + NodeCount + NodeIndexable + IntoNodeIdentifiers + Visitable,
    G::NodeId: Eq + Hash,
    F: Fn(G::NodeId, G::NodeId) -> K,
    K: Measure + Copy,
{
    let num_vertices = graph.node_count() as f64;
    let num_successes = graph
        .node_identifiers()
        .flat_map(|u| graph.node_identifiers().map(move |v| (u, v)))
        .map(|(u, v)| ((u != v) && greedy_routing(graph, u, v, &distance).is_ok()) as u32)
        .sum::<u32>() as f64;
    num_successes / (num_vertices * num_vertices - num_vertices)
}

/// Greedy routing shortest path algorithm.
///
/// Successively follows the neighbor closest to `destination` from `source` until `destination` is
/// reached. The closest neighbor is the node that minimizes the `distance` function.
///
/// Returns the greedy path and the sum of the distances in the path. A [`NodeNotReachedError`] is
/// returned if the greedy routing fails to reach `destination`.
///
/// # Example:
/// ```rust
///
/// use rustworkx_core::petgraph;
/// use rustworkx_core::geometry::{euclidean_distance, greedy_routing};
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (2, 5), (5, 6)
/// ]);
///
/// fn distance2d<G>(graph: &G, positions: &Vec<[f64; 2]>, u: G::NodeId, v: G::NodeId) -> f64 where G: petgraph::visit::NodeIndexable {
///     euclidean_distance(&positions[graph.to_index(u)], &positions[graph.to_index(v)]).unwrap()
/// }
///
/// let positions = vec![[0., 0.], [1., 0.] , [2., 0.], [2.5, 0.], [1., 1.], [2., 1.], [3., 1.]];
/// let (path, dist) = greedy_routing(&g, 0.into(), 6.into(), |u, v| {distance2d(&g, &positions, u, v)}).unwrap();
/// assert_eq!(path, vec![0.into(), 1.into(), 2.into(), 5.into(), 6.into()]);
/// assert!( (dist - (1.+1.+1.+1.)).abs() < 1e-15 );
/// ```
///
pub fn greedy_routing<G, F, K>(
    graph: G,
    source: G::NodeId,
    destination: G::NodeId,
    distance: F,
) -> Result<(Vec<G::NodeId>, K), NodeNotReachedError>
where
    G: IntoEdges + NodeCount + NodeIndexable + Visitable,
    G::NodeId: Eq + Hash,
    F: Fn(G::NodeId, G::NodeId) -> K,
    K: Measure + Copy,
{
    if source == destination {
        return Ok((vec![source], distance(source, destination)));
    }
    let mut visitmap = graph.visit_map();
    let mut path: Vec<G::NodeId> = vec![source];
    let mut total_distance = K::default();

    let mut current_vertex = source;
    while current_vertex != destination {
        if visitmap.is_visited(&current_vertex) {
            return Err(NodeNotReachedError {});
        }
        visitmap.visit(current_vertex);

        let mut min_distance = None;
        let mut next = current_vertex;
        for neighbor in graph.neighbors(current_vertex) {
            let d = distance(neighbor, destination);
            if min_distance.is_none() || d < min_distance.unwrap() {
                next = neighbor;
                min_distance = Some(d);
            }
        }
        if min_distance.is_none() {
            return Err(NodeNotReachedError {});
        }
        total_distance = total_distance + distance(current_vertex, next);
        current_vertex = next;
        path.push(current_vertex);
    }
    Ok((path, total_distance))
}

#[cfg(test)]
mod tests {
    use super::{greedy_routing, greedy_routing_success_rate, NodeNotReachedError};
    use crate::geometry::distances;
    use crate::petgraph::graph::UnGraph;
    use petgraph::visit::NodeIndexable;
    use petgraph::Graph;

    fn distance1d<G>(graph: &G, positions: &[f64], u: G::NodeId, v: G::NodeId) -> f64
    where
        G: NodeIndexable,
    {
        distances::lp_distance(
            &[positions[graph.to_index(u)]],
            &[positions[graph.to_index(v)]],
            1,
        )
        .unwrap()
    }
    fn distance2d<G>(graph: &G, positions: &[[f64; 2]], u: G::NodeId, v: G::NodeId) -> f64
    where
        G: NodeIndexable,
    {
        distances::euclidean_distance(&positions[graph.to_index(u)], &positions[graph.to_index(v)])
            .unwrap()
    }

    #[test]
    fn test_unreachable_destination_error() {
        // Disconnected graph
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(4, 2);

        let a = graph.add_node(());
        let b = graph.add_node(());

        let positions = vec![1., 2.];
        assert_eq!(
            greedy_routing(&graph, a, b, |u, v| distance1d(&graph, &positions, u, v)),
            Err(NodeNotReachedError {})
        )
    }

    #[test]
    fn test_greedy_loop_error() {
        // a -- b -- c -- d
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(4, 2);

        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());

        graph.extend_with_edges([(a, b), (b, c), (c, d)]);

        let positions = vec![[0., 0.], [1., 0.], [10., 0.], [4., 0.]];
        assert_eq!(
            greedy_routing(&graph, a, d, |u, v| distance2d(&graph, &positions, u, v)),
            Err(NodeNotReachedError {})
        )
    }

    #[test]
    fn test_correct_greedy_path() {
        // a -- b -- c -- d
        //      |    |
        //      e -- f -- g
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(7, 7);

        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        let f = graph.add_node(());
        let g = graph.add_node(());

        graph.extend_with_edges([(a, b), (b, c), (c, d), (b, e), (e, f), (c, f), (f, g)]);

        let positions = vec![
            [0., 0.],
            [1., 0.],
            [2., 0.],
            [2.5, 0.],
            [1., 1.],
            [2., 1.],
            [3., 1.],
        ];
        let (path, dist) =
            greedy_routing(&graph, a, d, |u, v| distance2d(&graph, &positions, u, v)).unwrap();
        assert_eq!(path, vec![a, b, c, d]);
        assert!((dist - (1. + 1. + 0.5)).abs() < 1e-15);

        let (path, dist) =
            greedy_routing(&graph, a, g, |u, v| distance2d(&graph, &positions, u, v)).unwrap();
        assert_eq!(path, vec![a, b, c, f, g]);
        assert!((dist - (1. + 1. + 1. + 1.)).abs() < 1e-15);
    }

    #[test]
    fn test_correct_greedy_success_rate() {
        // b -- c -- d
        // |    |
        // e -- f -- g
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(6, 6);

        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        let f = graph.add_node(());
        let g = graph.add_node(());

        graph.extend_with_edges([(b, c), (c, d), (b, e), (e, f), (c, f), (f, g)]);

        let positions = vec![
            [1., 0.],
            [2., 3.],
            [3., 0.],
            [1., -1.],
            [2., -1.],
            [3., -1.],
        ];
        let success_rate =
            greedy_routing_success_rate(&graph, |u, v| distance2d(&graph, &positions, u, v));
        // Greedy paths b -> d, e->d, f->d and g->d fail.
        assert!((success_rate - 26. / 30.).abs() < 1e-15);
    }
}
