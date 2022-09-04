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

// This module is copied and forked from the upstream petgraph repository,
// specifically:
// https://github.com/petgraph/petgraph/blob/0.5.1/src/astar.rs and
// https://github.com/petgraph/petgraph/blob/0.5.1/src/scored.rs
// this was necessary to modify the error handling to allow python callables
// to be use for the input functions for is_goal, edge_cost, estimate_cost
// and return any exceptions raised in Python instead of panicking

use std::collections::BinaryHeap;
use std::hash::Hash;

use hashbrown::hash_map::Entry::{Occupied, Vacant};
use hashbrown::HashMap;

use petgraph::algo::Measure;
use petgraph::visit::{EdgeRef, GraphBase, IntoEdges, VisitMap, Visitable};

use crate::min_scored::MinScored;

type AstarOutput<K, N> = Option<(K, Vec<N>)>;

/// A* shortest path algorithm.
///
/// Computes the shortest path from `start` to `finish`, including the total path cost.
///
/// `finish` is implicitly given via the `is_goal` callback, which should return `true` if the
/// given node is the finish node.
///
/// The function `edge_cost` should return the cost for a particular edge. Edge costs must be
/// non-negative.
///
/// The function `estimate_cost` should return the estimated cost to the finish for a particular
/// node. For the algorithm to find the actual shortest path, it should be admissible, meaning that
/// it should never overestimate the actual cost to get to the nearest goal node. Estimate costs
/// must also be non-negative.
///
/// The graph should be [`Visitable`] and implement [`IntoEdges`].
///
/// # Example
/// ```
/// use rustworkx_core::petgraph::graph::NodeIndex;
/// use rustworkx_core::petgraph::Graph;
/// use rustworkx_core::shortest_path::astar;
/// use rustworkx_core::Result;
///
/// let mut g = Graph::new();
/// let a = g.add_node((0., 0.));
/// let b = g.add_node((2., 0.));
/// let c = g.add_node((1., 1.));
/// let d = g.add_node((0., 2.));
/// let e = g.add_node((3., 3.));
/// let f = g.add_node((4., 2.));
/// g.extend_with_edges(&[
///     (a, b, 2),
///     (a, d, 4),
///     (b, c, 1),
///     (b, f, 7),
///     (c, e, 5),
///     (e, f, 1),
///     (d, e, 1),
/// ]);
///
/// // Graph represented with the weight of each edge
/// // Edges with '*' are part of the optimal path.
/// //
/// //     2       1
/// // a ----- b ----- c
/// // | 4*    | 7     |
/// // d       f       | 5
/// // | 1*    | 1*    |
/// // \------ e ------/
///
/// let res: Result<Option<(u64, Vec<NodeIndex>)>> = astar(
///     &g, a, |finish| Ok(finish == f), |e| Ok(*e.weight()), |_| Ok(0)
/// );
/// let path = res.unwrap();
/// assert_eq!(path, Some((6, vec![a, d, e, f])));
/// ```
///
/// Returns the total cost + the path of subsequent `NodeId` from start to finish, if one was
/// found.
pub fn astar<G, F, H, K, IsGoal, E>(
    graph: G,
    start: G::NodeId,
    mut is_goal: IsGoal,
    mut edge_cost: F,
    mut estimate_cost: H,
) -> Result<AstarOutput<K, G::NodeId>, E>
where
    G: IntoEdges + Visitable,
    IsGoal: FnMut(G::NodeId) -> Result<bool, E>,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    H: FnMut(G::NodeId) -> Result<K, E>,
    K: Measure + Copy,
{
    let mut visited = graph.visit_map();
    let mut visit_next = BinaryHeap::new();
    let mut scores = HashMap::new();
    let mut path_tracker = PathTracker::<G>::new();

    let zero_score = K::default();
    scores.insert(start, zero_score);
    let estimate = estimate_cost(start)?;
    visit_next.push(MinScored(estimate, start));

    while let Some(MinScored(_, node)) = visit_next.pop() {
        let result = is_goal(node)?;
        if result {
            let path = path_tracker.reconstruct_path_to(node);
            let cost = scores[&node];
            return Ok(Some((cost, path)));
        }

        // Don't visit the same node several times, as the first time it was visited it was using
        // the shortest available path.
        if !visited.visit(node) {
            continue;
        }

        // This lookup can be unwrapped without fear of panic since the node was necessarily scored
        // before adding him to `visit_next`.
        let node_score = scores[&node];

        for edge in graph.edges(node) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }

            let cost = edge_cost(edge)?;
            let mut next_score = node_score + cost;

            match scores.entry(next) {
                Occupied(ent) => {
                    let old_score = *ent.get();
                    if next_score < old_score {
                        *ent.into_mut() = next_score;
                        path_tracker.set_predecessor(next, node);
                    } else {
                        next_score = old_score;
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    path_tracker.set_predecessor(next, node);
                }
            }

            let estimate = estimate_cost(next)?;
            let next_estimate_score = next_score + estimate;
            visit_next.push(MinScored(next_estimate_score, next));
        }
    }

    Ok(None)
}

struct PathTracker<G>
where
    G: GraphBase,
    G::NodeId: Eq + Hash,
{
    came_from: HashMap<G::NodeId, G::NodeId>,
}

impl<G> PathTracker<G>
where
    G: GraphBase,
    G::NodeId: Eq + Hash,
{
    fn new() -> PathTracker<G> {
        PathTracker {
            came_from: HashMap::new(),
        }
    }

    fn set_predecessor(&mut self, node: G::NodeId, previous: G::NodeId) {
        self.came_from.insert(node, previous);
    }

    fn reconstruct_path_to(&self, last: G::NodeId) -> Vec<G::NodeId> {
        let mut path = vec![last];

        let mut current = last;
        while let Some(&previous) = self.came_from.get(&current) {
            path.push(previous);
            current = previous;
        }

        path.reverse();

        path
    }
}

#[cfg(test)]
mod tests {

    use crate::dictmap::DictMap;
    use crate::shortest_path::{astar, dijkstra};
    use crate::Result;
    use petgraph::graph::NodeIndex;
    use petgraph::Graph;

    #[test]
    fn test_astar_null_heuristic() {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        let d = g.add_node("D");
        let e = g.add_node("E");
        let f = g.add_node("F");
        g.add_edge(a, b, 7);
        g.add_edge(c, a, 9);
        g.add_edge(a, d, 14);
        g.add_edge(b, c, 10);
        g.add_edge(d, c, 2);
        g.add_edge(d, e, 9);
        g.add_edge(b, f, 15);
        g.add_edge(c, f, 11);
        g.add_edge(e, f, 6);

        let path: Result<Option<(i32, Vec<NodeIndex>)>> = astar(
            &g,
            a,
            |finish| Ok(finish == e),
            |e| Ok(*e.weight()),
            |_| Ok(0),
        );
        assert_eq!(path.unwrap(), Some((23, vec![a, d, e])));

        // check against dijkstra
        let dijkstra_run: Result<DictMap<NodeIndex, i32>> =
            dijkstra(&g, a, Some(e), |e| Ok(*e.weight()), None);
        assert_eq!(dijkstra_run.unwrap()[&e], 23);

        let path: Result<Option<(i32, Vec<NodeIndex>)>> = astar(
            &g,
            e,
            |finish| Ok(finish == b),
            |e| Ok(*e.weight()),
            |_| Ok(0),
        );
        assert_eq!(path.unwrap(), None);
    }

    #[test]
    fn test_astar_manhattan_heuristic() {
        let mut g = Graph::new();
        let a = g.add_node((0., 0.));
        let b = g.add_node((2., 0.));
        let c = g.add_node((1., 1.));
        let d = g.add_node((0., 2.));
        let e = g.add_node((3., 3.));
        let f = g.add_node((4., 2.));
        let _ = g.add_node((5., 5.)); // no path to node
        g.add_edge(a, b, 2.);
        g.add_edge(a, d, 4.);
        g.add_edge(b, c, 1.);
        g.add_edge(b, f, 7.);
        g.add_edge(c, e, 5.);
        g.add_edge(e, f, 1.);
        g.add_edge(d, e, 1.);

        let heuristic_for = |f: NodeIndex| {
            let g = &g;
            move |node: NodeIndex| -> Result<f32> {
                let (x1, y1): (f32, f32) = g[node];
                let (x2, y2): (f32, f32) = g[f];

                Ok((x2 - x1).abs() + (y2 - y1).abs())
            }
        };
        let path = astar(
            &g,
            a,
            |finish| Ok(finish == f),
            |e| Ok(*e.weight()),
            heuristic_for(f),
        );
        assert_eq!(path.unwrap(), Some((6., vec![a, d, e, f])));

        let dijkstra_run: Result<DictMap<NodeIndex, f32>> =
            dijkstra(&g, a, None, |e| Ok(*e.weight()), None);

        for end in g.node_indices() {
            let astar_path = astar(
                &g,
                a,
                |finish| Ok(finish == end),
                |e| Ok(*e.weight()),
                heuristic_for(end),
            );
            assert_eq!(
                dijkstra_run.as_ref().unwrap().get(&end).cloned(),
                astar_path.unwrap().map(|t| t.0)
            );
        }
    }

    #[test]
    fn test_astar_runtime_optimal() {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        let d = g.add_node("D");
        let e = g.add_node("E");
        g.add_edge(a, b, 2);
        g.add_edge(a, c, 3);
        g.add_edge(b, d, 3);
        g.add_edge(c, d, 1);
        g.add_edge(d, e, 1);

        let mut times_called = 0;

        let _: Result<Option<(i32, Vec<NodeIndex>)>> = astar(
            &g,
            a,
            |n| Ok(n == e),
            |edge| {
                times_called += 1;
                Ok(*edge.weight())
            },
            |_| Ok(0),
        );

        // A* is runtime optimal in the sense it won't expand more nodes than needed, for the given
        // heuristic. Here, A* should expand, in order: A, B, C, D, E. This should should ask for the
        // costs of edges (A, B), (A, C), (B, D), (C, D), (D, E). Node D will be added to `visit_next`
        // twice, but should only be expanded once. If it is erroneously expanded twice, it will call
        // for (D, E) again and `times_called` will be 6.
        assert_eq!(times_called, 5);
    }
}
