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

// This module was originally copied and forked from the upstream petgraph
// repository, specifically:
// https://github.com/petgraph/petgraph/blob/0.5.1/src/dijkstra.rs
// this was necessary to modify the error handling to allow python callables
// to be use for the input functions for edge_cost and return any exceptions
// raised in Python instead of panicking

use std::collections::BinaryHeap;
use std::hash::Hash;

use hashbrown::hash_map::Entry::{Occupied, Vacant};
use hashbrown::HashMap;

use petgraph::algo::Measure;
use petgraph::visit::{EdgeRef, IntoEdges, VisitMap, Visitable};

use pyo3::prelude::*;

use super::astar::MinScored;

/// \[Generic\] Dijkstra's shortest path algorithm.
///
/// Compute the length of the shortest path from `start` to every reachable
/// node.
///
/// The graph should be `Visitable` and implement `IntoEdges`. The function
/// `edge_cost` should return the cost for a particular edge, which is used
/// to compute path costs. Edge costs must be non-negative.
///
/// If `goal` is not `None`, then the algorithm terminates once the `goal` node's
/// cost is calculated.
///
/// If `path` is not `None`, then the algorithm will mutate the input
/// hashbrown::HashMap to insert an entry where the index is the dest node index
/// the value is a Vec of node indices of the path starting with `start` and
/// ending at the index.
///
/// Returns a `HashMap` that maps `NodeId` to path cost.
/// # Example
/// ```rust
/// use petgraph::Graph;
/// use petgraph::algo::dijkstra;
/// use petgraph::prelude::*;
/// use std::collections::HashMap;
///
/// let mut graph : Graph<(),(),Directed>= Graph::new();
/// let a = graph.add_node(()); // node with no weight
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// let d = graph.add_node(());
/// let e = graph.add_node(());
/// let f = graph.add_node(());
/// let g = graph.add_node(());
/// let h = graph.add_node(());
/// // z will be in another connected component
/// let z = graph.add_node(());
///
/// graph.extend_with_edges(&[
///     (a, b),
///     (b, c),
///     (c, d),
///     (d, a),
///     (e, f),
///     (b, e),
///     (f, g),
///     (g, h),
///     (h, e)
/// ]);
/// // a ----> b ----> e ----> f
/// // ^       |       ^       |
/// // |       v       |       v
/// // d <---- c       h <---- g
///
/// let expected_res: HashMap<NodeIndex, usize> = [
///      (a, 3),
///      (b, 0),
///      (c, 1),
///      (d, 2),
///      (e, 1),
///      (f, 2),
///      (g, 3),
///      (h, 4)
///     ].iter().cloned().collect();
/// let res = dijkstra(&graph,b,None, |_| 1);
/// assert_eq!(res, expected_res);
/// // z is not inside res because there is not path from b to z.
/// ```
pub fn dijkstra<G, F, K>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    mut edge_cost: F,
    mut path: Option<&mut HashMap<G::NodeId, Vec<G::NodeId>>>,
) -> PyResult<HashMap<G::NodeId, K>>
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> PyResult<K>,
    K: Measure + Copy,
{
    let mut visited = graph.visit_map();
    let mut scores = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();
    scores.insert(start, zero_score);
    visit_next.push(MinScored(zero_score, start));
    if path.is_some() {
        path.as_mut().unwrap().insert(start, vec![start]);
    }
    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        if goal.as_ref() == Some(&node) {
            break;
        }
        for edge in graph.edges(node) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }
            let cost = edge_cost(edge)?;
            let next_score = node_score + cost;
            match scores.entry(next) {
                Occupied(ent) => {
                    if next_score < *ent.get() {
                        *ent.into_mut() = next_score;
                        visit_next.push(MinScored(next_score, next));
                        if path.is_some() {
                            let mut node_path = path
                                .as_mut()
                                .unwrap()
                                .get(&node)
                                .unwrap()
                                .clone();
                            node_path.push(next);
                            path.as_mut().unwrap().entry(next).and_modify(
                                |new_vec| {
                                    *new_vec = node_path;
                                },
                            );
                        }
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    visit_next.push(MinScored(next_score, next));
                    if path.is_some() {
                        let mut node_path =
                            path.as_mut().unwrap().get(&node).unwrap().clone();
                        node_path.push(next);
                        path.as_mut().unwrap().entry(next).or_insert(node_path);
                    }
                }
            }
        }
        visited.visit(node);
    }
    Ok(scores)
}
