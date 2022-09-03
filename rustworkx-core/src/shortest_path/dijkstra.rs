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

use petgraph::algo::Measure;
use petgraph::visit::{EdgeRef, IntoEdges, NodeIndexable, VisitMap, Visitable};

use crate::dictmap::*;
use crate::distancemap::DistanceMap;
use crate::min_scored::MinScored;

/// Dijkstra's shortest path algorithm.
///
/// Compute the length of the shortest path from `start` to every reachable
/// node.
///
/// The graph should be [`Visitable`] and implement [`IntoEdges`]. The function
/// `edge_cost` should return the cost for a particular edge, which is used
/// to compute path costs. Edge costs must be non-negative.
///
/// If `goal` is not [`None`], then the algorithm terminates once the `goal` node's
/// cost is calculated.
///
/// If `path` is not [`None`], then the algorithm will mutate the input
/// [`DictMap`] to insert an entry where the index is the dest node index
/// the value is a Vec of node indices of the path starting with `start` and
/// ending at the index.
///
/// Returns a [`DistanceMap`] that maps `NodeId` to path cost.
/// # Example
/// ```rust
/// use rustworkx_core::petgraph::Graph;
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::dictmap::DictMap;
/// use rustworkx_core::shortest_path::dijkstra;
/// use rustworkx_core::Result;
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
/// let expected_res: DictMap<NodeIndex, usize> = [
///      (a, 3),
///      (b, 0),
///      (c, 1),
///      (d, 2),
///      (e, 1),
///      (f, 2),
///      (g, 3),
///      (h, 4)
///     ].iter().cloned().collect();
/// let res: Result<DictMap<NodeIndex, usize>> = dijkstra(
///     &graph, b, None, |_| Ok(1), None
/// );
/// assert_eq!(res.unwrap(), expected_res);
/// // z is not inside res because there is not path from b to z.
/// ```
pub fn dijkstra<G, F, K, E, S>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    mut edge_cost: F,
    mut path: Option<&mut DictMap<G::NodeId, Vec<G::NodeId>>>,
) -> Result<S, E>
where
    G: IntoEdges + Visitable + NodeIndexable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
    S: DistanceMap<G::NodeId, K>,
{
    let mut visited = graph.visit_map();
    let mut scores: S = S::build(graph.node_bound());
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();
    scores.put_item(start, zero_score);
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
            match scores.get_item(next) {
                Some(current_score) => {
                    if next_score < *current_score {
                        scores.put_item(next, next_score);
                        visit_next.push(MinScored(next_score, next));
                        if path.is_some() {
                            let mut node_path = path.as_mut().unwrap().get(&node).unwrap().clone();
                            node_path.push(next);
                            path.as_mut().unwrap().entry(next).and_modify(|new_vec| {
                                *new_vec = node_path;
                            });
                        }
                    }
                }
                None => {
                    scores.put_item(next, next_score);
                    visit_next.push(MinScored(next_score, next));
                    if path.is_some() {
                        let mut node_path = path.as_mut().unwrap().get(&node).unwrap().clone();
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

#[cfg(test)]
mod tests {
    use crate::dictmap::DictMap;
    use crate::shortest_path::dijkstra;
    use crate::Result;
    use petgraph::prelude::*;
    use petgraph::Graph;

    #[test]
    fn test_dijk() {
        let mut g = Graph::new_undirected();
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
        println!("{:?}", g);
        let scores: Result<DictMap<NodeIndex, usize>> =
            dijkstra(&g, a, None, |e| Ok(*e.weight()), None);
        let exp_scores: DictMap<NodeIndex, usize> =
            [(a, 0), (b, 7), (c, 9), (d, 11), (e, 20), (f, 20)]
                .iter()
                .cloned()
                .collect();
        assert_eq!(scores.unwrap(), exp_scores);
    }

    #[test]
    fn test_dijk_with_goal() {
        let mut g = Graph::new_undirected();
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
        println!("{:?}", g);

        let scores: Result<DictMap<NodeIndex, usize>> =
            dijkstra(&g, a, Some(c), |e| Ok(*e.weight()), None);
        assert_eq!(scores.unwrap()[&c], 9);
    }
}
