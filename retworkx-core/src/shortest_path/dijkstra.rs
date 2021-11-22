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

use std::hash::Hash;

use petgraph::algo::Measure;
use petgraph::visit::{Control, IntoEdges, Visitable};

use crate::dictmap::*;
use crate::traversal::{dijkstra_search, DijkstraEvent};

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
/// Returns a [`DictMap`] that maps `NodeId` to path cost.
/// # Example
/// ```rust
/// use retworkx_core::petgraph::Graph;
/// use retworkx_core::petgraph::prelude::*;
/// use retworkx_core::dictmap::DictMap;
/// use retworkx_core::shortest_path::dijkstra;
/// use retworkx_core::Result;
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
pub fn dijkstra<G, F, K, E>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    edge_cost: F,
    mut path: Option<&mut DictMap<G::NodeId, Vec<G::NodeId>>>,
) -> Result<DictMap<G::NodeId, K>, E>
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
{
    let mut scores = DictMap::new();
    if let Some(ref mut path) = path {
        path.insert(start, vec![start]);
    }

    dijkstra_search(graph, Some(start), edge_cost, |event| {
        match event {
            DijkstraEvent::Discover(node, score) => {
                scores.insert(node, score);
                if goal.as_ref() == Some(&node) {
                    return Control::Break(());
                }
            }
            DijkstraEvent::EdgeRelaxed(node, next, _) => {
                if let Some(ref mut path) = path {
                    let mut node_path = path[&node].clone();
                    node_path.push(next);
                    path.insert(next, node_path);
                }
            }
            _ => {}
        }

        Control::Continue
    })?;

    Ok(scores)
}
