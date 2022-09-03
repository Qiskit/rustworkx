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
// https://github.com/petgraph/petgraph/blob/master/src/k_shortest_path.rs
// prior to the 0.6.0 release. This was necessary to modify the error handling
// to allow python callables to be use for the input functions for edge_cost
// and return any exceptions raised in Python instead of panicking

use std::collections::BinaryHeap;
use std::hash::Hash;

use petgraph::algo::Measure;
use petgraph::visit::{
    EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable, Visitable,
};

use crate::distancemap::DistanceMap;
use crate::min_scored::MinScored;

/// k'th shortest path algorithm.
///
/// Compute the length of the k'th shortest path from `start` to every reachable
/// node.
///
/// The graph should be [`Visitable`] and implement [`IntoEdges`]. The function
/// `edge_cost` should return the cost for a particular edge, which is used
/// to compute path costs. Edge costs must be non-negative.
///
/// If `goal` is not [`None`], then the algorithm terminates once the `goal` node's
/// cost is calculated.
///
/// Computes in **O(k * (|E| + |V|*log(|V|)))** time (average).
///
/// Returns a [`DistanceMap`] that maps `NodeId` to path cost as the value.
///
/// # Example:
/// ```rust
///
/// use rustworkx_core::petgraph;
/// use rustworkx_core::petgraph::graph::NodeIndex;
/// use rustworkx_core::shortest_path::k_shortest_path;
/// use hashbrown::HashMap;
/// use rustworkx_core::Result;
///
/// let g = petgraph::graph::UnGraph::<i32, _>::from_edges(&[
///     (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (1, 4), (5, 6), (6, 7), (7, 5)
/// ]);
///
/// let res: Result<HashMap<NodeIndex, f64>> = k_shortest_path(
///     &g, NodeIndex::new(1), None, 2,
///     |e: rustworkx_core::petgraph::graph::EdgeReference<&'static str>| Ok(1.0),
/// );
///
/// let output = res.unwrap();
/// let expected: HashMap<NodeIndex, f64> = [
///     (NodeIndex::new(0), 3.0),
///     (NodeIndex::new(1), 2.0),
///     (NodeIndex::new(2), 3.0),
///     (NodeIndex::new(3), 2.0),
///     (NodeIndex::new(4), 3.0),
///     (NodeIndex::new(5), 4.0),
///     (NodeIndex::new(6), 4.0),
///     (NodeIndex::new(7), 4.0),
/// ].iter().cloned().collect();
/// assert_eq!(expected, output);
/// ```
pub fn k_shortest_path<G, F, E, K, S>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    k: usize,
    mut edge_cost: F,
) -> Result<S, E>
where
    G: IntoEdges + Visitable + NodeCount + NodeIndexable + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
    S: DistanceMap<G::NodeId, K>,
{
    let mut counter: Vec<usize> = vec![0; graph.node_bound()];
    let mut scores: S = S::build(graph.node_bound());
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();

    visit_next.push(MinScored(zero_score, start));

    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        counter[graph.to_index(node)] += 1;
        let current_counter = counter[graph.to_index(node)];

        if current_counter > k {
            continue;
        }

        if current_counter == k {
            scores.put_item(node, node_score);
        }

        //Already reached goal k times
        if goal.as_ref() == Some(&node) && current_counter == k {
            break;
        }

        for edge in graph.edges(node) {
            visit_next.push(MinScored(node_score + edge_cost(edge)?, edge.target()));
        }
    }

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use crate::shortest_path::k_shortest_path;
    use crate::Result;
    use petgraph::graph::NodeIndex;
    use petgraph::Graph;

    #[test]
    fn test_second_shortest_path() {
        let mut graph: Graph<(), _> = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        let f = graph.add_node(());
        let g = graph.add_node(());
        let h = graph.add_node(());
        let i = graph.add_node(());
        let j = graph.add_node(());
        let k = graph.add_node(());
        let l = graph.add_node(());
        let m = graph.add_node(());

        graph.extend_with_edges(&[
            (a, b),
            (b, c),
            (c, d),
            (b, f),
            (f, g),
            (c, g),
            (g, h),
            (d, e),
            (e, h),
            (h, i),
            (h, j),
            (h, k),
            (h, l),
            (i, m),
            (l, k),
            (j, k),
            (j, m),
            (k, m),
            (l, m),
            (m, e),
        ]);

        let res: Result<HashMap<NodeIndex, usize>> = k_shortest_path(
            &graph,
            a,
            None,
            2,
            |_: petgraph::graph::EdgeReference<&'static str>| Ok(1),
        );

        let expected_res: HashMap<NodeIndex, usize> = [
            (e, 7),
            (g, 3),
            (h, 4),
            (i, 5),
            (j, 5),
            (k, 5),
            (l, 5),
            (m, 6),
        ]
        .iter()
        .cloned()
        .collect();

        assert_eq!(res.unwrap(), expected_res);
    }
}
