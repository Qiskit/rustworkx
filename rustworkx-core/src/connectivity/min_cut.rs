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

use num_traits::Zero;
use std::ops::AddAssign;

use priority_queue::PriorityQueue;

use petgraph::{
    graph::IndexType,
    visit::{EdgeRef, GraphProp, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
    Undirected,
};

use crate::disjoint_set::DisjointSet;

type StCut<K, T> = Option<((T, T), K)>;
type MinCut<K, T> = Option<(K, Vec<T>)>;

fn zip<T, U>(a: Option<T>, b: Option<U>) -> Option<(T, U)> {
    match (a, b) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

fn stoer_wagner_phase<G, F, K>(
    graph: G,
    edge_cost: &mut F,
    clusters: &DisjointSet<G::NodeId>,
) -> StCut<K, G::NodeId>
where
    G: GraphProp<EdgeType = Undirected> + IntoEdges + IntoNodeIdentifiers,
    G::NodeId: IndexType,
    F: FnMut(G::EdgeRef) -> K,
    K: Copy + Ord + Zero + AddAssign,
{
    let mut pq = PriorityQueue::<G::NodeId, K, ahash::RandomState>::from(
        graph
            .node_identifiers()
            .filter_map(|nx| {
                if clusters.is_root(nx) {
                    Some((nx, K::zero()))
                } else {
                    None
                }
            })
            .collect::<Vec<(G::NodeId, K)>>(),
    );

    let mut cut_w = None;
    let (mut s, mut t) = (None, None);
    while let Some((nx, nx_val)) = pq.pop() {
        s = t;
        t = Some(nx);
        cut_w = Some(nx_val);
        for nx_equiv in clusters.set(nx) {
            for edge in graph.edges(nx_equiv) {
                let target = clusters.find(edge.target());
                pq.change_priority_by(&target, |p| {
                    *p += edge_cost(edge);
                })
            }
        }
    }

    zip(zip(s, t), cut_w)
}

/// Stoer-Wagner's min cut algorithm.
///
/// Compute a weighted minimum cut using the Stoer-Wagner algorithm [`stoer_simple_1997`](https://dl.acm.org/doi/10.1145/263867.263872).
///
/// The graph should be undirected. If the input graph is disconnected,
/// a cut with zero value will be returned. For graphs with less than
/// two nodes, this function returns [`None`]. The function `edge_cost`
/// should return the cost for a particular edge. Edge costs must be non-negative.
///
/// Returns a tuple containing the value of a minimum cut and a vector
/// of all `NodeId`s contained in one part of the partition that defines a minimum cut.
///
/// # Example
/// ```rust
/// use std::collections::HashSet;
/// use std::iter::FromIterator;
///
/// use retworkx_core::connectivity::stoer_wagner_min_cut;
/// use retworkx_core::petgraph::graph::{NodeIndex, UnGraph};
///
/// let mut graph : UnGraph<(), ()> = UnGraph::new_undirected();
/// let a = graph.add_node(()); // node with no weight
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// let d = graph.add_node(());
/// let e = graph.add_node(());
/// let f = graph.add_node(());
/// let g = graph.add_node(());
/// let h = graph.add_node(());
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
/// // a ---- b ---- e ---- f
/// // |      |      |      |
/// // d ---- c      h ---- g
///
/// let (min_cut, partition): (usize, Vec<_>) =
///     stoer_wagner_min_cut(&graph, |_| 1).unwrap();
/// assert_eq!(min_cut, 1);
/// assert_eq!(
///     HashSet::<NodeIndex>::from_iter(partition),
///     HashSet::from_iter([e, f, g, h])
/// );
/// ```
pub fn stoer_wagner_min_cut<G, F, K>(graph: G, mut edge_cost: F) -> MinCut<K, G::NodeId>
where
    G: GraphProp<EdgeType = Undirected>
        + IntoEdges
        + IntoNodeIdentifiers
        + NodeCount
        + NodeIndexable,
    G::NodeId: IndexType,
    F: FnMut(G::EdgeRef) -> K,
    K: Copy + Ord + Zero + AddAssign,
{
    let (mut min_cut, mut min_cut_val) = (None, None);
    let mut clusters = DisjointSet::<G::NodeId>::new(graph.node_bound());
    for _ in 1..graph.node_count() {
        if let Some(((s, t), cut_w)) = stoer_wagner_phase(graph, &mut edge_cost, &clusters) {
            if min_cut_val.is_none() || Some(cut_w) < min_cut_val {
                min_cut = Some(clusters.set(t));
                min_cut_val = Some(cut_w);
            }
            // now merge nodes ``s`` and  ``t``.
            clusters.union(s, t);
        }
    }

    zip(min_cut_val, min_cut)
}
