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

use hashbrown::HashMap;
use num_traits::Zero;
use std::{hash::Hash, ops::AddAssign};

use priority_queue::PriorityQueue;

use petgraph::{
    stable_graph::StableUnGraph,
    visit::{Bfs, EdgeCount, EdgeRef, GraphProp, IntoEdges, IntoNodeIdentifiers, NodeCount},
    Undirected,
};

type StCut<K, T> = Option<((T, T), K)>;
type MinCut<K, T, E> = Result<Option<(K, Vec<T>)>, E>;

fn zip<T, U>(a: Option<T>, b: Option<U>) -> Option<(T, U)> {
    match (a, b) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

fn stoer_wagner_phase<G, F, K>(graph: G, mut edge_cost: F) -> StCut<K, G::NodeId>
where
    G: GraphProp<EdgeType = Undirected> + IntoEdges + IntoNodeIdentifiers,
    G::NodeId: Hash + Eq,
    F: FnMut(G::EdgeRef) -> K,
    K: Copy + Ord + Zero + AddAssign,
{
    let mut pq = PriorityQueue::<G::NodeId, K, ahash::RandomState>::from(
        graph
            .node_identifiers()
            .map(|nx| (nx, K::zero()))
            .collect::<Vec<(G::NodeId, K)>>(),
    );

    let mut cut_w = None;
    let (mut s, mut t) = (None, None);
    while let Some((nx, nx_val)) = pq.pop() {
        s = t;
        t = Some(nx);
        cut_w = Some(nx_val);
        for edge in graph.edges(nx) {
            pq.change_priority_by(&edge.target(), |x| {
                *x += edge_cost(edge);
            });
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
/// use rustworkx_core::connectivity::stoer_wagner_min_cut;
/// use rustworkx_core::petgraph::graph::{NodeIndex, UnGraph};
/// use rustworkx_core::Result;
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
/// let min_cut_res: Result<Option<(usize, Vec<_>)>> =
///     stoer_wagner_min_cut(&graph, |_| Ok(1));
///
/// let (min_cut, partition) = min_cut_res.unwrap().unwrap();
/// assert_eq!(min_cut, 1);
/// assert_eq!(
///     HashSet::<NodeIndex>::from_iter(partition),
///     HashSet::from_iter([e, f, g, h])
/// );
/// ```
pub fn stoer_wagner_min_cut<G, F, K, E>(graph: G, mut edge_cost: F) -> MinCut<K, G::NodeId, E>
where
    G: GraphProp<EdgeType = Undirected> + IntoEdges + IntoNodeIdentifiers + NodeCount + EdgeCount,
    G::NodeId: Hash + Eq,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Copy + Ord + Zero + AddAssign,
{
    let mut graph_with_super_nodes =
        StableUnGraph::with_capacity(graph.node_count(), graph.edge_count());

    let mut node_map = HashMap::with_capacity(graph.node_count());
    let mut rev_node_map = HashMap::with_capacity(graph.node_count());

    for node in graph.node_identifiers() {
        let index = graph_with_super_nodes.add_node(());
        node_map.insert(node, index);
        rev_node_map.insert(index, node);
    }

    for edge in graph.edge_references() {
        let cost = edge_cost(edge)?;
        let source = node_map[&edge.source()];
        let target = node_map[&edge.target()];
        graph_with_super_nodes.add_edge(source, target, cost);
    }

    if graph_with_super_nodes.node_count() == 0 {
        return Ok(None);
    }

    let (mut best_phase, mut min_cut_val) = (None, None);

    let mut contractions = Vec::new();
    for phase in 0..(graph_with_super_nodes.node_count() - 1) {
        if let Some(((s, t), cut_w)) =
            stoer_wagner_phase(&graph_with_super_nodes, |edge| *edge.weight())
        {
            if min_cut_val.is_none() || Some(cut_w) < min_cut_val {
                best_phase = Some(phase);
                min_cut_val = Some(cut_w);
            }
            // now merge nodes ``s`` and  ``t``.
            contractions.push((s, t));
            let edges = graph_with_super_nodes
                .edges(t)
                .map(|edge| (s, edge.target(), *edge.weight()))
                .collect::<Vec<_>>();
            for (source, target, cost) in edges {
                if let Some(edge_index) = graph_with_super_nodes.find_edge(source, target) {
                    graph_with_super_nodes[edge_index] += cost;
                } else {
                    graph_with_super_nodes.add_edge(source, target, cost);
                }
            }
            graph_with_super_nodes.remove_node(t);
        }
    }

    // Recover the optimal partitioning from the contractions
    let min_cut = best_phase.map(|phase| {
        let mut clustered_graph = StableUnGraph::<(), ()>::default();
        clustered_graph.extend_with_edges(&contractions[..phase]);

        let node = contractions[phase].1;
        if clustered_graph.contains_node(node) {
            let mut cluster = Vec::new();
            let mut bfs = Bfs::new(&clustered_graph, node);
            while let Some(nx) = bfs.next(&clustered_graph) {
                cluster.push(rev_node_map[&nx])
            }
            cluster
        } else {
            vec![rev_node_map[&node]]
        }
    });

    Ok(zip(min_cut_val, min_cut))
}
