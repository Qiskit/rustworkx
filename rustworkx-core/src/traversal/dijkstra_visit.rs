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
use petgraph::visit::{ControlFlow, EdgeRef, IntoEdges, VisitMap, Visitable};

use crate::min_scored::MinScored;

use super::try_control;

macro_rules! try_control_with_result {
    ($e:expr, $p:stmt) => {
        try_control_with_result!($e, $p, ());
    };
    ($e:expr, $p:stmt, $q:stmt) => {
        match $e {
            x => {
                if x.should_break() {
                    return Ok(x);
                } else if x.should_prune() {
                    $p
                } else {
                    $q
                }
            }
        }
    };
}

/// A dijkstra search visitor event.
#[derive(Copy, Clone, Debug)]
pub enum DijkstraEvent<N, E, K> {
    /// This is invoked when a vertex is encountered for the first time and
    /// it's popped from the queue. Together with the node, we report the optimal
    /// distance of the node.
    Discover(N, K),
    /// This is invoked on every out-edge of each vertex after it is discovered.
    ExamineEdge(N, N, E),
    /// Upon examination, if the distance of the target of the edge is reduced, this event is emitted.
    EdgeRelaxed(N, N, E),
    /// Upon examination, if the edge is not relaxed, this event is emitted.
    EdgeNotRelaxed(N, N, E),
    /// All edges from a node have been reported.
    Finish(N),
}

/// Dijkstra traversal of a graph.
///
/// Starting points are the nodes in the iterator `starts` (specify just one
/// start vertex *x* by using `Some(x)`).
///
/// The traversal emits discovery and finish events for each reachable vertex,
/// and edge classification of each reachable edge. `visitor` is called for each
/// event, see [`DijkstraEvent`] for possible values.
///
/// The return value should implement the trait [`ControlFlow`], and can be used to change
/// the control flow of the search.
///
/// [`Control`](petgraph::visit::Control) Implements [`ControlFlow`] such that `Control::Continue` resumes the search.
/// `Control::Break` will stop the visit early, returning the contained value.
/// `Control::Prune` will stop traversing any additional edges from the current
/// node and proceed immediately to the `Finish` event.
///
/// There are implementations of [`ControlFlow`] for `()`, and [`Result<C, E>`] where
/// `C: ControlFlow`. The implementation for `()` will continue until finished.
/// For [`Result`], upon encountering an `E` it will break, otherwise acting the same as `C`.
///
/// ***Panics** if you attempt to prune a node from its `Finish` event.
///
/// The pseudo-code for the Dijkstra algorithm is listed below, with the annotated
/// event points, for which the given visitor object will be called with the
/// appropriate method.
///
/// ```norust
/// DIJKSTRA(G, source, weight)
///   for each vertex u in V
///       d[u] := infinity
///       p[u] := u
///   end for
///   d[source] := 0
///   INSERT(Q, source)
///   while (Q != Ø)
///       u := EXTRACT-MIN(Q)                         discover vertex u
///       for each vertex v in Adj[u]                 examine edge (u,v)
///           if (weight[(u,v)] + d[u] < d[v])        edge (u,v) relaxed
///               d[v] := weight[(u,v)] + d[u]
///               p[v] := u
///               DECREASE-KEY(Q, v)
///           else                                    edge (u,v) not relaxed
///               ...
///           if (d[v] was originally infinity)
///               INSERT(Q, v)
///       end for                                     finish vertex u
///   end while
/// ```
///
/// # Example returning [`Control`](petgraph::visit::Control).
///
/// Find the shortest path from vertex 0 to 5, and exit the visit as soon as
/// we reach the goal vertex.
///
/// ```
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::petgraph::graph::node_index as n;
/// use rustworkx_core::petgraph::visit::Control;
///
/// use rustworkx_core::traversal::{DijkstraEvent, dijkstra_search};
///
/// let gr: Graph<(), ()> = Graph::from_edges(&[
///     (0, 1), (0, 2), (0, 3), (0, 4),
///     (1, 3),
///     (2, 3), (2, 4),
///     (4, 5),
/// ]);
///
/// // record each predecessor, mapping node → node
/// let mut predecessor = vec![NodeIndex::end(); gr.node_count()];
/// let start = n(0);
/// let goal = n(5);
/// dijkstra_search(
///     &gr,
///     Some(start),
///     |edge| -> Result<usize, ()> {
///         Ok(1)
///     },
///     |event| {
///         match event {
///             DijkstraEvent::Discover(v, _) => {
///                 if v == goal {
///                     return Control::Break(v);
///                 }   
///             },
///             DijkstraEvent::EdgeRelaxed(u, v, _) => {
///                 predecessor[v.index()] = u;
///             },
///             _ => {}
///         };
///
///         Control::Continue
///     },
/// ).unwrap();
///
/// let mut next = goal;
/// let mut path = vec![next];
/// while next != start {
///     let pred = predecessor[next.index()];
///     path.push(pred);
///     next = pred;
/// }
/// path.reverse();
/// assert_eq!(&path, &[n(0), n(4), n(5)]);
/// ```
pub fn dijkstra_search<G, I, F, K, E, H, C>(
    graph: G,
    starts: I,
    mut edge_cost: F,
    mut visitor: H,
) -> Result<C, E>
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash,
    I: IntoIterator<Item = G::NodeId>,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
    H: FnMut(DijkstraEvent<G::NodeId, &G::EdgeWeight, K>) -> C,
    C: ControlFlow,
{
    let visited = &mut graph.visit_map();

    for start in starts {
        // `dijkstra_visitor` returns a "signal" to either continue or exit early
        // but it never "prunes", so we use `unreachable`.
        try_control!(
            dijkstra_visitor(graph, start, &mut edge_cost, &mut visitor, visited),
            unreachable!()
        );
    }

    Ok(C::continuing())
}

pub fn dijkstra_visitor<G, F, K, E, V, C>(
    graph: G,
    start: G::NodeId,
    mut edge_cost: F,
    mut visitor: V,
    visited: &mut G::Map,
) -> Result<C, E>
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
    V: FnMut(DijkstraEvent<G::NodeId, &G::EdgeWeight, K>) -> C,
    C: ControlFlow,
{
    if visited.is_visited(&start) {
        return Ok(C::continuing());
    }

    let mut scores = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();
    scores.insert(start, zero_score);
    visit_next.push(MinScored(zero_score, start));

    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if !visited.visit(node) {
            continue;
        }

        try_control_with_result!(visitor(DijkstraEvent::Discover(node, node_score)), continue);

        for edge in graph.edges(node) {
            let next = edge.target();
            try_control_with_result!(
                visitor(DijkstraEvent::ExamineEdge(node, next, edge.weight())),
                continue
            );

            if visited.is_visited(&next) {
                continue;
            }

            let cost = edge_cost(edge)?;
            let next_score = node_score + cost;
            match scores.entry(next) {
                Occupied(ent) => {
                    if next_score < *ent.get() {
                        try_control_with_result!(
                            visitor(DijkstraEvent::EdgeRelaxed(node, next, edge.weight())),
                            continue
                        );
                        *ent.into_mut() = next_score;
                        visit_next.push(MinScored(next_score, next));
                    } else {
                        try_control_with_result!(
                            visitor(DijkstraEvent::EdgeNotRelaxed(node, next, edge.weight())),
                            continue
                        );
                    }
                }
                Vacant(ent) => {
                    try_control_with_result!(
                        visitor(DijkstraEvent::EdgeRelaxed(node, next, edge.weight())),
                        continue
                    );
                    ent.insert(next_score);
                    visit_next.push(MinScored(next_score, next));
                }
            }
        }

        try_control_with_result!(
            visitor(DijkstraEvent::Finish(node)),
            panic!("Pruning on the `DijkstraEvent::Finish` is not supported!")
        );
    }

    Ok(C::continuing())
}
