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

use petgraph::visit::{ControlFlow, EdgeRef, IntoEdges, VisitMap, Visitable};
use std::collections::VecDeque;

use super::try_control;

/// A breadth first search (BFS) visitor event.
#[derive(Copy, Clone, Debug)]
pub enum BfsEvent<N, E> {
    Discover(N),
    /// An edge of the tree formed by the traversal.
    TreeEdge(N, N, E),
    /// An edge that does not belong to the tree.
    NonTreeEdge(N, N, E),
    /// For an edge *(u, v)*, if node *v* is currently in the queue
    /// at the time of examination, then it is a gray-target edge.
    GrayTargetEdge(N, N, E),
    /// For an edge *(u, v)*, if node *v* has been removed from the queue
    /// at the time of examination, then it is a black-target edge.
    BlackTargetEdge(N, N, E),
    /// All edges from a node have been reported.
    Finish(N),
}

/// An iterative breadth first search.
///
/// Starting points are the nodes in the iterator `starts` (specify just one
/// start vertex *x* by using `Some(x)`).
///
/// The traversal emits discovery and finish events for each reachable vertex,
/// and edge classification of each reachable edge. `visitor` is called for each
/// event, see [`BfsEvent`] for possible values.
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
/// The pseudo-code for the BFS algorithm is listed below, with the annotated
/// event points, for which the given visitor object will be called with the
/// appropriate method.
///
/// ```norust
/// BFS(G, s)
///   for each vertex u in V
///       color[u] := WHITE
///   end for
///   color[s] := GRAY
///   EQUEUE(Q, s)                             discover vertex s
///   while (Q != Ø)
///       u := DEQUEUE(Q)
///       for each vertex v in Adj[u]          (u,v) is a tree edge
///           if (color[v] = WHITE)
///               color[v] = GRAY
///           else                             (u,v) is a non - tree edge
///               if (color[v] = GRAY)         (u,v) has a gray target
///                   ...
///               else if (color[v] = BLACK)   (u,v) has a black target
///                   ...
///       end for
///       color[u] := BLACK                    finish vertex u
///   end while
/// ```
///
/// # Example returning [`Control`](petgraph::visit::Control).
///
/// Find a path from vertex 0 to 5, and exit the visit as soon as we reach
/// the goal vertex.
///
/// ```
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::petgraph::graph::node_index as n;
/// use rustworkx_core::petgraph::visit::Control;
///
/// use rustworkx_core::traversal::{BfsEvent, breadth_first_search};
///
/// let gr: Graph<(), ()> = Graph::from_edges(&[
///     (0, 1), (0, 2), (0, 3),
///     (1, 3),
///     (2, 3), (2, 4),
///     (4, 0), (4, 5),
/// ]);
///
/// // record each predecessor, mapping node → node
/// let mut predecessor = vec![NodeIndex::end(); gr.node_count()];
/// let start = n(0);
/// let goal = n(5);
/// breadth_first_search(&gr, Some(start), |event| {
///     if let BfsEvent::TreeEdge(u, v, _) = event {
///         predecessor[v.index()] = u;
///         if v == goal {
///             return Control::Break(v);
///         }
///     }
///     Control::Continue
/// });
///
/// let mut next = goal;
/// let mut path = vec![next];
/// while next != start {
///     let pred = predecessor[next.index()];
///     path.push(pred);
///     next = pred;
/// }
/// path.reverse();
/// assert_eq!(&path, &[n(0), n(2), n(4), n(5)]);
/// ```
///
/// # Example returning a `Result`.
/// ```
/// use rustworkx_core::petgraph::graph::node_index as n;
/// use rustworkx_core::petgraph::prelude::*;
///
/// use rustworkx_core::traversal::{BfsEvent, breadth_first_search};
///
/// let gr: Graph<(), ()> = Graph::from_edges(&[(0, 1), (1, 2), (1, 1), (2, 1)]);
/// let start = n(0);
/// let mut non_tree_edges = 0;
///
/// #[derive(Debug)]
/// struct NonTreeEdgeFound {
///     source: NodeIndex,
///     target: NodeIndex,
/// }
///
/// // Stop the search, the first time a BackEdge is encountered.
/// let result = breadth_first_search(&gr, Some(start), |event| {
///     match event {
///         BfsEvent::NonTreeEdge(u, v, _) => {
///             non_tree_edges += 1;
///             // the implementation of ControlFlow for Result,
///             // treats this Err value as Continue::Break
///             Err(NonTreeEdgeFound {source: u, target: v})
///         }
///         // In the cases where Ok(()) is returned,
///         // Result falls back to the implementation of Control on the value ().
///         // In the case of (), this is to always return Control::Continue.
///         // continuing the search.
///         _ => Ok(()),
///     }
/// });
///
/// assert_eq!(non_tree_edges, 1);
/// println!("number of non-tree edges encountered: {}", non_tree_edges);
/// println!("non-tree edge: ({:?})", result.unwrap_err());
/// ```
pub fn breadth_first_search<G, I, F, C>(graph: G, starts: I, mut visitor: F) -> C
where
    G: IntoEdges + Visitable,
    I: IntoIterator<Item = G::NodeId>,
    F: FnMut(BfsEvent<G::NodeId, &G::EdgeWeight>) -> C,
    C: ControlFlow,
{
    let discovered = &mut graph.visit_map();
    let finished = &mut graph.visit_map();

    for start in starts {
        // `bfs_visitor` returns a "signal" to either continue or exit early
        // but it never "prunes", so we use `unreachable`.
        try_control!(
            bfs_visitor(graph, start, &mut visitor, discovered, finished),
            unreachable!()
        );
    }
    C::continuing()
}

fn bfs_visitor<G, F, C>(
    graph: G,
    u: G::NodeId,
    visitor: &mut F,
    discovered: &mut G::Map,
    finished: &mut G::Map,
) -> C
where
    G: IntoEdges + Visitable,
    F: FnMut(BfsEvent<G::NodeId, &G::EdgeWeight>) -> C,
    C: ControlFlow,
{
    if !discovered.visit(u) {
        return C::continuing();
    }

    try_control!(visitor(BfsEvent::Discover(u)), {}, {
        let mut stack: VecDeque<G::NodeId> = VecDeque::new();
        stack.push_front(u);

        while let Some(u) = stack.pop_front() {
            for edge in graph.edges(u) {
                let v = edge.target();
                if !discovered.is_visited(&v) {
                    try_control!(visitor(BfsEvent::TreeEdge(u, v, edge.weight())), continue);
                    discovered.visit(v);
                    try_control!(visitor(BfsEvent::Discover(v)), continue);
                    stack.push_back(v);
                } else {
                    // non - tree edge.
                    try_control!(
                        visitor(BfsEvent::NonTreeEdge(u, v, edge.weight())),
                        continue
                    );

                    if !finished.is_visited(&v) {
                        try_control!(
                            visitor(BfsEvent::GrayTargetEdge(u, v, edge.weight())),
                            continue
                        );
                    } else {
                        try_control!(
                            visitor(BfsEvent::BlackTargetEdge(u, v, edge.weight())),
                            continue
                        );
                    }
                }
            }

            let first_finish = finished.visit(u);
            debug_assert!(first_finish);
            try_control!(
                visitor(BfsEvent::Finish(u)),
                panic!("Pruning on the `BfsEvent::Finish` is not supported!")
            );
        }
    });

    C::continuing()
}
