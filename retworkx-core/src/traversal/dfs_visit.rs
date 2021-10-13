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

// This module is an iterative implementation of the upstream petgraph
// ``depth_first_search`` function.
// https://github.com/petgraph/petgraph/blob/0.6.0/src/visit/dfsvisit.rs

use petgraph::visit::{
    ControlFlow, DfsEvent, IntoNeighbors, Time, VisitMap, Visitable,
};

/// Return if the expression is a break value, execute the provided statement
/// if it is a prune value.
/// https://github.com/petgraph/petgraph/blob/0.6.0/src/visit/dfsvisit.rs#L27
macro_rules! try_control {
    ($e:expr, $p:stmt) => {
        try_control!($e, $p, ());
    };
    ($e:expr, $p:stmt, $q:stmt) => {
        match $e {
            x => {
                if x.should_break() {
                    return x;
                } else if x.should_prune() {
                    $p
                } else {
                    $q
                }
            }
        }
    };
}

/// An iterative depth first search.
///
/// It is an iterative implementation of the upstream petgraph
/// [`depth_first_search`](https://docs.rs/petgraph/0.6.0/petgraph/visit/fn.depth_first_search.html) function.
///
/// Starting points are the nodes in the iterator `starts` (specify just one
/// start vertex *x* by using `Some(x)`).
///
/// The traversal emits discovery and finish events for each reachable vertex,
/// and edge classification of each reachable edge. `visitor` is called for each
/// event, see `petgraph::DfsEvent` for possible values.
///
/// The return value should implement the trait `ControlFlow`, and can be used to change
/// the control flow of the search.
///
/// `Control` Implements `ControlFlow` such that `Control::Continue` resumes the search.
/// `Control::Break` will stop the visit early, returning the contained value.
/// `Control::Prune` will stop traversing any additional edges from the current
/// node and proceed immediately to the `Finish` event.
///
/// There are implementations of `ControlFlow` for `()`, and `Result<C, E>` where
/// `C: ControlFlow`. The implementation for `()` will continue until finished.
/// For `Result`, upon encountering an `E` it will break, otherwise acting the same as `C`.
///
/// ***Panics** if you attempt to prune a node from its `Finish` event.
///
/// # Example returning `Control`.
///
/// Find a path from vertex 0 to 5, and exit the visit as soon as we reach
/// the goal vertex.
///
/// ```
/// use retworkx_core::petgraph::prelude::*;
/// use retworkx_core::petgraph::graph::node_index as n;
/// use retworkx_core::petgraph::visit::depth_first_search;
/// use retworkx_core::petgraph::visit::{DfsEvent, Control};
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
/// depth_first_search(&gr, Some(start), |event| {
///     if let DfsEvent::TreeEdge(u, v) = event {
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
/// use retworkx_core::petgraph::graph::node_index as n;
/// use retworkx_core::petgraph::prelude::*;
/// use retworkx_core::petgraph::visit::depth_first_search;
/// use retworkx_core::petgraph::visit::{DfsEvent, Time};
///
/// let gr: Graph<(), ()> = Graph::from_edges(&[(0, 1), (1, 2), (1, 1), (2, 1)]);
/// let start = n(0);
/// let mut back_edges = 0;
/// let mut discover_time = 0;
/// // Stop the search, the first time a BackEdge is encountered.
/// let result = depth_first_search(&gr, Some(start), |event| {
///     match event {
///         // In the cases where Ok(()) is returned,
///         // Result falls back to the implementation of Control on the value ().
///         // In the case of (), this is to always return Control::Continue.
///         // continuing the search.
///         DfsEvent::Discover(_, Time(t)) => {
///             discover_time = t;
///             Ok(())
///         }
///         DfsEvent::BackEdge(_, _) => {
///             back_edges += 1;
///             // the implementation of ControlFlow for Result,
///             // treats this Err value as Continue::Break
///             Err(event)
///         }
///         _ => Ok(()),
///     }
/// });
///
/// // Even though the graph has more than one cycle,
/// // The number of back_edges visited by the search should always be 1.
/// assert_eq!(back_edges, 1);
/// println!("discover time:{:?}", discover_time);
/// println!("number of backedges encountered: {}", back_edges);
/// println!("back edge: {:?}", result);
/// ```
pub fn depth_first_search<G, I, F, C>(graph: G, starts: I, mut visitor: F) -> C
where
    G: IntoNeighbors + Visitable,
    I: IntoIterator<Item = G::NodeId>,
    F: FnMut(DfsEvent<G::NodeId>) -> C,
    C: ControlFlow,
{
    let time = &mut Time(0);
    let discovered = &mut graph.visit_map();
    let finished = &mut graph.visit_map();

    for start in starts {
        try_control!(
            dfs_visitor(graph, start, &mut visitor, discovered, finished, time),
            unreachable!()
        );
    }
    C::continuing()
}

fn dfs_visitor<G, F, C>(
    graph: G,
    u: G::NodeId,
    visitor: &mut F,
    discovered: &mut G::Map,
    finished: &mut G::Map,
    time: &mut Time,
) -> C
where
    G: IntoNeighbors + Visitable,
    F: FnMut(DfsEvent<G::NodeId>) -> C,
    C: ControlFlow,
{
    if !discovered.visit(u) {
        return C::continuing();
    }

    try_control!(visitor(DfsEvent::Discover(u, time_post_inc(time))), {}, {
        let mut stack: Vec<(G::NodeId, <G as IntoNeighbors>::Neighbors)> =
            Vec::new();
        stack.push((u, graph.neighbors(u)));

        while let Some(elem) = stack.last_mut() {
            let u = elem.0;
            let neighbors = &mut elem.1;
            let mut next = None;

            for v in neighbors {
                if !discovered.is_visited(&v) {
                    try_control!(visitor(DfsEvent::TreeEdge(u, v)), continue);
                    discovered.visit(v);
                    try_control!(
                        visitor(DfsEvent::Discover(v, time_post_inc(time))),
                        continue
                    );
                    next = Some(v);
                    break;
                } else if !finished.is_visited(&v) {
                    try_control!(visitor(DfsEvent::BackEdge(u, v)), continue);
                } else {
                    try_control!(
                        visitor(DfsEvent::CrossForwardEdge(u, v)),
                        continue
                    );
                }
            }

            match next {
                Some(v) => stack.push((v, graph.neighbors(v))),
                None => {
                    let first_finish = finished.visit(u);
                    debug_assert!(first_finish);
                    try_control!(
                        visitor(DfsEvent::Finish(u, time_post_inc(time))),
                        panic!("Pruning on the `DfsEvent::Finish` is not supported!")
                    );
                    stack.pop();
                }
            };
        }
    });

    C::continuing()
}

fn time_post_inc(x: &mut Time) -> Time {
    let v = *x;
    x.0 += 1;
    v
}
