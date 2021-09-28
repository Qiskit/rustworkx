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

use pyo3::prelude::*;

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::{
    Control, ControlFlow, DfsEvent, IntoNeighbors, Time, VisitMap, Visitable,
};

use crate::PruneSearch;

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

#[derive(FromPyObject)]
pub struct PyDfsVisitor {
    discover_vertex: PyObject,
    finish_vertex: PyObject,
    tree_edge: PyObject,
    back_edge: PyObject,
    forward_or_cross_edge: PyObject,
}

pub fn handler(
    py: Python,
    vis: &PyDfsVisitor,
    event: DfsEvent<NodeIndex>,
) -> PyResult<Control<()>> {
    let res = match event {
        DfsEvent::Discover(u, Time(t)) => {
            vis.discover_vertex.call1(py, (u.index(), t))
        }
        DfsEvent::TreeEdge(u, v) => {
            let edge = (u.index(), v.index());
            vis.tree_edge.call1(py, (edge,))
        }
        DfsEvent::BackEdge(u, v) => {
            let edge = (u.index(), v.index());
            vis.back_edge.call1(py, (edge,))
        }
        DfsEvent::CrossForwardEdge(u, v) => {
            let edge = (u.index(), v.index());
            vis.forward_or_cross_edge.call1(py, (edge,))
        }
        DfsEvent::Finish(u, Time(t)) => {
            vis.finish_vertex.call1(py, (u.index(), t))
        }
    };

    match res {
        Err(e) => {
            if e.is_instance::<PruneSearch>(py) {
                Ok(Control::Prune)
            } else {
                Err(e)
            }
        }
        Ok(_) => Ok(Control::Continue),
    }
}
