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
// this was necessary to modify the error handling to allow python callables
// to be use for the input functions for edge_cost and return any exceptions
// raised in Python instead of panicking

use std::collections::BinaryHeap;
use std::hash::Hash;

use hashbrown::HashMap;

use petgraph::visit::{
    Data, EdgeRef, IntoEdges, NodeCount, NodeIndexable, Visitable,
};

use pyo3::prelude::*;

use super::astar::MinScored;

/// k'th shortest path algorithm.
///
/// Compute the length of the k'th shortest path from `start` to every reachable
/// node.
///
/// The graph should be `Visitable` and implement `IntoEdges`. The function
/// `edge_cost` should return the cost for a particular edge, which is used
/// to compute path costs. Edge costs must be non-negative.
///
/// If `goal` is not `None`, then the algorithm terminates once the `goal` node's
/// cost is calculated.
///
/// Computes in **O(k * (|E| + |V|*log(|V|)))** time (average).
///
/// Returns a `HashMap` that maps `NodeId` to path cost.
pub fn k_shortest_path<G, F>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    k: usize,
    mut edge_cost: F,
) -> PyResult<HashMap<G::NodeId, f64>>
where
    G: IntoEdges + Visitable + NodeCount + NodeIndexable,
    G: Data<NodeWeight = PyObject, EdgeWeight = PyObject>,
    G::NodeId: Eq + Hash,
    F: FnMut(&PyObject) -> PyResult<f64>,
{
    let mut counter: Vec<usize> = vec![0; graph.node_count()];
    let mut scores = HashMap::with_capacity(graph.node_count());
    let mut visit_next = BinaryHeap::new();
    let zero_score = 0.0;

    visit_next.push(MinScored(zero_score, start));

    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        counter[graph.to_index(node)] += 1;
        let current_counter = counter[graph.to_index(node)];

        if current_counter > k {
            continue;
        }

        if current_counter == k {
            scores.insert(node, node_score);
        }

        //Already reached goal k times
        if goal.as_ref() == Some(&node) && current_counter == k {
            break;
        }

        for edge in graph.edges(node) {
            visit_next.push(MinScored(
                node_score + edge_cost(edge.weight())?,
                edge.target(),
            ));
        }
    }
    Ok(scores)
}
