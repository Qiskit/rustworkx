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

// This module was forked from petgraph:
//
// https://github.com/petgraph/petgraph/blob/9ff688872b467d3e1b5adef19f5c52f519d3279c/src/algo/simple_paths.rs
//
// to add support for returning all simple paths to a list of targets instead
// of just between a single node pair.

use hashbrown::HashSet;
use indexmap::map::Entry;
use indexmap::IndexSet;
use petgraph::visit::{IntoNeighborsDirected, NodeCount};
use petgraph::Direction::Outgoing;
use std::{hash::Hash, iter::FromIterator};

use crate::dictmap::*;

/// Returns an iterator that produces all simple paths from `from` node to `to`, which contains at least `min_intermediate_nodes` nodes
/// and at most `max_intermediate_nodes`, if given, or limited by the graph's order otherwise. The simple path is a path without repetitions.
///
/// This algorithm is adapted from <https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.all_simple_paths.html>.
///
/// # Example
/// ```
/// use petgraph::prelude::*;
/// use hashbrown::HashSet;
/// use retworkx_core::connectivity::all_simple_paths_multiple_targets;
///
/// let mut graph = DiGraph::<&str, i32>::new();
///
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// let d = graph.add_node("d");
///
/// graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (a, b, 1), (b, d, 1)]);
///
/// let mut to_set = HashSet::new();
/// to_set.insert(d);
///
/// let ways = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, None);
///
/// let d_path = ways.get(&d).unwrap();
/// assert_eq!(4, d_path.len());
/// ```
pub fn all_simple_paths_multiple_targets<G>(
    graph: G,
    from: G::NodeId,
    to: &HashSet<G::NodeId>,
    min_intermediate_nodes: usize,
    max_intermediate_nodes: Option<usize>,
) -> DictMap<G::NodeId, Vec<Vec<G::NodeId>>>
where
    G: NodeCount,
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    // how many nodes are allowed in simple path up to target node
    // it is min/max allowed path length minus one, because it is more appropriate when implementing lookahead
    // than constantly add 1 to length of current path
    let max_length = if let Some(l) = max_intermediate_nodes {
        l + 1
    } else {
        graph.node_count() - 1
    };

    let min_length = min_intermediate_nodes + 1;

    // list of visited nodes
    let mut visited: IndexSet<G::NodeId> = IndexSet::from_iter(Some(from));
    // list of childs of currently exploring path nodes,
    // last elem is list of childs of last visited node
    let mut stack = vec![graph.neighbors_directed(from, Outgoing)];

    let mut output: DictMap<G::NodeId, Vec<Vec<G::NodeId>>> = DictMap::with_capacity(to.len());

    while let Some(children) = stack.last_mut() {
        if let Some(child) = children.next() {
            if visited.len() < max_length {
                if !visited.contains(&child) {
                    if to.contains(&child) && visited.len() >= min_length {
                        let new_path: Vec<G::NodeId> =
                            visited.iter().copied().chain([child]).collect();
                        match output.entry(child) {
                            Entry::Vacant(e) => {
                                e.insert(vec![new_path]);
                            }
                            Entry::Occupied(mut e) => {
                                e.get_mut().push(new_path);
                            }
                        }
                    }
                    visited.insert(child);
                    stack.push(graph.neighbors_directed(child, Outgoing));
                }
            // visited.len() == max_length
            } else {
                let mut temp: IndexSet<G::NodeId> = children.collect();
                temp.insert(child);
                for c in temp {
                    if to.contains(&c) && !visited.contains(&c) && visited.len() >= min_length {
                        let new_path: Vec<G::NodeId> = visited.iter().cloned().chain([c]).collect();
                        match output.entry(c) {
                            Entry::Vacant(e) => {
                                e.insert(vec![new_path]);
                            }
                            Entry::Occupied(mut e) => {
                                e.get_mut().push(new_path);
                            }
                        }
                    }
                }
                stack.pop();
                visited.pop();
            }
        } else {
            stack.pop();
            visited.pop();
        }
    }
    output
}
