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

use hashbrown::HashSet;
use petgraph::visit::{IntoNeighborsDirected, NodeCount, Visitable};

/// Returns all nodes at a fixed `distance` from `source` in `G`.
/// Args:
///     `graph`:
///     `source`:
///     `distance`:
pub fn descendants_at_distance<G>(graph: G, source: G::NodeId, distance: usize) -> Vec<G::NodeId>
where
    G: Visitable + IntoNeighborsDirected + NodeCount,
    G::NodeId: std::cmp::Eq + std::hash::Hash,
{
    let mut current_layer: Vec<G::NodeId> = vec![source];
    let mut layers: usize = 0;
    let mut visited: HashSet<G::NodeId> = HashSet::with_capacity(graph.node_count());
    visited.insert(source);
    while !current_layer.is_empty() && layers < distance {
        let mut next_layer: Vec<G::NodeId> = Vec::new();
        for node in current_layer {
            for child in graph.neighbors_directed(node, petgraph::Outgoing) {
                if !visited.contains(&child) {
                    visited.insert(child);
                    next_layer.push(child);
                }
            }
        }
        current_layer = next_layer;
        layers += 1;
    }
    current_layer
}
