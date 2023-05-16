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

use crate::dictmap::*;
use hashbrown::{HashMap, HashSet};
use std::cmp::Reverse;

use std::hash::Hash;

use petgraph::visit::{
    EdgeCount, EdgeRef, IntoEdgeReferences, IntoEdges, IntoNeighborsDirected, IntoNodeIdentifiers,
    NodeCount, NodeIndexable, Visitable,
};

use rayon::prelude::*;

pub fn greedy_color<G>(graph: G) -> DictMap<G::NodeId, usize>
where
    G: NodeCount
        + EdgeCount
        + IntoEdges
        + Visitable
        + NodeIndexable
        + IntoNeighborsDirected
        + IntoNodeIdentifiers
        + IntoEdgeReferences,
    G::NodeId: Hash + Eq + Send + Sync,
{
    let mut colors: DictMap<G::NodeId, usize> = DictMap::new();
    let mut node_vec: Vec<G::NodeId> = graph.node_identifiers().collect();

    let mut sort_map: HashMap<G::NodeId, usize> = HashMap::with_capacity(graph.node_count());
    for k in node_vec.iter() {
        sort_map.insert(*k, graph.edges(*k).count());
    }
    node_vec.par_sort_by_key(|k| Reverse(sort_map.get(k)));

    for node in node_vec {
        let mut neighbor_colors: HashSet<usize> = HashSet::new();
        for edge in graph.edges(node) {
            let target = edge.target();
            let existing_color = match colors.get(&target) {
                Some(color) => color,
                None => continue,
            };
            neighbor_colors.insert(*existing_color);
        }
        let mut current_color: usize = 0;
        loop {
            if !neighbor_colors.contains(&current_color) {
                break;
            }
            current_color += 1;
        }
        colors.insert(node, current_color);
    }

    colors
}
