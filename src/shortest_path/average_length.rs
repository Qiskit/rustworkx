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

use petgraph::prelude::*;
use petgraph::EdgeType;

use rayon::prelude::*;

use crate::StablePyGraph;

pub fn compute_distance_sum<Ty: EdgeType + Sync>(
    graph: &StablePyGraph<Ty>,
    parallel_threshold: usize,
    as_undirected: bool,
) -> (usize, usize) {
    let n = graph.node_count();
    let bfs_traversal = |start_index: NodeIndex| -> (usize, usize) {
        let mut seen: HashSet<NodeIndex> = HashSet::with_capacity(n);
        let mut level = 0;
        let mut next_level: HashSet<NodeIndex> = HashSet::new();
        next_level.insert(start_index);
        let mut count: usize = 0;
        let mut conn_pairs: usize = 0;
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<NodeIndex> = Vec::new();
            for v in this_level {
                if seen.insert(v) {
                    found.push(v);
                    count += level;
                }
            }

            conn_pairs += found.len();
            if seen.len() == n {
                break;
            }
            for node in found {
                for v in graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
                    next_level.insert(v);
                }
                if graph.is_directed() && as_undirected {
                    for v in graph.neighbors_directed(node, petgraph::Direction::Incoming) {
                        next_level.insert(v);
                    }
                }
            }
            level += 1
        }
        (count, conn_pairs - 1)
    };
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    if n < parallel_threshold {
        node_indices
            .iter()
            .map(|index| bfs_traversal(*index))
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        node_indices
            .par_iter()
            .map(|index| bfs_traversal(*index))
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }
}
