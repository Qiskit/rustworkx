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

use hashbrown::{HashMap, HashSet};

use petgraph::prelude::*;
use petgraph::EdgeType;
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn compute_distance_sum<Ty: EdgeType + Sync>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
    parallel_threshold: usize,
    as_undirected: bool,
) -> usize {
    let n = graph.node_count();
    let bfs_traversal = |start_index: NodeIndex| -> usize {
        let mut seen: HashMap<NodeIndex, usize> = HashMap::with_capacity(n);
        let mut level = 0;
        let mut next_level: HashSet<NodeIndex> = HashSet::new();
        next_level.insert(start_index);
        let mut count = 0;
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<NodeIndex> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    count += level;
                }
            }
            if seen.len() == n {
                return count;
            }
            for node in found {
                for v in graph
                    .neighbors_directed(node, petgraph::Direction::Outgoing)
                {
                    next_level.insert(v);
                }
                if graph.is_directed() && as_undirected {
                    for v in graph
                        .neighbors_directed(node, petgraph::Direction::Incoming)
                    {
                        next_level.insert(v);
                    }
                }
            }
            level += 1
        }
        count
    };
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    if n < parallel_threshold {
        node_indices.iter().map(|index| bfs_traversal(*index)).sum()
    } else {
        node_indices
            .par_iter()
            .map(|index| bfs_traversal(*index))
            .sum()
    }
}
