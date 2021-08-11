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
use pyo3::PyObject;

/// Return True if the graph is connected
pub fn is_connected<Ty: EdgeType>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
) -> bool {
    let source = graph.node_indices().next().unwrap();
    let mut seen: HashSet<NodeIndex> =
        HashSet::with_capacity(graph.node_count());
    let mut next_level: HashSet<NodeIndex> = HashSet::new();
    next_level.insert(source);
    while !next_level.is_empty() {
        let this_level = next_level;
        next_level = HashSet::new();
        for bfs_node in this_level {
            if !seen.contains(&bfs_node) {
                seen.insert(bfs_node);
                for neighbor in graph.neighbors(bfs_node) {
                    next_level.insert(neighbor);
                }
            }
        }
    }
    seen.len() == graph.node_count()
}
