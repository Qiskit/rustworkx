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
use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::visit::Visitable;
use petgraph::Undirected;

use rustworkx_core::connectivity;

#[test]
fn test_is_connected() {
    let graph = Graph::<(), (), Undirected>::from_edges(&[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
    ]);
    let node = NodeIndex::new(6);
    let component: HashSet<usize> =
        connectivity::bfs_undirected(&graph, node, &mut graph.visit_map())
            .into_iter()
            .map(|x| x.index())
            .collect();
    assert_eq!(component.len(), graph.node_count());
}
