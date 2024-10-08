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

use std::hash::Hash;

use petgraph::data::{Build, Create};
use petgraph::visit::{Data, NodeIndexable};

/// Generates Zachary's Karate Club graph.
///
/// Zachary's Karate Club graph is a well-known social network that represents
/// the relations between 34 members of a karate club.
pub fn karate_club_graph<G, T, F, H, M>(mut default_node_weight: F, mut default_edge_weight: H) -> G
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut(bool) -> T,
    H: FnMut(usize) -> M,
    G::NodeId: Eq + Hash,
{
    const N: usize = 34;
    const M: usize = 78;
    let mr_hi_members: [u8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21];
    let membership: std::collections::HashSet<u8> = mr_hi_members.into_iter().collect();

    let adjacency_list: Vec<Vec<(usize, usize)>> = vec![
        vec![],
        vec![(0, 4)],
        vec![(0, 5), (1, 6)],
        vec![(0, 3), (1, 3), (2, 3)],
        vec![(0, 3)],
        vec![(0, 3)],
        vec![(0, 3), (4, 2), (5, 5)],
        vec![(0, 2), (1, 4), (2, 4), (3, 3)],
        vec![(0, 2), (2, 5)],
        vec![(2, 1)],
        vec![(0, 2), (4, 3), (5, 3)],
        vec![(0, 3)],
        vec![(0, 1), (3, 3)],
        vec![(0, 3), (1, 5), (2, 3), (3, 3)],
        vec![],
        vec![],
        vec![(5, 3), (6, 3)],
        vec![(0, 2), (1, 1)],
        vec![],
        vec![(0, 2), (1, 2)],
        vec![],
        vec![(0, 2), (1, 2)],
        vec![],
        vec![],
        vec![],
        vec![(23, 5), (24, 2)],
        vec![],
        vec![(2, 2), (23, 4), (24, 3)],
        vec![(2, 2)],
        vec![(23, 3), (26, 4)],
        vec![(1, 2), (8, 3)],
        vec![(0, 2), (24, 2), (25, 7), (28, 2)],
        vec![
            (2, 2),
            (8, 3),
            (14, 3),
            (15, 3),
            (18, 1),
            (20, 3),
            (22, 2),
            (23, 5),
            (29, 4),
            (30, 3),
            (31, 4),
        ],
        vec![
            (8, 4),
            (9, 2),
            (13, 3),
            (14, 2),
            (15, 4),
            (18, 2),
            (19, 1),
            (20, 1),
            (23, 4),
            (26, 2),
            (27, 4),
            (28, 2),
            (29, 2),
            (30, 3),
            (31, 4),
            (32, 5),
            (22, 3),
        ],
    ];

    let mut graph = G::with_capacity(N, M);

    let mut node_indices = Vec::with_capacity(N);
    for (row, neighbors) in adjacency_list.into_iter().enumerate() {
        let node_id = graph.add_node(default_node_weight(membership.contains(&(row as u8))));
        node_indices.push(node_id);

        for (neighbor, weight) in neighbors.into_iter() {
            graph.add_edge(
                node_indices[neighbor],
                node_indices[row],
                default_edge_weight(weight),
            );
        }
    }
    graph
}
