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

use crate::StablePyGraph;
use hashbrown::HashSet;
use petgraph::EdgeType;
use petgraph::stable_graph::NodeIndex;
use std::cmp::max;

// Implemented after ``Simple`` from
// "Enumerating Connected Induced Subgraphs: Improved Delay and Experimental Comparison".
// Christian Komusiewicz and Frank Sommer. Some more runtime and spaceimprovements can be gained by
// implementing the data structure defined for Lemma 4 (essentially a more efficient way of
// tracking set ``x`` and ``p``)
pub fn k_connected_subgraphs<Ty: EdgeType + Sync>(
    graph: &StablePyGraph<Ty>,
    k: usize,
) -> Vec<Vec<usize>> {
    let mut connected_subgraphs = Vec::new();
    let mut graph = graph.clone();

    while let Some(v) = graph.node_indices().next() {
        if graph.node_count() < max(k, 1) {
            break;
        }

        let mut p: HashSet<NodeIndex> = HashSet::new();
        p.insert(v);
        let mut x: HashSet<NodeIndex> = graph.neighbors(v).collect();
        simple_enum(&mut p, &mut x, &graph, &mut connected_subgraphs, k);
        graph.remove_node(v);
    }
    connected_subgraphs
}

fn simple_enum<Ty: EdgeType + Sync>(
    p: &mut HashSet<NodeIndex>,
    x: &mut HashSet<NodeIndex>,
    graph: &StablePyGraph<Ty>,
    subgraphs: &mut Vec<Vec<usize>>,
    k: usize,
) -> bool {
    if p.len() == k {
        subgraphs.push(p.iter().map(|n| n.index()).collect::<Vec<usize>>());
        return true;
    }
    let mut is_leaf_node: bool = false;
    while let Some(u) = x.iter().next().cloned() {
        x.remove(&u);

        let nu: HashSet<NodeIndex> = graph.neighbors(u).collect();
        let np: HashSet<NodeIndex> = p
            .iter()
            .flat_map(|n| graph.neighbors(*n))
            .collect::<HashSet<NodeIndex>>()
            .union(p)
            .cloned()
            .collect();
        //X' = X u N(u)/ N|P|
        let mut x_next: HashSet<NodeIndex> = x
            .union(&nu.difference(&np).cloned().collect())
            .cloned()
            .collect();
        let mut p_next: HashSet<NodeIndex> = p.clone();
        p_next.insert(u);

        if simple_enum(&mut p_next, &mut x_next, graph, subgraphs, k) {
            is_leaf_node = true;
        } else {
            return is_leaf_node;
        }
    }
    is_leaf_node
}
