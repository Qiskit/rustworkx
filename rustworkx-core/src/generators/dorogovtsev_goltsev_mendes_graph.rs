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

use petgraph::{
    data::Create,
    visit::{Data, EdgeIndexable},
};

use super::InvalidInputError;

// TODO: docs
pub fn dorogovtsev_goltsev_mendes_graph<G, T, F, H, M>(
    t: isize,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: Create + Data<NodeWeight = T, EdgeWeight = M>,
    F: FnMut() -> T,
    H: FnMut() -> M,
{
    if t < -1 {
        return Err(InvalidInputError {});
    }
    let n_edges = usize::pow(3, t as u32 + 1); // Check against overflow?
    let n_nodes = (n_edges + 3) / 2;
    let mut graph = G::with_capacity(n_nodes, n_edges);

    let node_0 = graph.add_node(default_node_weight());
    let node_1 = graph.add_node(default_node_weight());
    graph
        .add_edge(node_0, node_1, default_edge_weight())
        .unwrap();
    let mut current_endpoints = vec![(node_0, node_1)];

    for _ in 0..t + 1 {
        let mut new_endpoints = vec![];
        for (source, target) in current_endpoints.iter() {
            let new_node = graph.add_node(default_node_weight());
            graph.add_edge(*source, new_node, default_edge_weight());
            new_endpoints.push((*source, new_node));
            graph.add_edge(*target, new_node, default_edge_weight());
            new_endpoints.push((*target, new_node));
        }
        current_endpoints.extend(new_endpoints);
    }
    Ok(graph)
}

// TODO: tests
