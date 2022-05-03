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

use std::collections::VecDeque;
use std::hash::Hash;

use fixedbitset::FixedBitSet;
use petgraph::algo::{is_cyclic_directed, Measure};
use petgraph::graph::IndexType;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{
    EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable, Visitable,
};

use crate::dictmap::*;
use crate::distancemap::DistanceMap;

/// Bellman-Ford shortest path algorithm
/// using SPFA
pub fn bellman_ford<G, F, K, E, S>(
    graph: G,
    start: G::NodeId,
    mut edge_cost: F,
    mut path: Option<&mut DictMap<G::NodeId, Vec<G::NodeId>>>,
) -> Result<Option<S>, E>
where
    G: IntoEdges + Visitable + NodeIndexable + NodeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + IndexType,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
    S: DistanceMap<G::NodeId, K>,
{
    let node_count = graph.node_count();
    let mut in_queue = FixedBitSet::with_capacity(graph.node_bound());
    let mut scores: S = S::build(graph.node_bound());
    let mut predecessor: Vec<Option<G::NodeId>> = vec![None; graph.node_bound()];
    let mut visit_next = VecDeque::with_capacity(graph.node_bound());
    let zero_score = K::default();
    let mut relaxation_count: usize = 0;

    scores.put_item(start, zero_score);
    visit_next.push_back(start);

    while let Some(node) = visit_next.pop_front() {
        in_queue.set(node.index(), false);
        let node_score = *scores.get_item(node).unwrap();

        for edge in graph.edges(node) {
            let next = edge.target();
            let current_score = scores.get_item(next);
            let cost = edge_cost(edge)?;
            let next_score = node_score + cost;

            if current_score.is_none() || next_score < *current_score.unwrap() {
                scores.put_item(next, next_score);
                predecessor[next.index()] = Some(node);
                relaxation_count += 1;

                if relaxation_count == node_count {
                    relaxation_count = 0;

                    if check_for_negative_cycle(
                        predecessor.iter().map(|x| x.map(|y| y.index())).collect(),
                    ) {
                        return Ok(None);
                    }
                }

                if !in_queue.contains(next.index()) {
                    visit_next.push_back(next);
                    in_queue.set(next.index(), true);
                }
            }
        }
    }

    if path.is_some() {
        for node in graph.node_identifiers() {
            if scores.get_item(node).is_some() {
                let mut node_path = Vec::<G::NodeId>::new();
                let mut current_node = node;
                node_path.push(current_node);
                while predecessor[current_node.index()].is_some() {
                    current_node = predecessor[current_node.index()].unwrap();
                    node_path.push(current_node);
                }
                node_path.reverse();
                path.as_mut().unwrap().insert(node, node_path);
            }
        }
    }

    Ok(Some(scores))
}

/// Find negative cycle
/// using SPFA
pub fn negative_cycle_finder<G, F, K, E>(
    graph: G,
    mut edge_cost: F,
) -> Result<Option<Vec<G::NodeId>>, E>
where
    G: IntoEdges + Visitable + NodeIndexable + NodeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + IndexType,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
{
    let node_count = graph.node_count();
    let mut in_queue = FixedBitSet::with_capacity(graph.node_bound());
    let zero_score = K::default();
    let mut scores: Vec<K> = vec![zero_score; graph.node_bound()];
    let mut predecessor: Vec<Option<G::NodeId>> = vec![None; graph.node_bound()];
    let mut visit_next = VecDeque::with_capacity(graph.node_bound());
    let mut relaxation_count: usize = 0;

    for node in graph.node_identifiers() {
        visit_next.push_back(node);
        in_queue.set(node.index(), true);
    }

    while let Some(node) = visit_next.pop_front() {
        in_queue.set(node.index(), false);
        let node_score = scores[node.index()];

        for edge in graph.edges(node) {
            let next = edge.target();
            let current_score = scores[next.index()];
            let cost = edge_cost(edge)?;
            let next_score = node_score + cost;

            if next_score < current_score {
                scores[next.index()] = next_score;
                predecessor[next.index()] = Some(node);
                relaxation_count += 1;

                if relaxation_count == node_count {
                    relaxation_count = 0;

                    if check_for_negative_cycle(
                        predecessor.iter().map(|x| x.map(|y| y.index())).collect(),
                    ) {
                        return Ok(Some(recover_negative_cycle_from_predecessors(
                            graph,
                            predecessor,
                        )));
                    }
                }

                if !in_queue.contains(next.index()) {
                    visit_next.push_back(next);
                    in_queue.set(next.index(), true);
                }
            }
        }
    }

    Ok(None)
}

fn check_for_negative_cycle(predecessor: Vec<Option<usize>>) -> bool {
    let mut path_graph =
        StableDiGraph::<usize, ()>::with_capacity(predecessor.len(), predecessor.len());

    let node_indices: Vec<_> = (0..predecessor.len())
        .map(|x| path_graph.add_node(x))
        .collect();

    for (u, pred_u) in predecessor.into_iter().enumerate() {
        if let Some(v) = pred_u {
            path_graph.add_edge(node_indices[v], node_indices[u], ());
        }
    }

    is_cyclic_directed(&path_graph)
}

fn recover_negative_cycle_from_predecessors<G>(
    _graph: G,
    _predecessor: Vec<Option<G::NodeId>>,
) -> Vec<G::NodeId>
where
    G: IntoEdges + Visitable + NodeIndexable + NodeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + IndexType,
{
    // TODO: replace with actual code to find cycle
    Vec::new()
}
