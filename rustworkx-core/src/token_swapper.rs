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
use hashbrown::hash_map::Entry;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::{
    EdgeCount, GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount,
};
use petgraph::Directed;
use petgraph::Direction::Outgoing;
use rand::prelude::*;
use rand::distributions::Uniform;
use rand_pcg::Pcg64;
use std::hash::Hash;
use std::usize::MAX;
//use std::collections::hash_map::Entry;

//use crate::shortest_path::distance_matrix::compute_distance_matrix;

/// This module performs an approximately optimal Token Swapping algorithm
/// Supports partial mappings (i.e. not-permutations) for graphs with missing tokens.
///
/// Based on the paper: Approximation and Hardness for Token Swapping by Miltzow et al. (2016)
/// ArXiV: https://arxiv.org/abs/1602.05150
/// and generalization based on our own work.
///
/// The inputs are a partial `mapping` to be implemented in swaps, and the number of `trials` to
/// perform the mapping. It's minimized over the trials.
///
/// It returns a list of tuples representing the swaps to perform.
///
/// # Example
/// ```
/// use petgraph::prelude::*;
/// use hashbrown::HashSet;
/// use rustworkx_core::connectivity::all_simple_paths_multiple_targets;
///
/// let mut graph = DiGraph::<&str, i32>::new();
///
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// let d = graph.add_node("d");
///
/// graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (a, b, 1), (b, d, 1)]);
///
/// let mut to_set = HashSet::new();
/// to_set.insert(d);
///
/// let ways = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, None);
///
/// let d_path = ways.get(&d).unwrap();
/// assert_eq!(4, d_path.len());
/// ```

type Swap<G> = (<G as GraphBase>::NodeId, <G as GraphBase>::NodeId);

pub fn token_swapper<G>(
    graph: G,
    mapping: HashMap<G::NodeId, G::NodeId>,
    trials: Option<usize>,
    seed: Option<u64>,
) -> Vec<Swap<G>>
where
    G: NodeCount,
    G: EdgeCount,
    G: IntoNeighborsDirected,
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
{
    let mut digraph = StableGraph::with_capacity(graph.node_count(), graph.edge_count());
    // sub_digraph is digraph without self edges
    let mut sub_digraph = StableGraph::with_capacity(graph.node_count(), graph.edge_count());
    let mut todo_nodes: HashSet<G::NodeId> = HashSet::new();
    let tokens: HashMap<G::NodeId, G::NodeId> = mapping.into_iter().collect();

    let mut trials: usize = match trials {
        Some(trials) => trials,
        None => 4,
    };
    let mut rng_seed: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let between = Uniform::new(0, graph.node_count());
    let random: usize = between.sample(&mut rng_seed);

    for (node, destination) in tokens.iter() {
        if node != destination {
            todo_nodes.insert(*node);
        }
    }
    let mut node_map = HashMap::with_capacity(graph.node_count());
    let mut rev_node_map = HashMap::with_capacity(graph.node_count());

    for node in graph.node_identifiers() {
        let index = digraph.add_node(());
        sub_digraph.add_node(());
        node_map.insert(node, index);
        rev_node_map.insert(index, node);
    }
    for node in graph.node_identifiers() {
        add_token_edges::<G>(
            graph,
            node,
            &tokens,
            &mut digraph,
            &mut sub_digraph,
            &node_map,
        );
    }
    let mut trial_results: Vec<Vec<Swap<G>>> = Vec::new();
    for _ in 0..trials {
        let results = trial_map::<G>(
            &mut digraph.clone(),
            &mut sub_digraph.clone(),
            &mut todo_nodes.clone(),
            &mut tokens.clone(),
            &random,
        );
        trial_results.push(results);
    }
    let mut first_results: Vec<Vec<Swap<G>>> = Vec::new();
    for result in trial_results {
        if result.len() == 0 {
            first_results.push(result);
            break;
        }
        first_results.push(result);
    }
    let mut res_min = MAX;
    let mut final_result: Vec<Swap<G>> = Vec::new();
    for res in first_results {
        let res_len = res.len();
        if res_len < res_min {
            final_result = res;
            res_min = res_len;
        }
    }
    final_result
}

fn add_token_edges<G>(
    graph: G,
    node: G::NodeId,
    tokens: &HashMap<G::NodeId, G::NodeId>,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
) where
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    if !(tokens.contains_key(&node)) {
        return;
    }
    if tokens[&node] == node {
        digraph.add_edge(node_map[&node], node_map[&node], ());
        return;
    }
    for neighbor in graph.neighbors(node) {
        if distance(graph, neighbor, tokens[&node]) < distance(graph, node, tokens[&node]) {
            digraph.add_edge(node_map[&node], node_map[&neighbor], ());
            sub_digraph.add_edge(node_map[&node], node_map[&neighbor], ());
        }
    }
}

fn distance<G>(graph: G, node0: G::NodeId, node1: G::NodeId) -> usize
where
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    88
    //compute_distance_matrix(graph)[node0, node1]
}

fn trial_map<'a, G>(
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    todo_nodes: &'a mut HashSet<G::NodeId>,
    tokens: &'a mut HashMap<G::NodeId, G::NodeId>,
    random: &usize,
) -> Vec<Swap<G>>
where
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    let x = todo_nodes.iter().next().unwrap();
    vec![(*x, *x)]
}

fn swap<G>(
    node1: G::NodeId,
    node2: G::NodeId,
    graph: G,
    tokens: &mut HashMap<G::NodeId, G::NodeId>,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    todo_nodes: &mut HashSet<G::NodeId>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
) where
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    let token1 = tokens.remove_entry(&node1);
    let token2 = tokens.remove_entry(&node2);
    if token2.is_some() {
        tokens.insert(token2.unwrap().0, token2.unwrap().1);
    }
    if token1.is_some() {
        tokens.insert(token1.unwrap().0, token1.unwrap().1);
    }
    for node in [node1, node2] {
        let mut edge_nodes = vec![];
        for successor in digraph.neighbors_directed(node_map[&node], Outgoing) {
            edge_nodes.push((node_map[&node], successor));
        }
        for (edge_node1, edge_node2) in edge_nodes {
            let edge = digraph.find_edge(edge_node1, edge_node2).unwrap();
            digraph.remove_edge(edge);
        }
        let mut edge_nodes = vec![];
        for successor in sub_digraph.neighbors_directed(node_map[&node], Outgoing) {
            edge_nodes.push((node_map[&node], successor));
        }
        for (edge_node1, edge_node2) in edge_nodes {
            let edge = sub_digraph.find_edge(edge_node1, edge_node2).unwrap();
            sub_digraph.remove_edge(edge);
        }
        add_token_edges::<G>(graph, node, &tokens, digraph, sub_digraph, &node_map);
        if tokens.contains_key(&node) && tokens[&node] != node {
            todo_nodes.insert(node);
        } else if todo_nodes.contains(&node) {
            todo_nodes.remove(&node);
        }
    }
}
