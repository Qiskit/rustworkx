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

use hashbrown::{HashSet, HashMap};
use petgraph::Directed;
use petgraph::visit::{NodeCount, EdgeCount, IntoNeighborsDirected, IntoNodeIdentifiers};
use petgraph::stable_graph::{StableGraph, NodeIndex};
use std::hash::Hash;
use rand::SeedableRng;

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
pub fn token_swapper<G, SeedableRng>(
    graph: G,
    mapping: HashMap<G::NodeId, G::NodeId>,
    trials: Option<usize>,
    seed: Option<Seed>,
) -> usize
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

    if trials.is_some() {
        num_trials = trials;
    } else {
        num_trials = 4;
    }
    if seed.is_some() {
        trial_seed = seed;
    } else {
        trial_seed = Seed::from_seed(99);
    }

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
        add_token_edges::<G>(graph, node, &tokens, &mut digraph, &mut sub_digraph, &node_map, &rev_node_map);
    }

    88
}

fn add_token_edges<G>(
    graph: G,
    node: G::NodeId,
    tokens: &HashMap<G::NodeId, G::NodeId>,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
    rev_node_map: &HashMap<NodeIndex, G::NodeId>,
)
where
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

fn distance<G>(
    graph: G,
    node0: G::NodeId,
    node1: G::NodeId,
) -> usize
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
    todo_nodes: &mut HashSet<G::NodeId>,
    tokens: &'a mut HashMap<G::NodeId, G::NodeId>,
) -> impl Iterator<Item = &'a (usize, usize)>//G::NodeId, G::NodeId)> 
where
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    [(88, 88), (99, 99)].iter()
}

fn swap<G>(
    node1: G::NodeId,
    node2: G::NodeId,
    tokens: &mut HashMap<G::NodeId, G::NodeId>,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    todo_nodes: &mut HashSet<G::NodeId>,
)
where
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    ()
}
