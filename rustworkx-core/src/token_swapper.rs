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

use rand::distributions::Uniform;
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::fmt::Debug;
use std::hash::Hash;
use std::usize::MAX;

use hashbrown::{HashMap, HashSet};
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::{
    EdgeCount, GraphBase, IntoEdges, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount,
    NodeIndexable, Visitable,
};
use petgraph::Directed;
use petgraph::Direction::{Incoming, Outgoing};

use crate::dictmap::*;
use crate::shortest_path::dijkstra;
use crate::traversal::dfs_edges;

type Swap = (NodeIndex, NodeIndex);
type Edge = (NodeIndex, NodeIndex);

struct TokenSwapper<G: GraphBase>
where
    G::NodeId: Eq + Hash + Debug,
{
    graph: G,
    // The user-supplied mapping to use for swapping tokens
    input_mapping: HashMap<G::NodeId, G::NodeId>,
    // Number of trials
    trials: usize,
    // Seed for random selection of a node for a trial
    rng_seed: Option<u64>,
    // Directed graph with nodes matching ``graph`` and
    // edges for neighbors closer than nodes
    digraph: StableGraph<(), (), Directed>,
    // Same as digraph with no self edges
    sub_digraph: StableGraph<(), (), Directed>,
    // The mapping as NodeIndex's
    tokens: HashMap<NodeIndex, NodeIndex>,
    // A list of nodes that are remaining to be tried
    todo_nodes: Vec<NodeIndex>,
    // Map of NodeId to NodeIndex
    node_map: HashMap<G::NodeId, NodeIndex>,
    // Map of NodeIndex to NodeId
    rev_node_map: HashMap<NodeIndex, G::NodeId>,
}

impl<G> TokenSwapper<G>
where
    G: NodeCount
        + EdgeCount
        + IntoEdges
        + Visitable
        + NodeIndexable
        + IntoNeighborsDirected
        + IntoNodeIdentifiers
        + Debug,
    G::NodeId: Hash + Eq + Debug,
{
    fn new(
        graph: G,
        mapping: HashMap<G::NodeId, G::NodeId>,
        trials: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        let num_nodes = graph.node_count();
        let num_edges = graph.edge_count();
        let map_len = mapping.len();

        TokenSwapper {
            graph,
            input_mapping: mapping,
            trials: match trials {
                Some(trials) => trials,
                None => 4,
            },
            rng_seed: seed,
            digraph: StableGraph::with_capacity(num_nodes, num_edges),
            sub_digraph: StableGraph::with_capacity(num_nodes, num_edges),
            tokens: HashMap::with_capacity(map_len),
            todo_nodes: Vec::with_capacity(map_len),
            node_map: HashMap::with_capacity(num_nodes),
            rev_node_map: HashMap::with_capacity(num_nodes),
        }
    }

    fn map(&mut self) -> Vec<Swap> {
        let mut rng_seed: Pcg64 = match self.rng_seed {
            Some(rng_seed) => Pcg64::seed_from_u64(rng_seed),
            None => Pcg64::from_entropy(),
        };
        for node in self.graph.node_identifiers() {
            let index = self.digraph.add_node(());
            self.sub_digraph.add_node(());
            self.node_map.insert(node, index);
            self.rev_node_map.insert(index, node);
        }
        self.tokens = self
            .input_mapping
            .iter()
            .map(|(k, v)| (self.node_map[&k], self.node_map[&v]))
            .collect();
        for (node, dest) in &self.tokens {
            if node != dest {
                self.todo_nodes.push(*node);
            }
        }
        let mut di_dummy = StableGraph::with_capacity(0, 0);
        let mut sub_di_dummy = StableGraph::with_capacity(0, 0);
        let mut token_dummy = HashMap::new();
         for node in self.graph.node_identifiers() {
            self.add_token_edges(
                self.node_map[&node],
                &mut token_dummy,
                &mut di_dummy,
                &mut sub_di_dummy,
                false,
            );
        }
        let mut trial_results: Vec<Vec<Swap>> = Vec::new();
        for _ in 0..self.trials {
            let results = self.trial_map(
                &mut self.digraph.clone(),
                &mut self.sub_digraph.clone(),
                &mut self.tokens.clone(),
                &mut self.todo_nodes.clone(),
                &mut rng_seed,
            );
            trial_results.push(results);
        }
        let mut first_results: Vec<Vec<Swap>> = Vec::new();
        for result in trial_results {
            if result.len() == 0 {
                first_results.push(result);
                break;
            }
            first_results.push(result);
        }
        let mut res_min = MAX;
        let mut final_result: Vec<Swap> = Vec::new();
        for res in first_results {
            let res_len = res.len();
            if res_len < res_min {
                final_result = res;
                res_min = res_len;
            }
        }
        final_result
    }

    fn add_token_edges(
        &mut self,
        node: NodeIndex,
        tokens: &mut HashMap<NodeIndex, NodeIndex>,
        digraph: &mut StableGraph<(), (), Directed>,
        sub_digraph: &mut StableGraph<(), (), Directed>,
        not_self: bool,
    ) {
        let tokens2: &mut HashMap<NodeIndex, NodeIndex>;
        if not_self {
            tokens2 = tokens;
        } else {
            tokens2 = &mut self.tokens;
        }
        if !(tokens2.contains_key(&node)) {
            return;
        }
        if tokens2[&node] == node {
            if not_self {
                digraph.update_edge(node, node, ());
            } else {
                self.digraph.update_edge(node, node, ());
            }
            return;
        }
        let id_node = self.rev_node_map[&node];
        let id_token = self.rev_node_map[&tokens2[&node]];
        for id_neighbor in self.graph.neighbors(id_node) {
            let neighbor = self.node_map[&id_neighbor];
            let dist_neighbor: Result<DictMap<G::NodeId, usize>, &str> = dijkstra(
                &self.graph,
                id_neighbor,
                Some(id_token),
                |_| Ok::<usize, &str>(1),
                None,
            );
            let dist_node: Result<DictMap<G::NodeId, usize>, &str> = dijkstra(
                &self.graph,
                id_node,
                Some(id_token),
                |_| Ok::<usize, &str>(1),
                None,
            );
            if dist_neighbor.unwrap()[&id_token] < dist_node.unwrap()[&id_token] {
                if not_self {
                    digraph.update_edge(node, neighbor, ());
                    sub_digraph.update_edge(node, neighbor, ());
                } else {
                    self.digraph.update_edge(node, neighbor, ());
                    self.sub_digraph.update_edge(node, neighbor, ());
                }
            }
        }
    }

    fn trial_map(
        &mut self,
        digraph: &mut StableGraph<(), (), Directed>,
        sub_digraph: &mut StableGraph<(), (), Directed>,
        tokens: &mut HashMap<NodeIndex, NodeIndex>,
        todo_nodes: &mut Vec<NodeIndex>,
        rng_seed: &mut Pcg64,
    ) -> Vec<Swap> {
        let mut steps = 0;
        let mut swap_edges: Vec<Swap> = vec![];
        while todo_nodes.len() > 0 && steps <= 4 * digraph.node_count() ^ 2 {
            let between = Uniform::new(0, todo_nodes.len());
            let random: usize = between.sample(rng_seed);
            let todo_node = todo_nodes[random];

            let cycle = find_cycle(sub_digraph, Some(todo_node));
            if cycle.len() > 0 {
                for edge in cycle[1..].iter().rev() {
                    swap_edges.push(*edge);
                    self.swap(edge.0, edge.1, digraph, sub_digraph, tokens, todo_nodes);
                }
                steps += cycle.len() - 1;
            } else {
                let mut found = false;
                let sub2 = &sub_digraph.clone();
                for edge in dfs_edges(sub2, Some(todo_node)) {
                    let new_edge = (NodeIndex::new(edge.0), NodeIndex::new(edge.1));
                    if !tokens.contains_key(&new_edge.1) {
                        swap_edges.push(new_edge);
                        self.swap(
                            new_edge.0,
                            new_edge.1,
                            digraph,
                            sub_digraph,
                            tokens,
                            todo_nodes,
                        );
                        steps += 1;
                        found = true;
                        break;
                    }
                }
                if !found {
                    let cycle: Vec<Edge> = find_cycle(digraph, Some(todo_node));
                    let unhappy_node = cycle[0].0;
                    let mut found = false;
                    let di2 = &mut digraph.clone();
                    for predecessor in di2.neighbors_directed(unhappy_node, Incoming) {
                        if predecessor != unhappy_node {
                            swap_edges.push((unhappy_node, predecessor));
                            self.swap(
                                unhappy_node,
                                predecessor,
                                digraph,
                                sub_digraph,
                                tokens,
                                todo_nodes,
                            );
                            steps += 1;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        panic!("unexpected stop")
                    }
                }
            }
        }
        if todo_nodes.len() != 0 {
            panic!("got todo nodes");
        }
        swap_edges
    }

    fn swap(
        &mut self,
        node1: NodeIndex,
        node2: NodeIndex,
        digraph: &mut StableGraph<(), (), Directed>,
        sub_digraph: &mut StableGraph<(), (), Directed>,
        tokens: &mut HashMap<NodeIndex, NodeIndex>,
        todo_nodes: &mut Vec<NodeIndex>,
    ) {
        let token1 = tokens.remove(&node1);
        let token2 = tokens.remove(&node2);
        if token2.is_some() {
            tokens.insert(node1, token2.unwrap());
        }
        if token1.is_some() {
            tokens.insert(node2, token1.unwrap());
        }
        for node in [node1, node2] {
            let mut edge_nodes = vec![];
            for successor in digraph.neighbors_directed(node, Outgoing) {
                edge_nodes.push((node, successor));
            }
            for (edge_node1, edge_node2) in edge_nodes {
                let edge = digraph.find_edge(edge_node1, edge_node2).unwrap();
                digraph.remove_edge(edge);
            }
            let mut edge_nodes = vec![];
            for successor in sub_digraph.neighbors_directed(node, Outgoing) {
                edge_nodes.push((node, successor));
            }
            for (edge_node1, edge_node2) in edge_nodes {
                let edge = sub_digraph.find_edge(edge_node1, edge_node2).unwrap();
                sub_digraph.remove_edge(edge);
            }
            self.add_token_edges(node, tokens, digraph, sub_digraph, true);
            if tokens.contains_key(&node) && tokens[&node] != node {
                if !todo_nodes.contains(&node) {
                    println!("DID PUSH");
                    todo_nodes.push(node);
                }
            } else if todo_nodes.contains(&node) {
                println!("DID REMOVE");
                todo_nodes.swap_remove(todo_nodes.iter().position(|x| *x == node).unwrap());
            }
        }
    }
}

fn find_cycle(graph: &mut StableGraph<(), (), Directed>, source: Option<NodeIndex>) -> Vec<Edge> {
    let mut graph_nodes: HashSet<NodeIndex> = graph.node_identifiers().collect();

    let mut cycle: Vec<Edge> = Vec::with_capacity(graph.edge_count());
    let temp_value: NodeIndex;
    // If source is not set get an arbitrary node from the set of graph
    // nodes we've not "examined"
    let source_index = match source {
        Some(source_value) => source_value,
        None => {
            temp_value = *graph_nodes.iter().next().unwrap();
            graph_nodes.remove(&temp_value);
            temp_value
        }
    };
    // Stack (ie "pushdown list") of vertices already in the spanning tree
    let mut stack: Vec<NodeIndex> = vec![source_index];
    // map to store parent of a node
    let mut pred: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    // a node is in the visiting set if at least one of its child is unexamined
    let mut visiting = HashSet::new();
    // a node is in visited set if all of its children have been examined
    let mut visited = HashSet::new();
    while !stack.is_empty() {
        let mut z = *stack.last().unwrap();
        visiting.insert(z);

        let children = graph.neighbors_directed(z, Outgoing);
        for child in children {
            //cycle is found
            if visiting.contains(&child) {
                cycle.push((z, child));
                //backtrack
                loop {
                    if z == child {
                        cycle.reverse();
                        break;
                    }
                    cycle.push((pred[&z], z));
                    z = pred[&z];
                }
                return cycle;
            }
            //if an unexplored node is encountered
            if !visited.contains(&child) {
                stack.push(child);
                pred.insert(child, z);
            }
        }
        let top = *stack.last().unwrap();
        //if no further children and explored, move to visited
        if top == z {
            stack.pop();
            visiting.remove(&z);
            visited.insert(z);
        }
    }
    cycle
}

/// This module performs an approximately optimal Token Swapping algorithm
/// Supports partial mappings (i.e. not-permutations) for graphs with missing self.tokens.
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
/// ```rust
/// use hashbrown::HashMap;
/// use rustworkx_core::petgraph;
/// use rustworkx_core::token_swapper::token_swapper;
/// use rustworkx_core::petgraph::graph::NodeIndex;
///
///  let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3)]);
///  let mapping = HashMap::from([
///   (NodeIndex::new(0), NodeIndex::new(0)),
///   (NodeIndex::new(1), NodeIndex::new(3)),
///   (NodeIndex::new(3), NodeIndex::new(1)),
///   (NodeIndex::new(2), NodeIndex::new(2)),
///  ]);
///  // Do the token swap
///  let output = token_swapper(&g, mapping, Some(4), Some(4));
///  //let output = token_swapper.map();
///
///  assert_eq!(3, output.len());
///
/// ```

pub fn token_swapper<G>(
    graph: G,
    mapping: HashMap<G::NodeId, G::NodeId>,
    trials: Option<usize>,
    seed: Option<u64>,
) -> Vec<Swap>
where
    G: NodeCount
        + EdgeCount
        + IntoEdges
        + Visitable
        + NodeIndexable
        + IntoNeighborsDirected
        + IntoNodeIdentifiers
        + Debug,
    G::NodeId: Hash + Eq + Debug,
{
    let mut swapper = TokenSwapper::new(graph, mapping, trials, seed);
    swapper.map()
}

#[cfg(test)]
mod test_token_swapper {

    use crate::petgraph;
    use crate::token_swapper::token_swapper;
    use hashbrown::HashMap;
    use petgraph::graph::NodeIndex;

    #[test]
    fn test_simple_swap() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3)]);
        let mapping = HashMap::from([
            (NodeIndex::new(0), NodeIndex::new(0)),
            (NodeIndex::new(1), NodeIndex::new(3)),
            (NodeIndex::new(3), NodeIndex::new(1)),
            (NodeIndex::new(2), NodeIndex::new(2)),
        ]);
        // Do the token swap
        let output = token_swapper(&g, mapping, Some(4), Some(4));
        assert_eq!(3, output.len());
    }
}
