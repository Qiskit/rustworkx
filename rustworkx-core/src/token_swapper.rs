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

use crate::dictmap::*;
use crate::shortest_path::dijkstra;
use hashbrown::{HashMap, HashSet};
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::{
    EdgeCount, GraphBase, IntoEdges, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount,
    NodeIndexable, Visitable,
};
use petgraph::Directed;
use petgraph::Direction::Outgoing;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::hash::Hash;
use std::usize::MAX;

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
type Edge<G> = (<G as GraphBase>::NodeId, <G as GraphBase>::NodeId);

pub fn token_swapper<G, E>(
    graph: G,
    mapping: HashMap<G::NodeId, G::NodeId>,
    trials: Option<usize>,
    seed: Option<u64>,
) -> Vec<Swap<G>>
where
    G: GraphBase,
    G: NodeCount,
    G: EdgeCount,
    G: IntoEdges,
    G: Visitable,
    G: NodeIndexable,
    G: IntoNeighborsDirected,
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    E: std::fmt::Debug,
{
    let mut digraph = StableGraph::with_capacity(graph.node_count(), graph.edge_count());
    // sub_digraph is digraph without self edges
    let mut sub_digraph = StableGraph::with_capacity(graph.node_count(), graph.edge_count());
    let mut todo_nodes: Vec<G::NodeId> = Vec::new();
    let tokens: HashMap<G::NodeId, G::NodeId> = mapping.into_iter().collect();

    let trials: usize = match trials {
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
            todo_nodes.push(*node);
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
        add_token_edges::<G, E>(
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
        let results = trial_map::<G, E>(
            graph,
            &mut digraph.clone(),
            &mut sub_digraph.clone(),
            &mut todo_nodes.clone(),
            &mut tokens.clone(),
            &node_map,
            &rev_node_map,
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

fn add_token_edges<G, E>(
    graph: G,
    node: G::NodeId,
    tokens: &HashMap<G::NodeId, G::NodeId>,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
) where
    G: IntoNeighborsDirected,
    G: IntoEdges,
    G: Visitable,
    G: NodeIndexable,
    G::NodeId: Eq + Hash,
    E: std::fmt::Debug,
{
    if !(tokens.contains_key(&node)) {
        return;
    }
    if tokens[&node] == node {
        digraph.add_edge(node_map[&node], node_map[&node], ());
        return;
    }
    for neighbor in graph.neighbors(node) {
        if distance::<G, E>(graph, neighbor, tokens[&node]).unwrap()[&neighbor]
            < distance::<G, E>(graph, node, tokens[&node]).unwrap()[&node]
        {
            digraph.add_edge(node_map[&node], node_map[&neighbor], ());
            sub_digraph.add_edge(node_map[&node], node_map[&neighbor], ());
        }
    }
}

fn distance<G, E>(
    graph: G,
    node0: G::NodeId,
    node1: G::NodeId,
) -> Result<DictMap<G::NodeId, usize>, E>
where
    G: IntoEdges,
    G: Visitable,
    G: NodeIndexable,
    G::NodeId: Eq + Hash,
{
    dijkstra(&graph, node0, Some(node1), |_| Ok(1), None)
}

fn trial_map<G, E>(
    graph: G,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    todo_nodes: &mut Vec<G::NodeId>,
    tokens: &mut HashMap<G::NodeId, G::NodeId>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
    rev_node_map: &HashMap<NodeIndex, G::NodeId>,
    random: &usize,
) -> Vec<Swap<G>>
where
    G: EdgeCount,
    G: IntoEdges,
    G: IntoNeighborsDirected,
    G: NodeIndexable,
    G: IntoNodeIdentifiers,
    G: Visitable,
    G::NodeId: Eq + Hash,
    E: std::fmt::Debug,
{
    let mut steps = 0;
    let mut swap_edges: Vec<Swap<G>> = vec![];
    while todo_nodes.len() > 0 && steps <= 4 * digraph.node_count() ^ 2 {
        let todo_node = todo_nodes[*random];
        let cycle = digraph_find_cycle(graph, Some(todo_node), node_map, rev_node_map);
        if cycle.len() > 0 {
            for edge in cycle[1..].iter().rev() {
                swap_edges.push(*edge);
                swap::<G, E>(
                    edge.0,
                    edge.1,
                    graph,
                    tokens,
                    digraph,
                    sub_digraph,
                    todo_nodes,
                    node_map,
                );
                steps += cycle.len() - 1
        else:
                # Try to find a node without a token to swap with.
                try:
                    edge = next(
                        edge
                        for edge in rx.digraph_dfs_edges(sub_digraph, todo_node)
                        if edge[1] not in tokens
                    )
                    # Swap predecessor and successor, because successor does not have a token
                    yield edge
                    swap(edge[0], edge[1])
                    steps += 1
                except StopIteration:
                    # Unhappy swap case
                    cycle = rx.digraph_find_cycle(digraph, source=todo_node)
                    assert len(cycle) == 1, "The cycle was not unhappy."
                    unhappy_node = cycle[0][0]
                    # Find a node that wants to swap with this node.
                    try:
                        predecessor = next(
                            predecessor
                            for predecessor in digraph.predecessor_indices(unhappy_node)
                            if predecessor != unhappy_node
                        )
                    except StopIteration:
                        logger.error(
                            "Unexpected StopIteration raised when getting predecessors"
                            "in unhappy swap case."
                        )
                        return
                    yield unhappy_node, predecessor
                    swap(unhappy_node, predecessor)
                    steps += 1
        if todo_nodes:
            raise RuntimeError("Too many iterations while approximating the Token Swaps.")
            }
        }
        steps = 10000;
    }
    let x = todo_nodes.iter().next().unwrap();
    vec![(*x, *x)]
}

fn swap<G, E>(
    node1: G::NodeId,
    node2: G::NodeId,
    graph: G,
    tokens: &mut HashMap<G::NodeId, G::NodeId>,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    todo_nodes: &mut Vec<G::NodeId>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
) where
    G: IntoNeighborsDirected,
    G: IntoEdges,
    G: Visitable,
    G: NodeIndexable,
    G::NodeId: Eq + Hash,
    E: std::fmt::Debug,
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
        add_token_edges::<G, E>(graph, node, &tokens, digraph, sub_digraph, &node_map);
        if tokens.contains_key(&node) && tokens[&node] != node {
            todo_nodes.push(node);
        } else if todo_nodes.contains(&node) {
            todo_nodes.remove(todo_nodes.iter().position(|x| *x == node).unwrap());
        }
    }
}

fn digraph_find_cycle<G>(
    graph: G,
    source: Option<G::NodeId>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
    rev_node_map: &HashMap<NodeIndex, G::NodeId>,
) -> Vec<Edge<G>>
where
    G: EdgeCount,
    G: IntoNeighborsDirected,
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
{
    let mut graph_nodes: HashSet<NodeIndex> = HashSet::new();
    for id in graph.node_identifiers() {
        graph_nodes.insert(node_map[&id]);
    }
    let mut cycle: Vec<Edge<G>> = Vec::with_capacity(graph.edge_count());
    let temp_value: NodeIndex;
    // If source is not set get an arbitrary node from the set of graph
    // nodes we've not "examined"
    let source_index = match source {
        Some(source_value) => node_map[&source_value],
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

        let children = graph.neighbors_directed(rev_node_map[&z], petgraph::Direction::Outgoing);

        for child_id in children {
            let child = node_map[&child_id];
            //cycle is found
            if visiting.contains(&child) {
                cycle.push((rev_node_map[&z], rev_node_map[&child]));
                //backtrack
                loop {
                    if z == child {
                        cycle.reverse();
                        break;
                    }
                    cycle.push((rev_node_map[&pred[&z]], rev_node_map[&z]));
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
        if top.index() == z.index() {
            stack.pop();
            visiting.remove(&z);
            visited.insert(z);
        }
    }
    cycle
}
