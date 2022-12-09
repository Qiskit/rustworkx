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
use crate::traversal::dfs_edges;
use hashbrown::{HashMap, HashSet};
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::{
    EdgeCount, GraphBase, IntoEdges, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount,
    NodeIndexable, Visitable,
};
use petgraph::Directed;
use petgraph::Direction::{Incoming, Outgoing};
use rand::distributions::Uniform;
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::hash::Hash;
use std::usize::MAX;

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
/// ```rust
/// use hashbrown::HashMap;
/// use rustworkx_core::petgraph;
/// use rustworkx_core::petgraph::graph::NodeIndex;
/// use rustworkx_core::token_swapper::token_swapper;
///
/// let g = petgraph::graph::DiGraph::<(), ()>::from_edges(&[
///     (0, 1), (1, 2), (2, 3), (3, 4),
/// ]);
/// let mapping = HashMap::from([(NodeIndex::new(0), NodeIndex::new(0)), (NodeIndex::new(1), NodeIndex::new(3)), (NodeIndex::new(3), NodeIndex::new(1)), (NodeIndex::new(2), NodeIndex::new(2))]);
/// // Do the token swap
/// let output = token_swapper::<&petgraph::graph::Graph<(), ()>>(&g, mapping, Some(4), Some(4));
/// assert_eq!(3, output.len());
/// ```

type Swap = (NodeIndex, NodeIndex);
type Edge = (NodeIndex, NodeIndex);

pub fn token_swapper<G>(
    graph: G,
    mapping: HashMap<G::NodeId, G::NodeId>,
    trials: Option<usize>,
    seed: Option<u64>,
) -> Vec<Swap>
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
{
    let mut digraph = StableGraph::with_capacity(graph.node_count(), graph.edge_count());
    // sub_digraph is digraph without self edges
    let mut sub_digraph = StableGraph::with_capacity(graph.node_count(), graph.edge_count());
    //let mut todo_nodes: Vec<NodeIndex> = Vec::new();

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

    let mut node_map = HashMap::with_capacity(graph.node_count());
    let mut rev_node_map = HashMap::with_capacity(graph.node_count());

    for node in graph.node_identifiers() {
        let index = digraph.add_node(());
        sub_digraph.add_node(());
        node_map.insert(node, index);
        rev_node_map.insert(index, node);
    }
    let mut tokens: HashMap<NodeIndex, NodeIndex> = mapping.iter().map(|(k, v)| (node_map[&k], node_map[&v])).collect();
    print!("\nTOKENS {:?}", tokens);
    let mut todo_nodes: Vec<NodeIndex> = tokens.iter().map(|(node, dest)| {if node == dest {*node} else {*node}}).collect();

    for node in graph.node_identifiers() {
        add_token_edges::<G>(
            graph,
            node_map[&node],
            &tokens,
            &mut digraph,
            &mut sub_digraph,
            &node_map,
            &rev_node_map,
        );
    }
    let mut trial_results: Vec<Vec<Swap>> = Vec::new();
    for _ in 0..trials {
        let results = trial_map::<G>(
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

fn add_token_edges<G>(
    graph: G,
    node: NodeIndex,
    tokens: &HashMap<NodeIndex, NodeIndex>,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
    rev_node_map: &HashMap<NodeIndex, G::NodeId>,
) where
    G: IntoNeighborsDirected,
    G: IntoEdges,
    G: Visitable,
    G: NodeIndexable,
    G::NodeId: Eq + Hash,
{
    let id_node = rev_node_map[&node];
    if !(tokens.contains_key(&node)) {
        return;
    }
    if tokens[&node] == node {
        digraph.add_edge(node, node, ());
        return;
    }
    let id_token = rev_node_map[&tokens[&node]];
    for id_neighbor in graph.neighbors(id_node) {
        let neighbor = node_map[&id_neighbor];
        if distance::<G>(graph, id_neighbor, id_token)[&id_neighbor]
            < distance::<G>(graph, id_node, id_token)[&id_node]
        {
            digraph.add_edge(node, neighbor, ());
            sub_digraph.add_edge(node, neighbor, ());
        }
    }
}

fn distance<G>(graph: G, node0: G::NodeId, node1: G::NodeId) -> DictMap<G::NodeId, usize>
where
    G: IntoEdges,
    G: Visitable,
    G: NodeIndexable,
    G::NodeId: Eq + Hash,
{
    let res = dijkstra(&graph, node0, Some(node1), |_| Ok::<usize, &str>(1), None).unwrap();
    res
}

fn trial_map<G>(
    graph: G,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    todo_nodes: &mut Vec<NodeIndex>,
    tokens: &mut HashMap<NodeIndex, NodeIndex>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
    rev_node_map: &HashMap<NodeIndex, G::NodeId>,
    random: &usize,
) -> Vec<Swap>
where
    G: EdgeCount,
    G: IntoEdges,
    G: IntoNeighborsDirected,
    G: NodeIndexable,
    G: IntoNodeIdentifiers,
    G: Visitable,
    G::NodeId: Eq + Hash,
{
    print!("tokens {:?} todo_nodes {:?}", tokens, todo_nodes);
    let mut steps = 0;
    let mut swap_edges: Vec<Swap> = vec![];
    while todo_nodes.len() > 0 && steps <= 4 * digraph.node_count() ^ 2 {
        let todo_node = todo_nodes[*random];
        print!("\ntodo_node {:?}", todo_node);
        let cycle = find_cycle(
            graph,
            digraph,
            false,
            Some(todo_node),
            node_map,
            rev_node_map,
        );
        print!("\nFirst cycle {:?}", cycle);
        if cycle.len() > 0 {
            for edge in cycle[1..].iter().rev() {
                swap_edges.push(*edge);
                swap::<G>(
                    edge.0,
                    edge.1,
                    graph,
                    tokens,
                    todo_nodes,
                    digraph,
                    sub_digraph,
                    node_map,
                    rev_node_map,
                );
                steps += cycle.len() - 1
            }
        } else {
            let mut found = false;
            let sub2 = &sub_digraph.clone();
            for edge in dfs_edges(sub2, Some(todo_node)) {
                let new_edge = (NodeIndex::new(edge.0), NodeIndex::new(edge.1));
                if !tokens.contains_key(&new_edge.1) {
                    swap_edges.push(new_edge);
                    swap::<G>(
                        new_edge.0,
                        new_edge.1,
                        graph,
                        tokens,
                        todo_nodes,
                        digraph,
                        sub_digraph,
                        node_map,
                        rev_node_map,
                    );
                    steps += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                let di2 = &mut digraph.clone();
                let cycle = find_cycle(graph, di2, true, Some(todo_node), node_map, rev_node_map);
                println!("CYCLE {:?}", cycle);
                let unhappy_node = cycle[0].0;
                let mut found = false;
                for predecessor in di2.neighbors_directed(unhappy_node, Incoming) {
                    if predecessor == unhappy_node {
                        found = true;
                        swap_edges.push((unhappy_node, predecessor));
                        swap::<G>(
                            unhappy_node,
                            predecessor,
                            graph,
                            tokens,
                            todo_nodes,
                            digraph,
                            sub_digraph,
                            node_map,
                            rev_node_map,
                        );
                        steps += 1
                    }
                }
                if !found {
                    panic!("unexpected stop")
                }
            }
        }
        if todo_nodes.len() != 0 {
            panic!("got todo nodes");
        }
    }
    swap_edges
}

fn swap<G>(
    node1: NodeIndex,
    node2: NodeIndex,
    graph: G,
    tokens: &mut HashMap<NodeIndex, NodeIndex>,
    todo_nodes: &mut Vec<NodeIndex>,
    digraph: &mut StableGraph<(), (), Directed>,
    sub_digraph: &mut StableGraph<(), (), Directed>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
    rev_node_map: &HashMap<NodeIndex, G::NodeId>,
) where
    G: IntoNeighborsDirected,
    G: IntoEdges,
    G: Visitable,
    G: NodeIndexable,
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
        for successor in digraph.neighbors_directed(node, Outgoing) {
            edge_nodes.push((&node, successor));
        }
        for (edge_node1, edge_node2) in edge_nodes {
            let edge = digraph.find_edge(*edge_node1, edge_node2).unwrap();
            digraph.remove_edge(edge);
        }
        let mut edge_nodes = vec![];
        for successor in sub_digraph.neighbors_directed(node, Outgoing) {
            edge_nodes.push((&node, successor));
        }
        for (edge_node1, edge_node2) in edge_nodes {
            let edge = sub_digraph.find_edge(*edge_node1, edge_node2).unwrap();
            sub_digraph.remove_edge(edge);
        }
        add_token_edges::<G>(
            graph,
            node,
            &tokens,
            digraph,
            sub_digraph,
            &node_map,
            &rev_node_map,
        );
        if tokens.contains_key(&node) && tokens[&node] != node {
            todo_nodes.push(node);
        } else if todo_nodes.contains(&node) {
            todo_nodes.remove(todo_nodes.iter().position(|x| *x == node).unwrap());
        }
    }
}

fn find_cycle<G>(
    graph: G,
    digraph: &mut StableGraph<(), (), Directed>,
    use_digraph: bool,
    source: Option<NodeIndex>,
    node_map: &HashMap<G::NodeId, NodeIndex>,
    rev_node_map: &HashMap<NodeIndex, G::NodeId>,
) -> Vec<Edge>
where
    G: EdgeCount,
    G: IntoNeighborsDirected,
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
{
    let mut graph_nodes: HashSet<NodeIndex> = HashSet::new();
    if use_digraph {
        for id in graph.node_identifiers() {
            graph_nodes.insert(node_map[&id]);
        }
    } else {
        for id in digraph.node_identifiers() {
            graph_nodes.insert(id);
        }
    }
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

        if use_digraph {
            let children = digraph.neighbors_directed(z, petgraph::Direction::Outgoing);
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
        } else {
            let children =
                graph.neighbors_directed(rev_node_map[&z], petgraph::Direction::Outgoing);
            for child_id in children {
                let child = node_map[&child_id];
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

#[cfg(test)]
mod test_token_swapper {

    use crate::petgraph;
    use crate::token_swapper::token_swapper;
    use hashbrown::HashMap;
    use petgraph::graph::NodeIndex;

    #[test]
    fn test_simple_swap() {
        let g = petgraph::graph::DiGraph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let mapping = HashMap::from([
            (NodeIndex::new(0), NodeIndex::new(0)),
            (NodeIndex::new(1), NodeIndex::new(3)),
            (NodeIndex::new(3), NodeIndex::new(1)),
            (NodeIndex::new(2), NodeIndex::new(2)),
        ]);
        // Do the token swap
        let output =
            token_swapper::<&petgraph::graph::Graph<(), ()>>(&g, mapping, Some(4), Some(4));
        assert_eq!(3, output.len());
    }
}
