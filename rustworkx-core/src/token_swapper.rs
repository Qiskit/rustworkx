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
        TokenSwapper {
            graph,
            input_mapping: mapping,
            trials: match trials {
                Some(trials) => trials,
                None => 4,
            },
            rng_seed: seed,
            node_map: HashMap::with_capacity(graph.node_count()),
            rev_node_map: HashMap::with_capacity(graph.node_count()),
        }
    }

    fn map(&mut self) -> Vec<Swap> {
        let num_nodes = self.graph.node_count();
        let num_edges = self.graph.edge_count();

        // Directed graph with nodes matching ``graph`` and
        // edges for neighbors closer than nodes
        let mut digraph = StableGraph::with_capacity(num_nodes, num_edges);
        // Same as digraph with no self edges
        let mut sub_digraph = StableGraph::with_capacity(num_nodes, num_edges);
        // A list of nodes that are remaining to be tried
        let mut todo_nodes = Vec::with_capacity(self.input_mapping.len());

        let mut rng_seed: Pcg64 = match self.rng_seed {
            Some(rng_seed) => Pcg64::seed_from_u64(rng_seed),
            None => Pcg64::from_entropy(),
        };
        // Add nodes to the digraph/sub_digraph and build the node maps
        for node in self.graph.node_identifiers() {
            let index = digraph.add_node(());
            sub_digraph.add_node(());
            self.node_map.insert(node, index);
            self.rev_node_map.insert(index, node);
        }
        // The input mapping in HashMap form using NodeIndex
        let mut tokens: HashMap<NodeIndex, NodeIndex> = self
            .input_mapping
            .iter()
            .map(|(k, v)| (self.node_map[&k], self.node_map[&v]))
            .collect();
        for (node, dest) in &tokens {
            if node != dest {
                todo_nodes.push(*node);
            }
        }
        // Add initial edges to the digraph/sub_digraph
        for node in self.graph.node_identifiers() {
            self.add_token_edges(
                self.node_map[&node],
                &mut digraph,
                &mut sub_digraph,
                &mut tokens,
            );
        }
        // Do the trials and build a results Vec
        let mut trial_results: Vec<Vec<Swap>> = Vec::new();
        for _ in 0..self.trials {
            let results = self.trial_map(
                &mut digraph.clone(),
                &mut sub_digraph.clone(),
                &mut tokens.clone(),
                &mut todo_nodes.clone(),
                &mut rng_seed,
            );
            trial_results.push(results);
        }
        // Build the first results Vec until a 0 len result is found
        let mut first_results: Vec<Vec<Swap>> = Vec::new();
        for result in trial_results {
            if result.len() == 0 {
                first_results.push(result);
                break;
            }
            first_results.push(result);
        }
        // Return the min of the first results
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
        digraph: &mut StableGraph<(), (), Directed>,
        sub_digraph: &mut StableGraph<(), (), Directed>,
        tokens: &mut HashMap<NodeIndex, NodeIndex>,
    ) {
        if !(tokens.contains_key(&node)) {
            return;
        }
        if tokens[&node] == node {
            digraph.update_edge(node, node, ());
            return;
        }
        let id_node = self.rev_node_map[&node];
        let id_token = self.rev_node_map[&tokens[&node]];
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
                digraph.update_edge(node, neighbor, ());
                sub_digraph.update_edge(node, neighbor, ());
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
            self.add_token_edges(node, digraph, sub_digraph, tokens);
            if tokens.contains_key(&node) && tokens[&node] != node {
                if !todo_nodes.contains(&node) {
                    todo_nodes.push(node);
                }
            } else if todo_nodes.contains(&node) {
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

    #[test]
    fn test_small_swap() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]);
        let mapping = HashMap::from([
            (NodeIndex::new(0), NodeIndex::new(7)),
            (NodeIndex::new(1), NodeIndex::new(6)),
            (NodeIndex::new(2), NodeIndex::new(5)),
            (NodeIndex::new(3), NodeIndex::new(4)),
            (NodeIndex::new(4), NodeIndex::new(3)),
            (NodeIndex::new(5), NodeIndex::new(2)),
            (NodeIndex::new(6), NodeIndex::new(1)),
            (NodeIndex::new(7), NodeIndex::new(0)),
        ]);
        // let swaps = vec!([
        //     (NodeIndex::new(0), NodeIndex::new(0)),
        //     (NodeIndex::new(1), NodeIndex::new(1)),
        //     (NodeIndex::new(2), NodeIndex::new(2)),
        //     (NodeIndex::new(3), NodeIndex::new(3)),
        //     (NodeIndex::new(4), NodeIndex::new(4)),
        //     (NodeIndex::new(5), NodeIndex::new(5)),
        //     (NodeIndex::new(6), NodeIndex::new(6)),
        //     (NodeIndex::new(7), NodeIndex::new(7)),
        // ]);
        // Do the token swap
        let output = token_swapper(&g, mapping, Some(4), Some(4));
        println!("{:?}", output);
        assert_eq!(3, 4);
        //assert_eq!(output, swaps);
    }
}
// def test_small(self) -> None:
//     """Test an inverting permutation on a small path graph of size 8"""
//     graph = rx.generators.path_graph(8)
//     permutation = {i: 7 - i for i in range(8)}
//     swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]

//     out = list(swapper.map(permutation))
//     util.swap_permutation([out], permutation)
//     self.assertEqual({i: i for i in range(8)}, permutation)

// def test_bug1(self) -> None:
//     """Tests for a bug that occured in happy swap chains of length >2."""
//     graph = rx.PyGraph()
//     graph.extend_from_edge_list(
//         [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (3, 6)]
//     )
//     permutation = {0: 4, 1: 0, 2: 3, 3: 6, 4: 2, 6: 1}
//     swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]

//     out = list(swapper.map(permutation))
//     util.swap_permutation([out], permutation)
//     self.assertEqual({i: i for i in permutation}, permutation)

// def test_partial_simple(self) -> None:
//     """Test a partial mapping on a small graph."""
//     graph = rx.generators.path_graph(4)
//     mapping = {0: 3}
//     swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]
//     out = list(swapper.map(mapping))
//     self.assertEqual(3, len(out))
//     util.swap_permutation([out], mapping, allow_missing_keys=True)
//     self.assertEqual({3: 3}, mapping)

// def test_partial_small(self) -> None:
//     """Test an partial inverting permutation on a small path graph of size 5"""
//     graph = rx.generators.path_graph(4)
//     permutation = {i: 3 - i for i in range(2)}
//     swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]

//     out = list(swapper.map(permutation))
//     self.assertEqual(5, len(out))
//     util.swap_permutation([out], permutation, allow_missing_keys=True)
//     self.assertEqual({i: i for i in permutation.values()}, permutation)

// def test_large_partial_random(self) -> None:
//     """Test a random (partial) mapping on a large randomly generated graph"""
//     size = 100
//     # Note that graph may have "gaps" in the node counts, i.e. the numbering is noncontiguous.
//     graph = rx.undirected_gnm_random_graph(size, size**2 // 10)
//     for i in graph.node_indexes():
//         try:
//             graph.remove_edge(i, i)  # Remove self-loops.
//         except rx.NoEdgeBetweenNodes:
//             continue
//     # Make sure the graph is connected by adding C_n
//     graph.add_edges_from_no_data([(i, i + 1) for i in range(len(graph) - 1)])
//     swapper = ApproximateTokenSwapper(graph)  # type: ApproximateTokenSwapper[int]

//     # Generate a randomized permutation.
//     rand_perm = random.permutation(graph.nodes())
//     permutation = dict(zip(graph.nodes(), rand_perm))
//     mapping = dict(itertools.islice(permutation.items(), 0, size, 2))  # Drop every 2nd element.

//     out = list(swapper.map(mapping, trials=40))
//     util.swap_permutation([out], mapping, allow_missing_keys=True)
//     self.assertEqual({i: i for i in mapping.values()}, mapping)
