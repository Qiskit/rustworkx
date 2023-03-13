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
use std::hash::Hash;

use hashbrown::HashMap;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::{
    EdgeCount, GraphBase, IntoEdges, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount,
    NodeIndexable, Visitable,
};
use petgraph::Directed;
use petgraph::Direction::{Incoming, Outgoing};

use crate::connectivity::find_cycle;
use crate::dictmap::*;
use crate::shortest_path::dijkstra;
use crate::traversal::dfs_edges;

type Swap = (NodeIndex, NodeIndex);
type Edge = (NodeIndex, NodeIndex);

struct TokenSwapper<G: GraphBase>
where
    G::NodeId: Eq + Hash,
{
    // The input graph
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
        + IntoNodeIdentifiers,
    G::NodeId: Hash + Eq,
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
            trials: trials.unwrap_or(4),
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
            .map(|(k, v)| (self.node_map[k], self.node_map[v]))
            .collect();

        // todo_nodes are all the mapping entries where left != right
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
        // Do the trials
        (0..self.trials)
            .map(|_| {
                self.trial_map(
                    digraph.clone(),
                    sub_digraph.clone(),
                    tokens.clone(),
                    todo_nodes.clone(),
                    &mut rng_seed,
                )
            })
            .min_by_key(|result| result.len())
            .unwrap()
    }

    fn add_token_edges(
        &mut self,
        node: NodeIndex,
        digraph: &mut StableGraph<(), (), Directed>,
        sub_digraph: &mut StableGraph<(), (), Directed>,
        tokens: &mut HashMap<NodeIndex, NodeIndex>,
    ) {
        // Adds an edge to digraph if distance from the token to a neighbor is
        // less than distance from token to node. sub_digraph is same except
        // for self-edges.
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
            let dist_neighbor: DictMap<G::NodeId, usize> = dijkstra(
                &self.graph,
                id_neighbor,
                Some(id_token),
                |_| Ok::<usize, &str>(1),
                None,
            )
            .unwrap();

            let dist_node: DictMap<G::NodeId, usize> = dijkstra(
                &self.graph,
                id_node,
                Some(id_token),
                |_| Ok::<usize, &str>(1),
                None,
            )
            .unwrap();

            if dist_neighbor[&id_token] < dist_node[&id_token] {
                digraph.update_edge(node, neighbor, ());
                sub_digraph.update_edge(node, neighbor, ());
            }
        }
    }

    fn trial_map(
        &mut self,
        mut digraph: StableGraph<(), (), Directed>,
        mut sub_digraph: StableGraph<(), (), Directed>,
        mut tokens: HashMap<NodeIndex, NodeIndex>,
        mut todo_nodes: Vec<NodeIndex>,
        rng_seed: &mut Pcg64,
    ) -> Vec<Swap> {
        // Create a random trial list of swaps to move tokens to optimal positions
        let mut steps = 0;
        let mut swap_edges: Vec<Swap> = vec![];
        while !todo_nodes.is_empty() && steps <= 4 * digraph.node_count().pow(2) {
            // Choose a random todo_node
            let between = Uniform::new(0, todo_nodes.len());
            let random: usize = between.sample(rng_seed);
            let todo_node = todo_nodes[random];

            // If there's a cycle in sub_digraph, add it to swap_edges and do swap
            let cycle = find_cycle(&sub_digraph, Some(todo_node));
            if !cycle.is_empty() {
                for edge in cycle[1..].iter().rev() {
                    swap_edges.push(*edge);
                    self.swap(
                        edge.0,
                        edge.1,
                        &mut digraph,
                        &mut sub_digraph,
                        &mut tokens,
                        &mut todo_nodes,
                    );
                }
                steps += cycle.len() - 1;
            // If there's no cycle, see if there's an edge target that matches a token key.
            // If so, add to swap_edges and do swap
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
                            &mut digraph,
                            &mut sub_digraph,
                            &mut tokens,
                            &mut todo_nodes,
                        );
                        steps += 1;
                        found = true;
                        break;
                    }
                }
                // If none found, look for cycle in digraph which will result in
                // an unhappy node. Look for a predecessor and add node and pred
                // to swap_edges and do swap
                if !found {
                    let cycle: Vec<Edge> = find_cycle(&digraph, Some(todo_node));
                    let unhappy_node = cycle[0].0;
                    let mut found = false;
                    let di2 = &mut digraph.clone();
                    for predecessor in di2.neighbors_directed(unhappy_node, Incoming) {
                        if predecessor != unhappy_node {
                            swap_edges.push((unhappy_node, predecessor));
                            self.swap(
                                unhappy_node,
                                predecessor,
                                &mut digraph,
                                &mut sub_digraph,
                                &mut tokens,
                                &mut todo_nodes,
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
        if !todo_nodes.is_empty() {
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
        // Get token values for the 2 nodes and remove them
        let token1 = tokens.remove(&node1);
        let token2 = tokens.remove(&node2);

        // Swap the token edge values
        if let Some(t2) = token2 {
            tokens.insert(node1, t2);
        }
        if let Some(t1) = token1 {
            tokens.insert(node2, t1);
        }
        // For each node, remove the (node, successor) from digraph and
        // sub_digraph. Then add new token edges back in.
        for node in [node1, node2] {
            let edge_nodes: Vec<(NodeIndex, NodeIndex)> = digraph
                .neighbors_directed(node, Outgoing)
                .map(|successor| (node, successor))
                .collect();
            for (edge_node1, edge_node2) in edge_nodes {
                let edge = digraph.find_edge(edge_node1, edge_node2).unwrap();
                digraph.remove_edge(edge);
            }
            let edge_nodes: Vec<(NodeIndex, NodeIndex)> = sub_digraph
                .neighbors_directed(node, Outgoing)
                .map(|successor| (node, successor))
                .collect();
            for (edge_node1, edge_node2) in edge_nodes {
                let edge = sub_digraph.find_edge(edge_node1, edge_node2).unwrap();
                sub_digraph.remove_edge(edge);
            }
            self.add_token_edges(node, digraph, sub_digraph, tokens);

            // If a node is a token key and not equal to the value, add it to todo_nodes
            if tokens.contains_key(&node) && tokens[&node] != node {
                if !todo_nodes.contains(&node) {
                    todo_nodes.push(node);
                }
            // Otherwise if node is in todo_nodes, remove it
            } else if todo_nodes.contains(&node) {
                todo_nodes.swap_remove(todo_nodes.iter().position(|x| *x == node).unwrap());
            }
        }
    }
}

/// Module to perform an approximately optimal Token Swapping algorithm. Supports partial
/// mappings (i.e. not-permutations) for graphs with missing tokens.
///
/// Based on the paper: Approximation and Hardness for Token Swapping by Miltzow et al. (2016)
/// ArXiV: <https://arxiv.org/abs/1602.05150>
///
/// Arguments:
///
/// * `graph` - The graph on which to perform the token swapping.
/// * `mapping` - A partial mapping to be implemented in swaps.
/// * `trials` - Optional number of trials. If None, defaults to 4.
/// * `seed` - Optional integer seed. If None, the internal rng will be initialized from system entropy.
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
        + IntoNodeIdentifiers,
    G::NodeId: Hash + Eq,
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

    fn do_swap(mapping: &mut HashMap<NodeIndex, NodeIndex>, swaps: &Vec<(NodeIndex, NodeIndex)>) {
        // Apply the swaps to the mapping to get final result
        for (swap1, swap2) in swaps {
            //Need to create temp nodes in case of partial mapping
            let mut temp_node1: Option<NodeIndex> = None;
            let mut temp_node2: Option<NodeIndex> = None;
            if mapping.contains_key(swap1) {
                temp_node1 = Some(mapping[swap1]);
                mapping.remove(swap1);
            }
            if mapping.contains_key(swap2) {
                temp_node2 = Some(mapping[swap2]);
                mapping.remove(swap2);
            }
            if let Some(t1) = temp_node1 {
                mapping.insert(*swap2, t1);
            }
            if let Some(t2) = temp_node2 {
                mapping.insert(*swap1, t2);
            }
        }
    }

    #[test]
    fn test_simple_swap() {
        // Simple arbitrary swap
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3)]);
        let mapping = HashMap::from([
            (NodeIndex::new(0), NodeIndex::new(0)),
            (NodeIndex::new(1), NodeIndex::new(3)),
            (NodeIndex::new(3), NodeIndex::new(1)),
            (NodeIndex::new(2), NodeIndex::new(2)),
        ]);
        let swaps = token_swapper(&g, mapping, Some(4), Some(4));
        assert_eq!(3, swaps.len());
    }

    #[test]
    fn test_small_swap() {
        // Reverse all small swap
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
        ]);
        let mut mapping = HashMap::with_capacity(8);
        for i in 0..8 {
            mapping.insert(NodeIndex::new(i), NodeIndex::new(7 - i));
        }
        // Do the token swap
        let mut new_map = mapping.clone();
        let swaps = token_swapper(&g, mapping, Some(4), Some(4));
        do_swap(&mut new_map, &swaps);
        let mut expected = HashMap::with_capacity(8);
        for i in 0..8 {
            expected.insert(NodeIndex::new(i), NodeIndex::new(i));
        }
        assert_eq!(expected, new_map);
    }

    #[test]
    fn test_happy_swap_chain() {
        // Reverse all happy swap chain > 2
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 6),
        ]);
        let mapping = HashMap::from([
            (NodeIndex::new(0), NodeIndex::new(4)),
            (NodeIndex::new(1), NodeIndex::new(0)),
            (NodeIndex::new(2), NodeIndex::new(3)),
            (NodeIndex::new(3), NodeIndex::new(6)),
            (NodeIndex::new(4), NodeIndex::new(2)),
            (NodeIndex::new(6), NodeIndex::new(1)),
        ]);
        // Do the token swap
        let mut new_map = mapping.clone();
        let swaps = token_swapper(&g, mapping, Some(4), Some(4));
        do_swap(&mut new_map, &swaps);
        let mut expected = HashMap::with_capacity(6);
        for i in (0..5).chain(6..7) {
            expected.insert(NodeIndex::new(i), NodeIndex::new(i));
        }
        assert_eq!(expected, new_map);
    }

    #[test]
    fn test_partial_simple() {
        // Simple partial swap
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3)]);
        let mapping = HashMap::from([(NodeIndex::new(0), NodeIndex::new(3))]);
        let mut new_map = mapping.clone();
        let swaps = token_swapper(&g, mapping, Some(4), Some(4));
        do_swap(&mut new_map, &swaps);
        let mut expected = HashMap::with_capacity(4);
        expected.insert(NodeIndex::new(3), NodeIndex::new(3));
        assert_eq!(expected, new_map);
    }

    #[test]
    fn test_partial_small() {
        // Partial inverting on small path graph
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3)]);
        let mapping = HashMap::from([
            (NodeIndex::new(0), NodeIndex::new(3)),
            (NodeIndex::new(1), NodeIndex::new(2)),
        ]);
        let mut new_map = mapping.clone();
        let swaps = token_swapper(&g, mapping, Some(4), Some(4));
        do_swap(&mut new_map, &swaps);
        let expected = HashMap::from([
            (NodeIndex::new(2), NodeIndex::new(2)),
            (NodeIndex::new(3), NodeIndex::new(3)),
        ]);
        assert_eq!(5, swaps.len());
        assert_eq!(expected, new_map);
    }
}

// TODO: Port this test when rustworkx-core adds random graphs

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
