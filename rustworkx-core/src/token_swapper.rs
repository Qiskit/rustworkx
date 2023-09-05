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

use rand::distributions::{Standard, Uniform};
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::error::Error;
use std::fmt;
use std::hash::Hash;

use hashbrown::HashMap;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::{
    EdgeCount, GraphBase, IntoEdges, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount,
    NodeIndexable, Visitable,
};
use petgraph::Directed;
use petgraph::Direction::{Incoming, Outgoing};
use rayon::prelude::*;
use rayon_cond::CondIterator;

use crate::connectivity::find_cycle;
use crate::dictmap::*;
use crate::shortest_path::dijkstra;
use crate::traversal::dfs_edges;

type Swap = (NodeIndex, NodeIndex);
type Edge = (NodeIndex, NodeIndex);

/// Error returned by token swapper if the request mapping
/// is impossible
#[derive(Debug, PartialEq, Eq, Ord, PartialOrd, Copy, Clone)]
pub struct MapNotPossible;

impl Error for MapNotPossible {}
impl fmt::Display for MapNotPossible {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "No mapping possible.")
    }
}

struct TokenSwapper<G: GraphBase>
where
    G::NodeId: Eq + Hash,
{
    // The input graph
    graph: G,
    // The user-supplied mapping to use for swapping tokens
    mapping: HashMap<G::NodeId, G::NodeId>,
    // Number of trials
    trials: usize,
    // Seed for random selection of a node for a trial
    seed: Option<u64>,
    // Threshold for how many nodes will trigger parallel iterator
    parallel_threshold: usize,
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
        + Send
        + Sync,
    G::NodeId: Hash + Eq + Send + Sync,
{
    fn new(
        graph: G,
        mapping: HashMap<G::NodeId, G::NodeId>,
        trials: Option<usize>,
        seed: Option<u64>,
        parallel_threshold: Option<usize>,
    ) -> Self {
        TokenSwapper {
            graph,
            mapping,
            trials: trials.unwrap_or(4),
            seed,
            parallel_threshold: parallel_threshold.unwrap_or(50),
            node_map: HashMap::with_capacity(graph.node_count()),
            rev_node_map: HashMap::with_capacity(graph.node_count()),
        }
    }

    fn map(&mut self) -> Result<Vec<Swap>, MapNotPossible> {
        let num_nodes = self.graph.node_bound();
        let num_edges = self.graph.edge_count();

        // Directed graph with nodes matching ``graph`` and
        // edges for neighbors closer than nodes
        let mut digraph = StableGraph::with_capacity(num_nodes, num_edges);

        // First fill the digraph with nodes. Then since it's a stable graph,
        // must go through and remove nodes that were removed in original graph
        for _ in 0..self.graph.node_bound() {
            digraph.add_node(());
        }
        let mut count: usize = 0;
        for gnode in self.graph.node_identifiers() {
            let gidx = self.graph.to_index(gnode);
            if gidx != count {
                for idx in count..gidx {
                    digraph.remove_node(NodeIndex::new(idx));
                }
                count = gidx;
            }
            count += 1;
        }

        // Create maps between NodeId and NodeIndex
        for node in self.graph.node_identifiers() {
            self.node_map
                .insert(node, NodeIndex::new(self.graph.to_index(node)));
            self.rev_node_map
                .insert(NodeIndex::new(self.graph.to_index(node)), node);
        }
        // sub will become same as digraph but with no self edges in add_token_edges
        let mut sub_digraph = digraph.clone();

        // The mapping in HashMap form using NodeIndex
        let mut tokens: HashMap<NodeIndex, NodeIndex> = self
            .mapping
            .iter()
            .map(|(k, v)| (self.node_map[k], self.node_map[v]))
            .collect();

        // todo_nodes are all the mapping entries where left != right
        let mut todo_nodes: Vec<NodeIndex> = tokens
            .iter()
            .filter_map(|(node, dest)| if node != dest { Some(*node) } else { None })
            .collect();
        todo_nodes.par_sort();

        // Add initial edges to the digraph/sub_digraph
        for node in self.graph.node_identifiers() {
            self.add_token_edges(
                self.node_map[&node],
                &mut digraph,
                &mut sub_digraph,
                &mut tokens,
            )?;
        }
        // First collect the self.trial number of random numbers
        // into a Vec based on the given seed
        let outer_rng: Pcg64 = match self.seed {
            Some(rng_seed) => Pcg64::seed_from_u64(rng_seed),
            None => Pcg64::from_entropy(),
        };
        let trial_seeds_vec: Vec<u64> =
            outer_rng.sample_iter(&Standard).take(self.trials).collect();

        CondIterator::new(
            trial_seeds_vec,
            self.graph.node_count() >= self.parallel_threshold,
        )
        .map(|trial_seed| {
            self.trial_map(
                digraph.clone(),
                sub_digraph.clone(),
                tokens.clone(),
                todo_nodes.clone(),
                trial_seed,
            )
        })
        .min_by_key(|result| match result {
            Ok(res) => Ok(res.len()),
            Err(e) => Err(*e),
        })
        .unwrap()
    }

    fn add_token_edges(
        &self,
        node: NodeIndex,
        digraph: &mut StableGraph<(), (), Directed>,
        sub_digraph: &mut StableGraph<(), (), Directed>,
        tokens: &mut HashMap<NodeIndex, NodeIndex>,
    ) -> Result<(), MapNotPossible> {
        // Adds an edge to digraph if distance from the token to a neighbor is
        // less than distance from token to node. sub_digraph is same except
        // for self-edges.
        if !(tokens.contains_key(&node)) {
            return Ok(());
        }
        if tokens[&node] == node {
            digraph.update_edge(node, node, ());
            return Ok(());
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
            let neigh_dist = dist_neighbor.get(&id_token);
            let node_dist = dist_node.get(&id_token);
            if neigh_dist.is_none() {
                return Err(MapNotPossible {});
            }
            if node_dist.is_none() {
                return Err(MapNotPossible {});
            }

            if neigh_dist < node_dist {
                digraph.update_edge(node, neighbor, ());
                sub_digraph.update_edge(node, neighbor, ());
            }
        }
        Ok(())
    }

    fn trial_map(
        &self,
        mut digraph: StableGraph<(), (), Directed>,
        mut sub_digraph: StableGraph<(), (), Directed>,
        mut tokens: HashMap<NodeIndex, NodeIndex>,
        mut todo_nodes: Vec<NodeIndex>,
        trial_seed: u64,
    ) -> Result<Vec<Swap>, MapNotPossible> {
        // Create a random trial list of swaps to move tokens to optimal positions
        let mut steps = 0;
        let mut swap_edges: Vec<Swap> = vec![];
        let mut rng_seed: Pcg64 = Pcg64::seed_from_u64(trial_seed);
        while !todo_nodes.is_empty() && steps <= 4 * digraph.node_count().pow(2) {
            // Choose a random todo_node
            let between = Uniform::new(0, todo_nodes.len());
            let random: usize = between.sample(&mut rng_seed);
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
                    )?;
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
                        )?;
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
                            )?;
                            steps += 1;
                            found = true;
                            break;
                        }
                    }
                    assert!(
                        found,
                        "The token swap process has ended unexpectedly, this points to a bug in rustworkx, please open an issue."
                    );
                }
            }
        }
        assert!(
            todo_nodes.is_empty(),
            "The output final swap map is incomplete, this points to a bug in rustworkx, please open an issue."
        );
        Ok(swap_edges)
    }

    fn swap(
        &self,
        node1: NodeIndex,
        node2: NodeIndex,
        digraph: &mut StableGraph<(), (), Directed>,
        sub_digraph: &mut StableGraph<(), (), Directed>,
        tokens: &mut HashMap<NodeIndex, NodeIndex>,
        todo_nodes: &mut Vec<NodeIndex>,
    ) -> Result<(), MapNotPossible> {
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
            self.add_token_edges(node, digraph, sub_digraph, tokens)?;

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
        Ok(())
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
/// * `parallel_threshold` - Optional integer for the number of nodes in the graph that will
///     trigger the use of parallel threads. If the number of nodes in the graph is less than this value
///     it will run in a single thread. The default value is 50.
///
/// It returns a list of tuples representing the swaps to perform. The result will be an
/// `Err(MapNotPossible)` if the `token_swapper()` function can't find a mapping.
///
/// This function is multithreaded and will launch a thread pool with threads equal to
/// the number of CPUs by default. You can tune the number of threads with
/// the ``RAYON_NUM_THREADS`` environment variable. For example, setting ``RAYON_NUM_THREADS=4``
/// would limit the thread pool to 4 threads.
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
///  let output = token_swapper(&g, mapping, Some(4), Some(4), Some(50)).expect("Swap mapping failed.");
///  assert_eq!(3, output.len());
///
/// ```

pub fn token_swapper<G>(
    graph: G,
    mapping: HashMap<G::NodeId, G::NodeId>,
    trials: Option<usize>,
    seed: Option<u64>,
    parallel_threshold: Option<usize>,
) -> Result<Vec<Swap>, MapNotPossible>
where
    G: NodeCount
        + EdgeCount
        + IntoEdges
        + Visitable
        + NodeIndexable
        + IntoNeighborsDirected
        + IntoNodeIdentifiers
        + Send
        + Sync,
    G::NodeId: Hash + Eq + Send + Sync,
{
    let mut swapper = TokenSwapper::new(graph, mapping, trials, seed, parallel_threshold);
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
        let swaps = token_swapper(&g, mapping, Some(4), Some(4), Some(50));
        assert_eq!(3, swaps.expect("swap mapping errored").len());
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
        let swaps =
            token_swapper(&g, mapping, Some(4), Some(4), Some(50)).expect("swap mapping errored");
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
        let swaps =
            token_swapper(&g, mapping, Some(4), Some(4), Some(50)).expect("swap mapping errored");
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
        let swaps =
            token_swapper(&g, mapping, Some(4), Some(4), Some(1)).expect("swap mapping errored");
        do_swap(&mut new_map, &swaps);
        let mut expected = HashMap::with_capacity(4);
        expected.insert(NodeIndex::new(3), NodeIndex::new(3));
        assert_eq!(expected, new_map);
    }

    #[test]
    fn test_partial_simple_remove_node() {
        // Simple partial swap
        let mut g =
            petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let mapping = HashMap::from([(NodeIndex::new(0), NodeIndex::new(3))]);
        g.remove_node(NodeIndex::new(2));
        g.add_edge(NodeIndex::new(1), NodeIndex::new(3), ());
        let mut new_map = mapping.clone();
        let swaps =
            token_swapper(&g, mapping, Some(4), Some(4), Some(1)).expect("swap mapping errored");
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
        let swaps =
            token_swapper(&g, mapping, Some(4), Some(4), Some(50)).expect("swap mapping errored");
        do_swap(&mut new_map, &swaps);
        let expected = HashMap::from([
            (NodeIndex::new(2), NodeIndex::new(2)),
            (NodeIndex::new(3), NodeIndex::new(3)),
        ]);
        assert_eq!(5, swaps.len());
        assert_eq!(expected, new_map);
    }

    #[test]
    fn test_large_partial_random() {
        // Test a random (partial) mapping on a large randomly generated graph
        use crate::generators::gnm_random_graph;
        use rand::prelude::*;
        use rand_pcg::Pcg64;
        use std::iter::zip;

        let mut rng: Pcg64 = Pcg64::seed_from_u64(4);

        // Note that graph may have "gaps" in the node counts, i.e. the numbering is noncontiguous.
        let size = 100;
        let mut g: petgraph::stable_graph::StableGraph<(), ()> =
            gnm_random_graph(size, size.pow(2) / 10, Some(4), || (), || ()).unwrap();

        // Remove self-loops
        let nodes: Vec<_> = g.node_indices().collect();
        for node in nodes {
            let edge = g.find_edge(node, node);
            if edge.is_some() {
                g.remove_edge(edge.unwrap());
            }
        }
        // Make sure the graph is connected by adding C_n
        for i in 0..(g.node_count() - 1) {
            g.add_edge(NodeIndex::new(i), NodeIndex::new(i + 1), ());
        }

        // Get node indices and randomly shuffle
        let mut mapped_nodes: Vec<usize> = g.node_indices().map(|node| node.index()).collect();
        let nodes = mapped_nodes.clone();
        mapped_nodes.shuffle(&mut rng);

        // Zip nodes and shuffled nodes and remove every other one
        let mut mapping: Vec<(usize, usize)> = zip(nodes, mapped_nodes).collect();
        mapping.retain(|(a, _)| a % 2 == 0);

        // Convert mapping to HashMap of NodeIndex's
        let mapping: HashMap<NodeIndex, NodeIndex> = mapping
            .into_iter()
            .map(|(a, b)| (NodeIndex::new(a), NodeIndex::new(b)))
            .collect();
        let mut new_map = mapping.clone();
        let expected: HashMap<NodeIndex, NodeIndex> =
            mapping.values().map(|val| (*val, *val)).collect();

        let swaps =
            token_swapper(&g, mapping, Some(4), Some(4), Some(50)).expect("swap mapping errored");
        do_swap(&mut new_map, &swaps);
        assert_eq!(expected, new_map)
    }

    #[test]
    fn test_disjoint_graph_works() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (2, 3)]);
        let mapping = HashMap::from([
            (NodeIndex::new(1), NodeIndex::new(0)),
            (NodeIndex::new(0), NodeIndex::new(1)),
            (NodeIndex::new(2), NodeIndex::new(3)),
            (NodeIndex::new(3), NodeIndex::new(2)),
        ]);
        let mut new_map = mapping.clone();
        let swaps =
            token_swapper(&g, mapping, Some(10), Some(4), Some(50)).expect("swap mapping errored");
        do_swap(&mut new_map, &swaps);
        let expected = HashMap::from([
            (NodeIndex::new(2), NodeIndex::new(2)),
            (NodeIndex::new(3), NodeIndex::new(3)),
            (NodeIndex::new(1), NodeIndex::new(1)),
            (NodeIndex::new(0), NodeIndex::new(0)),
        ]);
        assert_eq!(2, swaps.len());
        assert_eq!(expected, new_map);
    }

    #[test]
    fn test_disjoint_graph_fails() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (2, 3)]);
        let mapping = HashMap::from([
            (NodeIndex::new(2), NodeIndex::new(0)),
            (NodeIndex::new(1), NodeIndex::new(1)),
            (NodeIndex::new(0), NodeIndex::new(2)),
            (NodeIndex::new(3), NodeIndex::new(3)),
        ]);
        match token_swapper(&g, mapping, Some(10), Some(4), Some(50)) {
            Ok(_) => panic!("This should error"),
            Err(_) => (),
        };
    }
}
