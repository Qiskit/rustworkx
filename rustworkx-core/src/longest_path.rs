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

use petgraph::algo;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Calculates the longest path in a directed acyclic graph (DAG).
///
/// This function computes the longest path by weight in a given DAG. It will return the longest path
/// along with its total weight, or `None` if the graph contains cycles which make the longest path
/// computation undefined.
///
/// # Arguments
/// * `graph`: Reference to a `DiGraph` representing the DAG.
/// * `weight_fn`: Function to determine the weight of each edge, given source and target `NodeIndex` and edge weight.
///
/// # Type Parameters
/// * `N`: The node type.
/// * `E`: The edge type.
/// * `F`: Type of the weight function.
/// * `T`: The type of the edge weight. Must support ordering, addition, zero initialization, and copying.
///
/// # Returns
/// * `None` if the graph contains a cycle.
/// * `Some((Vec<NodeIndex>, T))` representing the longest path as a sequence of nodes and its total weight.
///
/// # Example
/// ```rust
/// use petgraph::graph::{DiGraph, NodeIndex};
/// use rustworkx_core::longest_path::longest_path;
///
/// let mut graph = DiGraph::new();
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// let n3 = graph.add_node(());
/// graph.add_edge(n1, n2, 3);
/// graph.add_edge(n1, n3, 2);
/// graph.add_edge(n2, n3, 1);
///
/// let weight_fn = |_, _, &weight: &i32| weight;
/// let result = longest_path(&graph, weight_fn);
/// assert_eq!(result, Some((vec![n1, n2, n3], 4)));
/// ```
pub fn longest_path<N, E, F, T>(graph: &DiGraph<N, E>, mut weight_fn: F) -> Option<(Vec<NodeIndex>, T)>
where
    F: FnMut(NodeIndex, NodeIndex, &E) -> T,
    T: Ord + Copy + std::ops::Add<Output = T> + Default,
{
    // Check for cycles; return None if any are found
    if algo::is_cyclic_directed(graph) {
        return None;
    }

    let mut path: Vec<NodeIndex> = Vec::new(); // This will store the longest path
    let nodes = match algo::toposort(graph, None) { // Topologically sort the nodes
        Ok(nodes) => nodes,
        Err(_) => return None, // Should not happen since we check for cycles above
    };

    // Handle the trivial case where the graph is empty
    if nodes.is_empty() {
        return Some((path, T::default()));
    }

    let mut dist: HashMap<NodeIndex, (T, NodeIndex)> = HashMap::new(); // Distance map from node to (weight, prev_node)

    // Iterate over nodes in topological order
    for node in nodes {
        let parents = graph.edges_directed(node, petgraph::Direction::Incoming);
        let mut us: Vec<(T, NodeIndex)> = Vec::new(); // This will store weights from each parent
        // Process each parent edge of the current node
        for edge in parents {
            let (p_node, target, weight) = (edge.source(), edge.target(), edge.weight());
            let weight = weight_fn(p_node, target, weight); // Compute the weight using the provided function
            let length = *dist.get(&p_node).map_or(&T::default(), |(d, _)| d) + weight;
            us.push((length, p_node));
        }
        // Determine the longest path to this node from any of its parents
        if let Some(maxu) = us.iter().max_by_key(|x| x.0) {
            dist.insert(node, *maxu);
        } else {
            dist.insert(node, (T::default(), node));
        }
    }

    // Find the node that has the maximum distance
    let first = dist.keys().max_by_key(|&n| dist[n].0).unwrap();
    let mut v = *first;
    let mut u: Option<NodeIndex> = None;
    // Backtrack from this node to find the path
    while match u {
        Some(u) => u != v,
        None => true,
    } {
        path.push(v);
        u = Some(v);
        v = dist[&v].1;
    }
    path.reverse(); // Reverse the path to get the correct order
    let path_weight = dist[first].0; // The total weight of the longest path
    Some((path, path_weight))
}

#[cfg(test)]
mod test_longest_path {
    use super::*;
    use petgraph::graph::{DiGraph, NodeIndex};

    /// Tests an empty graph to ensure it returns a path length of zero.
    #[test]
    fn test_empty_graph() {
        let graph: DiGraph<(), ()> = DiGraph::new();
        let weight_fn = |_: NodeIndex, _: NodeIndex, _: &()| 0;
        assert_eq!(longest_path(&graph, weight_fn), Some((vec![], 0)));
    }

    /// Tests a graph with a single node to ensure it correctly handles graphs without edges.
    #[test]
    fn test_single_node() {
        let mut graph = DiGraph::new();
        graph.add_node(());

        let weight_fn = |_: NodeIndex, _: NodeIndex, _: &()| 1;
        assert_eq!(longest_path(&graph, weight_fn), Some((vec![NodeIndex::new(0)], 0)));
    }

    /// Tests a simple path from one node to another to ensure it computes path lengths correctly.
    #[test]
    fn test_simple_path() {
        let mut graph = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n1, n2, 1);

        let weight_fn = |_, _, &w: &i32| w;
        assert_eq!(longest_path(&graph, weight_fn), Some((vec![n1, n2], 1)));
    }

    /// Tests the path from the function description to ensure it computes the correct longest path.
    #[test]
    fn example_longest_path() {
        let mut graph = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        graph.add_edge(n1, n2, 3);
        graph.add_edge(n1, n3, 2);
        graph.add_edge(n2, n3, 1);

        let weight_fn = |_, _, &weight: &i32| weight;
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Some((vec![n1, n2, n3], 4)));
    }

    /// Tests a graph with multiple paths to ensure it selects the longest path correctly.
    #[test]
    fn test_dag_with_multiple_paths() {
        let mut graph = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        graph.add_edge(n1, n2, 1);
        graph.add_edge(n1, n3, 2);
        graph.add_edge(n2, n3, 1);

        let weight_fn = |_, _, &w: &i32| w;
        assert_eq!(longest_path(&graph, weight_fn), Some((vec![n1, n3], 2)));
    }

    /// Tests a graph with a cycle to ensure it returns None, as cycles invalidate longest path calculations in a DAG.
    #[test]
    fn test_graph_with_cycle() {
        let mut graph = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n1, n2, 1);
        graph.add_edge(n2, n1, 1); // Creates a cycle

        let weight_fn = |_, _, &w: &i32| w;
        assert_eq!(longest_path(&graph, weight_fn), None);
    }
}
