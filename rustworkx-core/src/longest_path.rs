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

use hashbrown::HashMap;
use num_traits::{Num, Zero};
use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::visit::{
    EdgeRef, GraphBase, GraphProp, IntoEdgesDirected, IntoNeighborsDirected, IntoNodeIdentifiers,
    Visitable,
};
use petgraph::Directed;

/// Calculates the longest path in a directed acyclic graph (DAG).
///
/// This function computes the longest path by weight in a given DAG. It will return the longest path
/// along with its total weight, or `None` if the graph contains cycles which make the longest path
/// computation undefined.
///
/// # Arguments
/// * `graph`: Reference to a directed graph.
/// * `weight_fn` - An input callable that will be passed the `EdgeRef` for each edge in the graph.
///  The callable should return the weight of the edge. The weight must be a type that implements
/// `Num`, `Zero`, `PartialOrd`, and `Copy`.
///
/// # Type Parameters
/// * `G`: Type of the graph. Must be a directed graph.`
/// * `E`: Type of the edge weight.
/// * `F`: Type of the weight function.
/// * `T`: The type of the edge weight. Must implement `Num`, `Zero`, `PartialOrd`, and `Copy`.
///
/// # Returns
/// * `None` if the graph contains a cycle.
/// * `Some((Vec<NodeIndex>, T))` representing the longest path as a sequence of nodes and its total weight.
/// ```
pub fn longest_path<G, F, T>(graph: G, mut weight_fn: F) -> Option<(Vec<NodeIndex>, T)>
where
    G: GraphProp<EdgeType = Directed>
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + IntoEdgesDirected
        + Visitable
        + GraphBase<NodeId = NodeIndex>,
    F: FnMut(G::EdgeRef) -> T,
    T: Num + Zero + PartialOrd + Copy,
{
    let mut path: Vec<NodeIndex> = Vec::new(); // This will store the longest path
    let nodes = match algo::toposort(graph, None) {
        // Topologically sort the nodes
        Ok(nodes) => nodes,
        Err(_) => return None, // Should not happen since we check for cycles above
    };

    // Handle the trivial case where the graph is empty
    if nodes.is_empty() {
        return Some((path, T::zero()));
    }

    let mut dist: HashMap<NodeIndex, (T, NodeIndex)> = HashMap::new(); // Distance map from node to (weight, prev_node)

    // Iterate over nodes in topological order
    for node in nodes {
        let parents = graph.edges_directed(node, petgraph::Direction::Incoming);
        let mut us: Vec<(T, NodeIndex)> = Vec::new();
        for p_edge in parents {
            let p_node = p_edge.source();
            let weight: T = weight_fn(p_edge);
            let length = dist[&p_node].0 + weight;
            us.push((length, p_node));
        }
        let maxu: (T, NodeIndex) = if !us.is_empty() {
            *us.iter()
                .max_by(|a, b| {
                    let weight_a = a.0;
                    let weight_b = b.0;
                    weight_a.partial_cmp(&weight_b).unwrap()
                })
                .unwrap()
        } else {
            (T::zero(), node)
        };
        dist.insert(node, maxu);
    }

    // Find the node that has the maximum distance
    let first = dist
        .keys()
        .max_by(|a, b| dist[*a].partial_cmp(&dist[*b]).unwrap())
        .unwrap();
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
    use petgraph::graph::DiGraph;
    use petgraph::stable_graph::StableDiGraph;

    #[test]
    fn test_empty_graph() {
        let graph: DiGraph<(), ()> = DiGraph::new();
        let weight_fn = |_: petgraph::graph::EdgeReference<()>| 0;
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Some((vec![], 0)));
    }

    #[test]
    fn test_single_node_graph() {
        let mut graph: DiGraph<(), ()> = DiGraph::new();
        let node = graph.add_node(());
        let weight_fn = |_: petgraph::graph::EdgeReference<()>| 0;
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Some((vec![node], 0)));
    }

    #[test]
    fn test_dag_with_multiple_paths() {
        let mut graph: DiGraph<(), i32> = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        graph.add_edge(n1, n2, 1);
        graph.add_edge(n1, n3, 2);
        graph.add_edge(n2, n3, 1);

        let weight_fn = |edge: petgraph::graph::EdgeReference<i32>| *edge.weight();
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Some((vec![n1, n3], 2)));
    }

    #[test]
    fn test_graph_with_cycle() {
        let mut graph: DiGraph<(), i32> = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n1, n2, 1);
        graph.add_edge(n2, n1, 1); // Creates a cycle

        let weight_fn = |edge: petgraph::graph::EdgeReference<i32>| *edge.weight();
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, None);
    }

    #[test]
    fn test_negative_weights() {
        let mut graph: DiGraph<(), i32> = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        graph.add_edge(n1, n2, -1);
        graph.add_edge(n1, n3, 2);
        graph.add_edge(n2, n3, -1);

        let weight_fn = |edge: petgraph::graph::EdgeReference<i32>| *edge.weight();
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Some((vec![n1, n3], 2)));
    }

    #[test]
    fn test_longest_path_in_stable_digraph() {
        let mut graph: StableDiGraph<(), i32> = StableDiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        graph.add_edge(n1, n2, 1);
        graph.add_edge(n1, n3, 2);
        graph.add_edge(n2, n3, 1);

        let weight_fn = |edge: petgraph::stable_graph::EdgeReference<'_, i32>| *edge.weight();
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Some((vec![n1, n3], 2)));
    }
}
