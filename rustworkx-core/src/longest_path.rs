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

/// Generic type parameters allow this function to be flexible.
/// T should be a numeric type that can be zero-initialized and compared.
pub fn longest_path<N, E, F, T>(graph: &DiGraph<N, E>, mut weight_fn: F) -> Option<(Vec<NodeIndex>, T)>
where
    F: FnMut(NodeIndex, NodeIndex, &E) -> T,
    T: Ord + Copy + std::ops::Add<Output = T> + Default,
{
    if algo::is_cyclic_directed(graph) {
        return None;
    }

    let mut path: Vec<NodeIndex> = Vec::new();
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_) => return None, // This case will not occur since we check for cycles above.
    };

    if nodes.is_empty() {
        return Some((path, T::default()));
    }

    let mut dist: HashMap<NodeIndex, (T, NodeIndex)> = HashMap::new();
    for node in nodes {
        let parents = graph.edges_directed(node, petgraph::Direction::Incoming);
        let mut us: Vec<(T, NodeIndex)> = Vec::new();
        for edge in parents {
            let (p_node, target, weight) = (edge.source(), edge.target(), edge.weight());
            let weight = weight_fn(p_node, target, weight);
            let length = *dist.get(&p_node).map_or(&T::default(), |(d, _)| d) + weight;
            us.push((length, p_node));
        }
        if let Some(maxu) = us.iter().max_by_key(|x| x.0) {
            dist.insert(node, *maxu);
        } else {
            dist.insert(node, (T::default(), node));
        }
    }

    let first = dist.keys().max_by_key(|&n| dist[n].0).unwrap();
    let mut v = *first;
    let mut u: Option<NodeIndex> = None;
    while match u {
        Some(u) => u != v,
        None => true,
    } {
        path.push(v);
        u = Some(v);
        v = dist[&v].1;
    }
    path.reverse();
    let path_weight = dist[first].0;
    Some((path, path_weight))
}

#[cfg(test)]
mod test_longest_path {
    use super::*;
    use petgraph::graph::DiGraph;
    use petgraph::graph::NodeIndex;

    #[test]
    fn test_empty_graph() {
        let graph: DiGraph<(), ()> = DiGraph::new();
        let weight_fn = |_: NodeIndex, _: NodeIndex, _: &()| 0;
        assert_eq!(longest_path(&graph, weight_fn), Some((vec![], 0)));
    }

    #[test]
    fn test_single_node() {
        let mut graph = DiGraph::new();
        graph.add_node(());

        let weight_fn = |_: NodeIndex, _: NodeIndex, _: &()| 1;
        assert_eq!(longest_path(&graph, weight_fn), Some((vec![NodeIndex::new(0)], 0)));
    }

    #[test]
    fn test_simple_path() {
        let mut graph = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n1, n2, 1);

        let weight_fn = |_, _, &w: &i32| w;
        assert_eq!(longest_path(&graph, weight_fn), Some((vec![n1, n2], 1)));
    }

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