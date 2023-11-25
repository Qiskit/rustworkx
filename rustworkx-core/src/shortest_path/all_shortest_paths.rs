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

// This module was originally copied and forked from the upstream petgraph
// repository, specifically:
// https://github.com/petgraph/petgraph/blob/0.5.1/src/dijkstra.rs
// this was necessary to modify the error handling to allow python callables
// to be use for the input functions for edge_cost and return any exceptions
// raised in Python instead of panicking

use std::collections::VecDeque;
use std::hash::Hash;

use petgraph::algo::Measure;
use petgraph::visit::{EdgeRef, IntoEdgesDirected, NodeIndexable, Visitable};
use petgraph::Direction::Incoming;

use super::dijkstra;
use crate::dictmap::*;

/// Dijkstra-based all shortest paths algorithm.
///
/// Compute every single shortest path from `start` to `goal`.
///
/// The graph should be [`Visitable`] and implement [`IntoEdgesDirected`]. The function
/// `edge_cost` should return the cost for a particular edge, which is used
/// to compute path costs. Edge costs must be non-negative.
///
///
/// Returns a [`Vec`] which contains all possible shortest paths. Each path
/// is a Vec of node indices of the path starting with `start` and ending `goal`.
/// # Example
/// ```rust
/// use rustworkx_core::petgraph::Graph;
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::dictmap::DictMap;
/// use rustworkx_core::shortest_path::all_shortest_paths;
/// use rustworkx_core::Result;
/// use ahash::HashSet;
///
/// let mut graph : Graph<(), (), Directed>= Graph::new();
/// let a = graph.add_node(()); // node with no weight
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// let d = graph.add_node(());
/// let e = graph.add_node(());
/// let f = graph.add_node(());
/// let g = graph.add_node(());
/// // z will be in another connected component
/// let z = graph.add_node(());
///
/// graph.extend_with_edges(&[
///     (a, b),
///     (a, c),
///     (b, d),
///     (b, f),
///     (c, d),
///     (d, e),
///     (f, e),
///     (e, g)    
/// ]);
/// // a ----> b ----> f
/// // |       |       |
/// // v       v       v
/// // c ----> d ----> e ----> g
///
/// let expected_res: Vec<Vec<NodeIndex>>= [
///      vec![a, b, d, e, g],
///      vec![a, c, d, e, g],
///      vec![a, b, f, e, g],
///     ].into_iter().collect();
/// let res: Result<Vec<Vec<NodeIndex>>> = all_shortest_paths(
///     &graph, a, g, |_| Ok(1)
/// );
/// assert_eq!(res.unwrap(), expected_res)
/// ```
pub fn all_shortest_paths<G, F, E, K>(
    graph: G,
    start: G::NodeId,
    goal: G::NodeId,
    mut edge_cost: F,
) -> Result<Vec<Vec<G::NodeId>>, E>
where
    G: IntoEdgesDirected + Visitable + NodeIndexable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
{
    let scores: DictMap<G::NodeId, K> = dijkstra(&graph, start, None, &mut edge_cost, None)?;
    if !scores.contains_key(&goal) {
        return Ok(vec![]);
    }
    let mut paths = vec![];
    let path = VecDeque::from([goal]);
    let mut queue = vec![(goal, path)];
    while let Some((curr, curr_path)) = queue.pop() {
        let curr_dist = *scores.get(&curr).unwrap();
        for edge in graph.edges_directed(curr, Incoming) {
            // Only simple paths
            if curr_path.contains(&edge.source()) {
                continue;
            }
            let next_dist = match scores.get(&edge.source()) {
                Some(x) => *x,
                None => continue,
            };
            if curr_dist == next_dist + edge_cost(edge)? {
                let mut new_path = curr_path.clone();
                new_path.push_front(edge.source());
                if edge.source() == start {
                    paths.push(new_path.into());
                    continue;
                }
                queue.push((edge.source(), new_path));
            }
        }
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use crate::shortest_path::all_shortest_paths;
    use crate::Result;
    use petgraph::prelude::*;
    use petgraph::Graph;

    #[test]
    fn test_all_shortest_paths() {
        let mut g = Graph::new_undirected();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        let d = g.add_node("D");
        let e = g.add_node("E");
        let f = g.add_node("F");
        g.add_edge(a, b, 7);
        g.add_edge(c, a, 9);
        g.add_edge(a, d, 11);
        g.add_edge(b, c, 10);
        g.add_edge(d, c, 2);
        g.add_edge(d, e, 9);
        g.add_edge(b, f, 15);
        g.add_edge(c, f, 11);
        g.add_edge(e, f, 6);

        let start = a;
        let goal = e;
        let paths: Result<Vec<Vec<NodeIndex>>> =
            all_shortest_paths(&g, start, goal, |e| Ok(*e.weight()));

        // a --> d --> e        (11 + 9)
        // a --> c --> d --> e  (9 + 2 + 9)
        let expected_paths: Vec<Vec<NodeIndex>> =
            [vec![a, d, e], vec![a, c, d, e]].into_iter().collect();
        assert_eq!(paths.unwrap(), expected_paths);
    }

    #[test]
    fn test_all_paths_no_path() {
        let mut g: Graph<&str, (), Undirected> = Graph::new_undirected();
        let a = g.add_node("A");
        let b = g.add_node("B");

        let start = a;
        let goal = b;
        let paths: Result<Vec<Vec<NodeIndex>>> = all_shortest_paths(&g, start, goal, |_| Ok(1));

        let expected_paths: Vec<Vec<NodeIndex>> = vec![];
        assert_eq!(paths.unwrap(), expected_paths);
    }

    #[test]
    fn test_all_paths_0_weight() {
        let mut g = Graph::new_undirected();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        let d = g.add_node("D");
        let e = g.add_node("E");
        let f = g.add_node("F");

        g.add_edge(a, b, 1);
        g.add_edge(b, f, 2);
        g.add_edge(a, c, 2);
        g.add_edge(c, d, 1);
        g.add_edge(d, e, 0);
        g.add_edge(e, f, 0);

        let start = a;
        let goal = f;

        let paths: Result<Vec<Vec<NodeIndex>>> =
            all_shortest_paths(&g, start, goal, |e| Ok(*e.weight()));

        assert_eq!(paths.unwrap().len(), 2);
    }

    #[test]
    fn test_all_paths_0_weight_cycles() {
        let mut g = Graph::new_undirected();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        let d = g.add_node("D");
        let e = g.add_node("E");
        let f = g.add_node("F");

        g.add_edge(a, b, 1);
        g.add_edge(b, c, 0);
        g.add_edge(c, f, 1);
        g.add_edge(b, d, 0);
        g.add_edge(d, e, 0);
        g.add_edge(e, c, 0);

        let start = a;
        let goal = f;

        let paths: Result<Vec<Vec<NodeIndex>>> =
            dbg!(all_shortest_paths(&g, start, goal, |e| Ok(*e.weight())));

        assert_eq!(paths.unwrap().len(), 2);
    }

    #[test]
    fn test_all_shortest_paths_nearly_fully_connected() {
        let mut g = Graph::new_undirected();
        let num_nodes = 100;
        let nodes: Vec<NodeIndex> = (0..num_nodes).map(|_| g.add_node(1)).collect();
        for n1 in nodes.iter() {
            for n2 in nodes.iter() {
                if n1 != n2 {
                    g.update_edge(*n1, *n2, 1);
                }
            }
        }
        let start = nodes[0];
        let goal = nodes[1];

        let paths: Result<Vec<Vec<NodeIndex>>> =
            all_shortest_paths(&g, start, goal, |e| Ok(*e.weight()));
        assert_eq!(paths.unwrap().len(), 1);

        let edge = g.edges_connecting(start, goal).next().unwrap();
        g.remove_edge(edge.id());

        let paths: Result<Vec<Vec<NodeIndex>>> =
            all_shortest_paths(&g, start, goal, |e| Ok(*e.weight()));

        assert_eq!(paths.unwrap().len(), num_nodes - 2);
    }
}
