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

//! Module for graph random walk algorithms.

use petgraph::{Direction::Outgoing, visit::IntoNeighborsDirected};
use std::hash::Hash;

use hashbrown::HashMap;

use rand::prelude::*;
use rand_pcg::Pcg64;

/// Return a random path (or random walk) on the graph.
///
/// The next node to visit is selected uniformly at random from the outgoing
/// neighbors. If a node has no outgoing neighbor, the path will stop early.
/// The graph may be directed or not.
///
/// # Arguments:
///
/// * `graph` - Graph on which the random walk is done.
/// * `source` - Starting node of the path.
/// * `length` - Maximum length of the path.
/// * `seed` - seed of the random number generator that chooses the next node.
///
/// # Returns
///
/// A vector of the visited nodes including the initial node `source`.
///
/// # Example
///
/// ```rust
/// use petgraph::graph::DiGraph;
/// use rustworkx_core::traversal::generate_random_path;
///
/// let mut graph: DiGraph<(), ()> = DiGraph::with_capacity(3, 3);
/// let a = graph.add_node(());
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// graph.extend_with_edges([(a, b), (b, c), (c, a)]);
/// let path = generate_random_path(&graph, a, 3, Some(5));
/// assert_eq!(path, vec![a, b, c, a]);
/// ```
pub fn generate_random_path<G>(
    graph: G,
    source: G::NodeId,
    length: usize,
    seed: Option<u64>,
) -> Vec<G::NodeId>
where
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_os_rng(),
    };

    let mut degrees: HashMap<G::NodeId, usize> = HashMap::new();
    let mut get_degree_lazy = |u: G::NodeId| {
        *degrees
            .entry(u)
            .or_insert_with(|| graph.neighbors_directed(u, Outgoing).count())
    };

    let mut path = Vec::with_capacity(length + 1);
    let mut current_node = source;
    path.push(source);
    for _ in 0..length {
        let degree = get_degree_lazy(current_node);
        if degree == 0 {
            return path;
        }
        let idx = rng.random_range(..degree);
        let neighbor = graph
            .neighbors_directed(current_node, Outgoing)
            .nth(idx)
            .unwrap();
        path.push(neighbor);
        current_node = neighbor;
    }
    path
}

#[cfg(test)]
mod tests {
    use crate::traversal::generate_random_path;
    use hashbrown::HashMap;
    use petgraph::graph::{DiGraph, UnGraph};

    #[test]
    fn test_degree_zero_shorter_path() {
        let mut graph: DiGraph<(), ()> = DiGraph::with_capacity(2, 1);
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, ());

        // Node b has no neighbor and the random walk terminates early.
        assert_eq!(generate_random_path(&graph, a, 10, None), vec![a, b]);
    }

    #[test]
    fn test_alternating_path_digraph() {
        let mut graph: DiGraph<(), ()> = DiGraph::with_capacity(2, 1);
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, a, ());

        assert!(generate_random_path(&graph, a, 3, None) == vec![a, b, a, b]);
    }

    #[test]
    fn test_alternating_path_graph() {
        let mut graph: UnGraph<(), ()> = UnGraph::with_capacity(2, 1);
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, ());

        assert!(generate_random_path(&graph, a, 3, None) == vec![a, b, a, b]);
    }

    #[test]
    fn test_path_visit_frequency() {
        // a -- b -- c -- d
        //         / |
        //      e -- f -- g
        let mut graph: UnGraph<(), ()> = UnGraph::with_capacity(7, 8);

        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        let f = graph.add_node(());
        let g = graph.add_node(());

        graph.extend_with_edges([(a, b), (b, c), (c, d), (e, f), (c, e), (c, f), (f, g)]);

        let path_length = 5_000;
        let mut frequencies = generate_random_path(&graph, a, path_length, Some(5))
            .iter()
            .copied()
            .fold(HashMap::new(), |mut map, val| {
                map.entry(val).and_modify(|frq| *frq += 1_f64).or_insert(1.);
                map
            });
        for (_, k) in frequencies.iter_mut() {
            *k /= path_length as f64 + 1.;
        }

        // Expected frequency is degree/2 number of edges.
        let tol = 1e-2;
        assert!((frequencies.get(&a).unwrap() - 1. / 14.).abs() < tol);
        assert!((frequencies.get(&b).unwrap() - 2. / 14.).abs() < tol);
        assert!((frequencies.get(&c).unwrap() - 4. / 14.).abs() < tol);
        assert!((frequencies.get(&d).unwrap() - 1. / 14.).abs() < tol);
        assert!((frequencies.get(&e).unwrap() - 2. / 14.).abs() < tol);
        assert!((frequencies.get(&f).unwrap() - 3. / 14.).abs() < tol);
        assert!((frequencies.get(&g).unwrap() - 1. / 14.).abs() < tol);
    }
}
