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

use hashbrown::HashSet;

use petgraph::visit::{IntoEdges, IntoNeighborsDirected, IntoNodeIdentifiers, NodeIndexable};
use rayon::prelude::*;

use std::hash::Hash;

/// Counts triangles and connected triples containing `node` of the given undirected graph.
fn graph_node_triangles<G>(graph: G, node: G::NodeId) -> (usize, usize)
where
    G: IntoEdges,
    G::NodeId: Hash + Eq,
{
    let node_neighbors: HashSet<G::NodeId> = graph.neighbors(node).filter(|v| *v != node).collect();

    let triangles: usize = node_neighbors
        .iter()
        .map(|&v| {
            graph
                .neighbors(v)
                .filter(|&x| (x != v) && node_neighbors.contains(&x))
                .count()
        })
        .sum();

    let triples = match node_neighbors.len() {
        0 => 0,
        d => (d * (d - 1)) / 2,
    };

    (triangles / 2, triples)
}

/// Compute the transitivity of an undirected graph.
///
/// The transitivity of a graph is 3*number of triangles / number of connected triples, where
/// “connected triple” means a single vertex with edges running to an unordered pair of others.
///
/// This function is multithreaded and will launch a thread pool with threads equal to the number
/// of CPUs by default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
/// to 4 threads.
///
/// The function implicitly assumes that there are no parallel edges or self loops. It may produce
/// incorrect/unexpected results if the input graph has self loops or parallel edges.
pub fn graph_transitivity<G>(graph: G) -> f64
where
    G: NodeIndexable + IntoEdges + IntoNodeIdentifiers + Send + Sync,
    G::NodeId: Hash + Eq + Send + Sync,
{
    let nodes: Vec<_> = graph.node_identifiers().collect();
    let (triangles, triples) = nodes
        .par_iter()
        .map(|node| graph_node_triangles(graph, *node))
        .reduce(
            || (0, 0),
            |(sumx, sumy), (resx, resy)| (sumx + resx, sumy + resy),
        );

    match triangles {
        0 => 0.0,
        _ => triangles as f64 / triples as f64,
    }
}

/// Counts triangles and triples containing `node` of the given directed graph.
fn digraph_node_triangles<G>(graph: G, node: G::NodeId) -> (usize, usize)
where
    G: IntoNeighborsDirected,
    G::NodeId: Hash + Eq,
{
    let out_neighbors: HashSet<G::NodeId> = graph
        .neighbors_directed(node, petgraph::Direction::Outgoing)
        .filter(|v| *v != node)
        .collect();

    let in_neighbors: HashSet<G::NodeId> = graph
        .neighbors_directed(node, petgraph::Direction::Incoming)
        .filter(|v| *v != node)
        .collect();

    let triangles: usize = out_neighbors
        .iter()
        .chain(in_neighbors.iter())
        .map(|v| {
            graph
                .neighbors_directed(*v, petgraph::Direction::Outgoing)
                .chain(graph.neighbors_directed(*v, petgraph::Direction::Incoming))
                .map(|x| {
                    ((x != *v) && out_neighbors.contains(&x)) as usize
                        + ((x != *v) && in_neighbors.contains(&x)) as usize
                })
                .sum::<usize>()
        })
        .sum();

    let d_tot = in_neighbors.len() + out_neighbors.len();
    let d_bil = out_neighbors.intersection(&in_neighbors).count();
    let triples = match d_tot {
        0 => 0,
        _ => d_tot * (d_tot - 1) - 2 * d_bil,
    };

    (triangles / 2, triples)
}

/// Compute the transitivity of a directed graph.
///
/// The transitivity of a directed graph is 3*number of triangles/number of all possible triangles.
/// A triangle is a connected triple of nodes. Different edge orientations counts as different
/// triangles.
///
/// This function is multithreaded and will launch a thread pool with threads equal to the number
/// of CPUs by default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
/// to 4 threads.
///
/// The function implicitly assumes that there are no parallel edges or self loops. It may produce
/// incorrect/unexpected results if the input graph has self loops or parallel edges.
pub fn digraph_transitivity<G>(graph: G) -> f64
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoNeighborsDirected + Send + Sync,
    G::NodeId: Hash + Eq + Send + Sync,
{
    let nodes: Vec<_> = graph.node_identifiers().collect();
    let (triangles, triples) = nodes
        .par_iter()
        .map(|node| digraph_node_triangles(graph, *node))
        .reduce(
            || (0, 0),
            |(sumx, sumy), (resx, resy)| (sumx + resx, sumy + resy),
        );

    match triangles {
        0 => 0.0,
        _ => triangles as f64 / triples as f64,
    }
}

#[cfg(test)]
mod test_transitivity {
    use petgraph::{
        Graph,
        graph::{DiGraph, UnGraph},
    };

    use super::{
        digraph_node_triangles, digraph_transitivity, graph_node_triangles, graph_transitivity,
    };

    #[test]
    fn test_node_triangles() {
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(5, 6);
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        graph.extend_with_edges([(a, b), (b, c), (a, c), (a, d), (c, d), (d, e)]);
        assert_eq!(graph_node_triangles(&graph, a), (2, 3));
    }

    #[test]
    fn test_node_triangles_disconnected() {
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(1, 0);
        let a = graph.add_node(());
        assert_eq!(graph_node_triangles(&graph, a), (0, 0));
    }

    #[test]
    fn test_transitivity() {
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(5, 5);
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        graph.extend_with_edges([(a, b), (a, c), (a, d), (a, e), (b, c)]);

        assert_eq!(graph_transitivity(&graph), 3. / 8.);
    }

    #[test]
    fn test_transitivity_triangle() {
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(3, 3);
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.extend_with_edges([(a, b), (a, c), (b, c)]);
        assert_eq!(graph_transitivity(&graph), 1.0)
    }

    #[test]
    fn test_transitivity_star() {
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(5, 4);
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        graph.extend_with_edges([(a, b), (a, c), (a, d), (a, e)]);
        assert_eq!(graph_transitivity(&graph), 0.0)
    }

    #[test]
    fn test_transitivity_empty() {
        let graph: UnGraph<(), ()> = Graph::with_capacity(0, 0);
        assert_eq!(graph_transitivity(&graph), 0.0)
    }

    #[test]
    fn test_transitivity_disconnected() {
        let mut graph: UnGraph<(), ()> = Graph::with_capacity(2, 1);
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_node(());
        graph.add_edge(a, b, ());
        assert_eq!(graph_transitivity(&graph), 0.0)
    }

    #[test]
    fn test_two_directed_node_triangles() {
        let mut graph: DiGraph<(), ()> = Graph::with_capacity(5, 7);
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        // The reciprocal edge (a, c) (c, a) double the number of triangles
        graph.extend_with_edges([(a, b), (b, c), (c, a), (a, c), (c, d), (d, a), (d, e)]);
        assert_eq!(digraph_node_triangles(&graph, a), (4, 10));
    }

    #[test]
    fn test_directed_node_triangles_disconnected() {
        let mut graph: DiGraph<(), ()> = Graph::with_capacity(1, 0);
        let a = graph.add_node(());
        assert_eq!(graph_node_triangles(&graph, a), (0, 0));
    }

    #[test]
    fn test_transitivity_directed() {
        let mut graph: DiGraph<(), ()> = Graph::with_capacity(5, 4);
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        graph.add_node(());
        graph.extend_with_edges([(a, b), (a, c), (a, d), (b, c)]);
        assert_eq!(digraph_transitivity(&graph), 3. / 10.);
    }

    #[test]
    fn test_transitivity_triangle_directed() {
        let mut graph: DiGraph<(), ()> = Graph::with_capacity(3, 3);
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.extend_with_edges([(a, b), (a, c), (b, c)]);
        assert_eq!(digraph_transitivity(&graph), 0.5);
    }

    #[test]
    fn test_transitivity_fulltriangle_directed() {
        let mut graph: DiGraph<(), ()> = Graph::with_capacity(3, 6);
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.extend_with_edges([(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)]);
        assert_eq!(digraph_transitivity(&graph), 1.0);
    }

    #[test]
    fn test_transitivity_empty_directed() {
        let graph: DiGraph<(), ()> = Graph::with_capacity(0, 0);
        assert_eq!(digraph_transitivity(&graph), 0.0);
    }
}
