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

#![allow(clippy::float_cmp)]

use std::hash::Hash;

use petgraph::data::{Build, Create};
use petgraph::visit::{Data, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, NodeIndexable};

use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand_pcg::Pcg64;

use super::InvalidInputError;

/// Generate a G<sub>np</sub> random graph, also known as an
/// Erdős-Rényi graph or a binomial graph.
///
/// For number of nodes `n` and probability `p`, the G<sub>np</sub>
/// graph algorithm creates `n` nodes, and for all the `n * (n - 1)` possible edges,
/// each edge is created independently with probability `p`.
/// In general, for any probability `p`, the expected number of edges returned
/// is `m = p * n * (n - 1)`. If `p = 0` or `p = 1`, the returned
/// graph is not random and will always be an empty or a complete graph respectively.
/// An empty graph has zero edges and a complete directed graph has `n (n - 1)` edges.
/// The run time is `O(n + m)` where `m` is the expected number of edges mentioned above.
/// When `p = 0`, run time always reduces to `O(n)`, as the lower bound.
/// When `p = 1`, run time always goes to `O(n + n * (n - 1))`, as the upper bound.
///
/// For `0 < p < 1`, the algorithm is based on the implementation of the networkx function
/// ``fast_gnp_random_graph``,
/// <https://github.com/networkx/networkx/blob/networkx-2.4/networkx/generators/random_graphs.py#L49-L120>
///
/// Vladimir Batagelj and Ulrik Brandes,
///    "Efficient generation of large random networks",
///    Phys. Rev. E, 71, 036113, 2005.
///
/// Arguments:
///
/// * `num_nodes` - The number of nodes for creating the random graph.
/// * `probability` - The probability of creating an edge between two nodes as a float.
/// * `seed` - An optional seed to use for the random number generator.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::gnp_random_graph;
///
/// let g: petgraph::graph::DiGraph<(), ()> = gnp_random_graph(
///     20,
///     1.0,
///     None,
///     || {()},
///     || {()},
/// ).unwrap();
/// assert_eq!(g.node_count(), 20);
/// assert_eq!(g.edge_count(), 20 * (20 - 1));
/// ```
pub fn gnp_random_graph<G, T, F, H, M>(
    num_nodes: usize,
    probability: f64,
    seed: Option<u64>,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable + GraphProp,
    F: FnMut() -> T,
    H: FnMut() -> M,
    G::NodeId: Eq + Hash,
{
    if num_nodes == 0 {
        return Err(InvalidInputError {});
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut graph = G::with_capacity(num_nodes, num_nodes);
    let directed = graph.is_directed();

    for _ in 0..num_nodes {
        graph.add_node(default_node_weight());
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(InvalidInputError {});
    }
    if probability > 0.0 {
        if (probability - 1.0).abs() < std::f64::EPSILON {
            for u in 0..num_nodes {
                let start_node = if directed { 0 } else { u + 1 };
                for v in start_node..num_nodes {
                    if !directed || u != v {
                        // exclude self-loops
                        let u_index = graph.from_index(u);
                        let v_index = graph.from_index(v);
                        graph.add_edge(u_index, v_index, default_edge_weight());
                    }
                }
            }
        } else {
            let mut v: isize = if directed { 0 } else { 1 };
            let mut w: isize = -1;
            let num_nodes: isize = num_nodes as isize;
            let lp: f64 = (1.0 - probability).ln();

            let between = Uniform::new(0.0, 1.0);
            while v < num_nodes {
                let random: f64 = between.sample(&mut rng);
                let lr: f64 = (1.0 - random).ln();
                let ratio: isize = (lr / lp) as isize;
                w = w + 1 + ratio;

                if directed {
                    // avoid self loops
                    if v == w {
                        w += 1;
                    }
                }
                while v < num_nodes && ((directed && num_nodes <= w) || (!directed && v <= w)) {
                    w -= v;
                    v += 1;
                    // avoid self loops
                    if directed && v == w {
                        w -= v;
                        v += 1;
                    }
                }
                if v < num_nodes {
                    let v_index = graph.from_index(v as usize);
                    let w_index = graph.from_index(w as usize);
                    graph.add_edge(v_index, w_index, default_edge_weight());
                }
            }
        }
    }
    Ok(graph)
}

// /// Return a `G_{nm}` directed graph, also known as an
// /// Erdős-Rényi graph.
// ///
// /// Generates a random directed graph out of all the possible graphs with `n` nodes and
// /// `m` edges. The generated graph will not be a multigraph and will not have self loops.
// ///
// /// For `n` nodes, the maximum edges that can be returned is `n (n - 1)`.
// /// Passing `m` higher than that will still return the maximum number of edges.
// /// If `m = 0`, the returned graph will always be empty (no edges).
// /// When a seed is provided, the results are reproducible. Passing a seed when `m = 0`
// /// or `m >= n (n - 1)` has no effect, as the result will always be an empty or a complete graph respectively.
// ///
// /// This algorithm has a time complexity of `O(n + m)`

/// Generate a G<sub>nm</sub> random graph, also known as an
/// Erdős-Rényi graph.
///
/// Generates a random directed graph out of all the possible graphs with `n` nodes and
/// `m` edges. The generated graph will not be a multigraph and will not have self loops.
///
/// For `n` nodes, the maximum edges that can be returned is `n * (n - 1)`.
/// Passing `m` higher than that will still return the maximum number of edges.
/// If `m = 0`, the returned graph will always be empty (no edges).
/// When a seed is provided, the results are reproducible. Passing a seed when `m = 0`
/// or `m >= n * (n - 1)` has no effect, as the result will always be an empty or a
/// complete graph respectively.
///
/// This algorithm has a time complexity of `O(n + m)`
///
/// Arguments:
///
/// * `num_nodes` - The number of nodes to create in the graph.
/// * `num_edges` - The number of edges to create in the graph.
/// * `seed` - An optional seed to use for the random number generator.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::gnm_random_graph;
///
/// let g: petgraph::graph::DiGraph<(), ()> = gnm_random_graph(
///     20,
///     12,
///     None,
///     || {()},
///     || {()},
/// ).unwrap();
/// assert_eq!(g.node_count(), 20);
/// assert_eq!(g.edge_count(), 12);
/// ```
pub fn gnm_random_graph<G, T, F, H, M>(
    num_nodes: usize,
    num_edges: usize,
    seed: Option<u64>,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: GraphProp + Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut() -> T,
    H: FnMut() -> M,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoEdgeReferences,
    G::NodeId: Eq + Hash,
{
    if num_nodes == 0 {
        return Err(InvalidInputError {});
    }

    fn find_edge<G>(graph: &G, source: usize, target: usize) -> bool
    where
        G: GraphBase + NodeIndexable,
        for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoEdgeReferences,
    {
        let mut found = false;
        for edge in graph.edge_references() {
            if graph.to_index(edge.source()) == source && graph.to_index(edge.target()) == target {
                found = true;
                break;
            }
        }
        found
    }

    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut graph = G::with_capacity(num_nodes, num_edges);
    let directed = graph.is_directed();

    for _ in 0..num_nodes {
        graph.add_node(default_node_weight());
    }
    // if number of edges to be created is >= max,
    // avoid randomly missed trials and directly add edges between every node
    let div_by = if directed { 1 } else { 2 };
    if num_edges >= num_nodes * (num_nodes - 1) / div_by {
        for u in 0..num_nodes {
            let start_node = if directed { 0 } else { u + 1 };
            for v in start_node..num_nodes {
                // avoid self-loops
                if !directed || u != v {
                    let u_index = graph.from_index(u);
                    let v_index = graph.from_index(v);
                    graph.add_edge(u_index, v_index, default_edge_weight());
                }
            }
        }
    } else {
        let mut created_edges: usize = 0;
        let between = Uniform::new(0, num_nodes);
        while created_edges < num_edges {
            let u = between.sample(&mut rng);
            let v = between.sample(&mut rng);
            let u_index = graph.from_index(u);
            let v_index = graph.from_index(v);
            // avoid self-loops and multi-graphs
            if u != v && !find_edge(&graph, u, v) {
                graph.add_edge(u_index, v_index, default_edge_weight());
                created_edges += 1;
            }
        }
    }
    Ok(graph)
}

#[inline]
fn pnorm(x: f64, p: f64) -> f64 {
    if p == 1.0 || p == std::f64::INFINITY {
        x.abs()
    } else if p == 2.0 {
        x * x
    } else {
        x.abs().powf(p)
    }
}

fn distance(x: &[f64], y: &[f64], p: f64) -> f64 {
    let it = x.iter().zip(y.iter()).map(|(xi, yi)| pnorm(xi - yi, p));

    if p == std::f64::INFINITY {
        it.fold(-1.0, |max, x| if x > max { x } else { max })
    } else {
        it.sum()
    }
}

/// Generate a random geometric graph in the unit cube of dimensions `dim`.
///
/// The random geometric graph model places `num_nodes` nodes uniformly at
/// random in the unit cube. Two nodes are joined by an edge if the
/// distance between the nodes is at most `radius`.
///
/// Each node has a node attribute ``'pos'`` that stores the
/// position of that node in Euclidean space as provided by the
/// ``pos`` keyword argument or, if ``pos`` was not provided, as
/// generated by this function.
///
/// Arguments
///
/// * `num_nodes` - The number of nodes to create in the graph.
/// * `radius` - Distance threshold value.
/// * `dim` - Dimension of node positions. Default: 2
/// * `pos` - Optional list with node positions as values.
/// * `p` - Which Minkowski distance metric to use.  `p` has to meet the condition
///     ``1 <= p <= infinity``.
///     If this argument is not specified, the L<sup>2</sup> metric
///     (the Euclidean distance metric), `p = 2` is used.
/// * `seed` - An optional seed to use for the random number generator.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::random_geometric_graph;
///
/// let g: petgraph::graph::UnGraph<Vec<f64>, ()> = random_geometric_graph(
///     10,
///     1.42,
///     2,
///     None,
///     2.0,
///     None,
///     || {()},
/// ).unwrap();
/// assert_eq!(g.node_count(), 10);
/// assert_eq!(g.edge_count(), 45);
/// ```
pub fn random_geometric_graph<G, H, M>(
    num_nodes: usize,
    radius: f64,
    dim: usize,
    pos: Option<Vec<Vec<f64>>>,
    p: f64,
    seed: Option<u64>,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: GraphBase + Build + Create + Data<NodeWeight = Vec<f64>, EdgeWeight = M> + NodeIndexable,
    H: FnMut() -> M,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoEdgeReferences,
    G::NodeId: Eq + Hash,
{
    if num_nodes == 0 {
        return Err(InvalidInputError {});
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut graph = G::with_capacity(num_nodes, num_nodes);

    let radius_p = pnorm(radius, p);
    let dist = Uniform::new(0.0, 1.0);
    let pos = pos.unwrap_or_else(|| {
        (0..num_nodes)
            .map(|_| (0..dim).map(|_| dist.sample(&mut rng)).collect())
            .collect()
    });
    if num_nodes != pos.len() {
        return Err(InvalidInputError {});
    }
    for pval in pos.iter() {
        graph.add_node(pval.clone());
    }
    for u in 0..(num_nodes - 1) {
        for v in (u + 1)..num_nodes {
            if distance(&pos[u], &pos[v], p) < radius_p {
                graph.add_edge(
                    graph.from_index(u),
                    graph.from_index(v),
                    default_edge_weight(),
                );
            }
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::InvalidInputError;
    use crate::generators::{gnm_random_graph, gnp_random_graph, random_geometric_graph};
    use crate::petgraph;

    // Test gnp_random_graph

    #[test]
    fn test_gnp_random_graph_directed() {
        let g: petgraph::graph::DiGraph<(), ()> =
            gnp_random_graph(20, 0.5, Some(10), || (), || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 104);
    }

    #[test]
    fn test_gnp_random_graph_directed_empty() {
        let g: petgraph::graph::DiGraph<(), ()> =
            gnp_random_graph(20, 0.0, None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_gnp_random_graph_directed_complete() {
        let g: petgraph::graph::DiGraph<(), ()> =
            gnp_random_graph(20, 1.0, None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 20 * (20 - 1));
    }

    #[test]
    fn test_gnp_random_graph_undirected() {
        let g: petgraph::graph::UnGraph<(), ()> =
            gnp_random_graph(20, 0.5, Some(10), || (), || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 105);
    }

    #[test]
    fn test_gnp_random_graph_undirected_empty() {
        let g: petgraph::graph::UnGraph<(), ()> =
            gnp_random_graph(20, 0.0, None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_gnp_random_graph_undirected_complete() {
        let g: petgraph::graph::UnGraph<(), ()> =
            gnp_random_graph(20, 1.0, None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 20 * (20 - 1) / 2);
    }

    #[test]
    fn test_gnp_random_graph_error() {
        match gnp_random_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
            0,
            3.0,
            None,
            || (),
            || (),
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }

    // Test gnm_random_graph

    #[test]
    fn test_gnm_random_graph_directed() {
        let g: petgraph::graph::DiGraph<(), ()> =
            gnm_random_graph(20, 100, None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 100);
    }

    #[test]
    fn test_gnm_random_graph_directed_empty() {
        let g: petgraph::graph::DiGraph<(), ()> =
            gnm_random_graph(20, 0, None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_gnm_random_graph_directed_complete() {
        let g: petgraph::graph::DiGraph<(), ()> =
            gnm_random_graph(20, 20 * (20 - 1), None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 20 * (20 - 1));
    }

    #[test]
    fn test_gnm_random_graph_directed_max_edges() {
        let n = 20;
        let max_m = n * (n - 1);
        let g: petgraph::graph::DiGraph<(), ()> =
            gnm_random_graph(n, max_m, None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), n);
        assert_eq!(g.edge_count(), max_m);
        // passing the max edges for the passed number of nodes
        let g: petgraph::graph::DiGraph<(), ()> =
            gnm_random_graph(n, max_m + 1, None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), n);
        assert_eq!(g.edge_count(), max_m);
        // passing a seed when passing max edges has no effect
        let g: petgraph::graph::DiGraph<(), ()> =
            gnm_random_graph(n, max_m, Some(55), || (), || ()).unwrap();
        assert_eq!(g.node_count(), n);
        assert_eq!(g.edge_count(), max_m);
    }

    #[test]
    fn test_gnm_random_graph_error() {
        match gnm_random_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
            0,
            0,
            None,
            || (),
            || (),
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }

    // Test random_geometric_graph

    #[test]
    fn test_random_geometric_empty() {
        let g: petgraph::graph::UnGraph<Vec<f64>, ()> =
            random_geometric_graph(20, 0.0, 2, None, 2.0, None, || ()).unwrap();
        assert_eq!(g.node_count(), 20);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_random_geometric_complete() {
        let g: petgraph::graph::UnGraph<Vec<f64>, ()> =
            random_geometric_graph(10, 1.42, 2, None, 2.0, None, || ()).unwrap();
        assert_eq!(g.node_count(), 10);
        assert_eq!(g.edge_count(), 45);
    }

    #[test]
    fn test_random_geometric_bad_num_nodes() {
        match random_geometric_graph::<petgraph::graph::UnGraph<Vec<f64>, ()>, _, ()>(
            0,
            1.0,
            2,
            None,
            2.0,
            None,
            || (),
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }

    #[test]
    fn test_random_geometric_bad_pos() {
        match random_geometric_graph::<petgraph::graph::UnGraph<Vec<f64>, ()>, _, ()>(
            3,
            0.15,
            3,
            Some(vec![vec![0.5, 0.5]]),
            2.0,
            None,
            || (),
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }
}
