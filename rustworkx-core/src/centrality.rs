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

use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::RwLock;

use hashbrown::HashSet;

use hashbrown::HashMap;
use petgraph::algo::dijkstra;
use petgraph::visit::{
    Bfs, EdgeCount, EdgeIndexable, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, IntoEdges,
    IntoEdgesDirected, IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount,
    NodeIndexable, Reversed, ReversedEdgeReference, Visitable,
};
use rayon_cond::CondIterator;

/// Compute the betweenness centrality of all nodes in a graph.
///
/// The algorithm used in this function is based on:
///
/// Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
/// Journal of Mathematical Sociology 25(2):163-177, 2001.
///
/// This function is multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold``. If the
/// function will be running in parallel the env var ``RAYON_NUM_THREADS`` can
/// be used to adjust how many threads will be used.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `include_endpoints` - Whether to include the endpoints of paths in the path
///   lengths used to compute the betweenness
/// * `normalized` - Whether to normalize the betweenness scores by the number
///   of distinct paths between all pairs of nodes
/// * `parallel_threshold` - The number of nodes to calculate the betweenness
///   centrality in parallel at, if the number of nodes in `graph` is less
///   than this value it will run in a single thread. A good default to use
///   here if you're not sure is `50` as that was found to be roughly the
///   number of nodes where parallelism improves performance
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::centrality::betweenness_centrality;
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 4), (1, 2), (2, 3), (3, 4), (1, 4)
/// ]);
/// // Calculate the betweenness centrality
/// let output = betweenness_centrality(&g, true, true, 200);
/// assert_eq!(
///     vec![Some(0.4), Some(0.5), Some(0.45), Some(0.5), Some(0.75)],
///     output
/// );
/// ```
/// # See Also
/// [`edge_betweenness_centrality`]
pub fn betweenness_centrality<G>(
    graph: G,
    include_endpoints: bool,
    normalized: bool,
    parallel_threshold: usize,
) -> Vec<Option<f64>>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + NodeCount
        + GraphProp
        + GraphBase
        + std::marker::Sync,
    <G as GraphBase>::NodeId: std::cmp::Eq + Send,
    // rustfmt deletes the following comments if placed inline above
    // + IntoNodeIdentifiers // for node_identifiers()
    // + IntoNeighborsDirected // for neighbors()
    // + NodeCount // for node_count
    // + GraphProp // for is_directed
{
    // Correspondence of variable names to quantities in the paper is as follows:
    //
    // P -- predecessors
    // S -- verts_sorted_by_distance,
    //      vertices in order of non-decreasing distance from s
    // Q -- Q
    // sigma -- sigma
    // delta -- delta
    // d -- distance
    let max_index = graph.node_bound();

    let mut betweenness: Vec<Option<f64>> = vec![None; max_index];
    for node_s in graph.node_identifiers() {
        let is: usize = graph.to_index(node_s);
        betweenness[is] = Some(0.0);
    }
    let locked_betweenness = RwLock::new(&mut betweenness);
    let node_indices: Vec<G::NodeId> = graph.node_identifiers().collect();

    CondIterator::new(node_indices, graph.node_count() >= parallel_threshold)
        .map(|node_s| (shortest_path_for_centrality(&graph, &node_s), node_s))
        .for_each(|(mut shortest_path_calc, node_s)| {
            _accumulate_vertices(
                &locked_betweenness,
                max_index,
                &mut shortest_path_calc,
                node_s,
                &graph,
                include_endpoints,
            );
        });

    _rescale(
        &mut betweenness,
        graph.node_count(),
        normalized,
        graph.is_directed(),
        include_endpoints,
    );

    betweenness
}

/// Compute the edge betweenness centrality of all edges in a graph.
///
/// The algorithm used in this function is based on:
///
/// Ulrik Brandes: On Variants of Shortest-Path Betweenness
/// Centrality and their Generic Computation.
/// Social Networks 30(2):136-145, 2008.
/// <https://doi.org/10.1016/j.socnet.2007.11.001>.
///
/// This function is multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold``. If the
/// function will be running in parallel the env var ``RAYON_NUM_THREADS`` can
/// be used to adjust how many threads will be used.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `normalized` - Whether to normalize the betweenness scores by the number
///   of distinct paths between all pairs of nodes
/// * `parallel_threshold` - The number of nodes to calculate the betweenness
///   centrality in parallel at, if the number of nodes in `graph` is less
///   than this value it will run in a single thread. A good default to use
///   here if you're not sure is `50` as that was found to be roughly the
///   number of nodes where parallelism improves performance
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::centrality::edge_betweenness_centrality;
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 4), (1, 2), (1, 3), (2, 3), (3, 4), (1, 4)
/// ]);
///
/// let output = edge_betweenness_centrality(&g, false, 200);
/// let expected = vec![Some(4.0), Some(2.0), Some(1.0), Some(2.0), Some(3.0), Some(3.0)];
/// assert_eq!(output, expected);
/// ```
/// # See Also
/// [`betweenness_centrality`]
pub fn edge_betweenness_centrality<G>(
    graph: G,
    normalized: bool,
    parallel_threshold: usize,
) -> Vec<Option<f64>>
where
    G: NodeIndexable
        + EdgeIndexable
        + IntoEdges
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + NodeCount
        + EdgeCount
        + GraphProp
        + Sync,
    G::NodeId: Eq + Send,
    G::EdgeId: Eq + Send,
{
    let max_index = graph.node_bound();
    let mut betweenness = vec![None; graph.edge_bound()];
    for edge in graph.edge_references() {
        let is: usize = EdgeIndexable::to_index(&graph, edge.id());
        betweenness[is] = Some(0.0);
    }
    let locked_betweenness = RwLock::new(&mut betweenness);
    let node_indices: Vec<G::NodeId> = graph.node_identifiers().collect();
    CondIterator::new(node_indices, graph.node_count() >= parallel_threshold)
        .map(|node_s| shortest_path_for_edge_centrality(&graph, &node_s))
        .for_each(|mut shortest_path_calc| {
            accumulate_edges(
                &locked_betweenness,
                max_index,
                &mut shortest_path_calc,
                &graph,
            );
        });

    _rescale(
        &mut betweenness,
        graph.node_count(),
        normalized,
        graph.is_directed(),
        true,
    );
    betweenness
}

fn _rescale(
    betweenness: &mut [Option<f64>],
    node_count: usize,
    normalized: bool,
    directed: bool,
    include_endpoints: bool,
) {
    let mut do_scale = true;
    let mut scale = 1.0;
    if normalized {
        if include_endpoints {
            if node_count < 2 {
                do_scale = false;
            } else {
                scale = 1.0 / (node_count * (node_count - 1)) as f64;
            }
        } else if node_count <= 2 {
            do_scale = false;
        } else {
            scale = 1.0 / ((node_count - 1) * (node_count - 2)) as f64;
        }
    } else if !directed {
        scale = 0.5;
    } else {
        do_scale = false;
    }
    if do_scale {
        for x in betweenness.iter_mut() {
            *x = x.map(|y| y * scale);
        }
    }
}

fn _accumulate_vertices<G>(
    locked_betweenness: &RwLock<&mut Vec<Option<f64>>>,
    max_index: usize,
    path_calc: &mut ShortestPathData<G>,
    node_s: <G as GraphBase>::NodeId,
    graph: G,
    include_endpoints: bool,
) where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + NodeCount
        + GraphProp
        + GraphBase
        + std::marker::Sync,
    <G as GraphBase>::NodeId: std::cmp::Eq,
{
    let mut delta = vec![0.0; max_index];
    for w in &path_calc.verts_sorted_by_distance {
        let iw = graph.to_index(*w);
        let coeff = (1.0 + delta[iw]) / path_calc.sigma[iw];
        let p_w = path_calc.predecessors.get(iw).unwrap();
        for iv in p_w {
            delta[*iv] += path_calc.sigma[*iv] * coeff;
        }
    }
    let mut betweenness = locked_betweenness.write().unwrap();
    if include_endpoints {
        let i_node_s = graph.to_index(node_s);
        betweenness[i_node_s] = betweenness[i_node_s]
            .map(|x| x + ((path_calc.verts_sorted_by_distance.len() - 1) as f64));
        for w in &path_calc.verts_sorted_by_distance {
            if *w != node_s {
                let iw = graph.to_index(*w);
                betweenness[iw] = betweenness[iw].map(|x| x + delta[iw] + 1.0);
            }
        }
    } else {
        for w in &path_calc.verts_sorted_by_distance {
            if *w != node_s {
                let iw = graph.to_index(*w);
                betweenness[iw] = betweenness[iw].map(|x| x + delta[iw]);
            }
        }
    }
}

fn accumulate_edges<G>(
    locked_betweenness: &RwLock<&mut Vec<Option<f64>>>,
    max_index: usize,
    path_calc: &mut ShortestPathDataWithEdges<G>,
    graph: G,
) where
    G: NodeIndexable + EdgeIndexable + Sync,
    G::NodeId: Eq,
    G::EdgeId: Eq,
{
    let mut delta = vec![0.0; max_index];
    for w in &path_calc.verts_sorted_by_distance {
        let iw = NodeIndexable::to_index(&graph, *w);
        let coeff = (1.0 + delta[iw]) / path_calc.sigma[iw];
        let p_w = path_calc.predecessors.get(iw).unwrap();
        let e_w = path_calc.predecessor_edges.get(iw).unwrap();
        let mut betweenness = locked_betweenness.write().unwrap();
        for i in 0..p_w.len() {
            let v = p_w[i];
            let iv = NodeIndexable::to_index(&graph, v);
            let ie = EdgeIndexable::to_index(&graph, e_w[i]);
            let c = path_calc.sigma[iv] * coeff;
            betweenness[ie] = betweenness[ie].map(|x| x + c);
            delta[iv] += c;
        }
    }
}
/// Compute the degree centrality of all nodes in a graph.
///
/// For undirected graphs, this calculates the normalized degree for each node.
/// For directed graphs, this calculates the normalized out-degree for each node.
///
/// Arguments:
///
/// * `graph` - The graph object to calculate degree centrality for
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph::graph::{UnGraph, DiGraph};
/// use rustworkx_core::centrality::degree_centrality;
///
/// // Undirected graph example
/// let graph = UnGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2), (2, 3), (3, 0)
/// ]);
/// let centrality = degree_centrality(&graph, None);
///
/// // Directed graph example
/// let digraph = DiGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)
/// ]);
/// let centrality = degree_centrality(&digraph, None);
/// ```
pub fn degree_centrality<G>(graph: G, direction: Option<petgraph::Direction>) -> Vec<f64>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighbors
        + IntoNeighborsDirected
        + NodeCount
        + GraphProp,
    G::NodeId: Eq,
{
    let node_count = graph.node_count() as f64;
    let mut centrality = vec![0.0; graph.node_bound()];

    for node in graph.node_identifiers() {
        let (degree, normalization) = match (graph.is_directed(), direction) {
            (true, None) => {
                let out_degree = graph
                    .neighbors_directed(node, petgraph::Direction::Outgoing)
                    .count() as f64;
                let in_degree = graph
                    .neighbors_directed(node, petgraph::Direction::Incoming)
                    .count() as f64;
                let total = in_degree + out_degree;
                // Use 2(n-1) normalization only if this is a complete graph
                let norm = if total == 2.0 * (node_count - 1.0) {
                    2.0 * (node_count - 1.0)
                } else {
                    node_count - 1.0
                };
                (total, norm)
            }
            (true, Some(dir)) => (
                graph.neighbors_directed(node, dir).count() as f64,
                node_count - 1.0,
            ),
            (false, _) => (graph.neighbors(node).count() as f64, node_count - 1.0),
        };
        centrality[graph.to_index(node)] = degree / normalization;
    }

    centrality
}

struct ShortestPathData<G>
where
    G: GraphBase,
    <G as GraphBase>::NodeId: std::cmp::Eq,
{
    verts_sorted_by_distance: Vec<G::NodeId>,
    predecessors: Vec<Vec<usize>>,
    sigma: Vec<f64>,
}
fn shortest_path_for_centrality<G>(graph: G, node_s: &G::NodeId) -> ShortestPathData<G>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoNeighborsDirected + NodeCount + GraphBase,
    <G as GraphBase>::NodeId: std::cmp::Eq,
{
    let c = graph.node_count();
    let max_index = graph.node_bound();
    let mut verts_sorted_by_distance: Vec<G::NodeId> = Vec::with_capacity(c); // a stack
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); max_index];
    let mut sigma: Vec<f64> = vec![0.; max_index];
    let mut distance: Vec<Option<usize>> = vec![None; max_index];
    #[allow(non_snake_case)]
    let mut Q: VecDeque<G::NodeId> = VecDeque::with_capacity(c);
    let node_s_index = graph.to_index(*node_s);
    sigma[node_s_index] = 1.0;
    distance[node_s_index] = Some(0);
    Q.push_back(*node_s);
    while let Some(v) = Q.pop_front() {
        verts_sorted_by_distance.push(v);
        let v_idx = graph.to_index(v);
        let distance_v = distance[v_idx].unwrap();
        for w in graph.neighbors(v) {
            let w_idx = graph.to_index(w);
            if distance[w_idx].is_none() {
                Q.push_back(w);
                distance[w_idx] = Some(distance_v + 1);
            }
            if distance[w_idx] == Some(distance_v + 1) {
                sigma[w_idx] += sigma[v_idx];
                predecessors[w_idx].push(v_idx);
            }
        }
    }
    verts_sorted_by_distance.reverse(); // will be effectively popping from the stack
    ShortestPathData {
        verts_sorted_by_distance,
        predecessors,
        sigma,
    }
}

struct ShortestPathDataWithEdges<G>
where
    G: GraphBase,
    G::NodeId: Eq,
    G::EdgeId: Eq,
{
    verts_sorted_by_distance: Vec<G::NodeId>,
    predecessors: Vec<Vec<G::NodeId>>,
    predecessor_edges: Vec<Vec<G::EdgeId>>,
    sigma: Vec<f64>,
}

fn shortest_path_for_edge_centrality<G>(
    graph: G,
    node_s: &G::NodeId,
) -> ShortestPathDataWithEdges<G>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + NodeCount
        + GraphBase
        + IntoEdges,
    G::NodeId: Eq,
    G::EdgeId: Eq,
{
    let mut verts_sorted_by_distance: Vec<G::NodeId> = Vec::new(); // a stack
    let c = graph.node_bound();
    let mut predecessors = vec![Vec::new(); c];
    let mut predecessor_edges = vec![Vec::new(); c];
    let mut sigma = vec![0.0; c];
    let mut distance: Vec<Option<usize>> = vec![None; c];
    #[allow(non_snake_case)]
    let mut Q: VecDeque<G::NodeId> = VecDeque::with_capacity(c);

    sigma[graph.to_index(*node_s)] = 1.;
    distance[graph.to_index(*node_s)] = Some(0);
    Q.push_back(*node_s);
    while let Some(v) = Q.pop_front() {
        verts_sorted_by_distance.push(v);
        let v_index = graph.to_index(v);
        let distance_v = distance[v_index].unwrap();
        for edge in graph.edges(v) {
            let w = edge.target();
            let w_index = graph.to_index(w);
            if distance[w_index].is_none() {
                Q.push_back(w);
                distance[w_index] = Some(distance_v + 1);
            }
            if distance[w_index] == Some(distance_v + 1) {
                sigma[w_index] += sigma[v_index];
                let e_p = predecessors.get_mut(w_index).unwrap();
                e_p.push(v);
                predecessor_edges.get_mut(w_index).unwrap().push(edge.id());
            }
        }
    }
    verts_sorted_by_distance.reverse(); // will be effectively popping from the stack
    ShortestPathDataWithEdges {
        verts_sorted_by_distance,
        predecessors,
        predecessor_edges,
        sigma,
    }
}

#[cfg(test)]
mod test_edge_betweenness_centrality {
    use crate::centrality::edge_betweenness_centrality;
    use petgraph::Undirected;
    use petgraph::graph::edge_index;
    use petgraph::prelude::StableGraph;

    macro_rules! assert_almost_equal {
        ($x:expr, $y:expr, $d:expr) => {
            if ($x - $y).abs() >= $d {
                panic!("{} != {} within delta of {}", $x, $y, $d);
            }
        };
    }

    #[test]
    fn test_undirected_graph_normalized() {
        let graph = petgraph::graph::UnGraph::<(), ()>::from_edges([
            (0, 6),
            (0, 4),
            (0, 1),
            (0, 5),
            (1, 6),
            (1, 7),
            (1, 3),
            (1, 4),
            (2, 6),
            (2, 3),
            (3, 5),
            (3, 7),
            (3, 6),
            (4, 5),
            (5, 6),
        ]);
        let output = edge_betweenness_centrality(&graph, true, 50);
        let result = output.iter().map(|x| x.unwrap()).collect::<Vec<f64>>();
        let expected_values = [
            0.1023809, 0.0547619, 0.0922619, 0.05654762, 0.09940476, 0.125, 0.09940476, 0.12440476,
            0.12857143, 0.12142857, 0.13511905, 0.125, 0.06547619, 0.08869048, 0.08154762,
        ];
        for i in 0..15 {
            assert_almost_equal!(result[i], expected_values[i], 1e-4);
        }
    }

    #[test]
    fn test_undirected_graph_unnormalized() {
        let graph = petgraph::graph::UnGraph::<(), ()>::from_edges([
            (0, 2),
            (0, 4),
            (0, 1),
            (1, 3),
            (1, 5),
            (1, 7),
            (2, 7),
            (2, 3),
            (3, 5),
            (3, 6),
            (4, 6),
            (5, 7),
        ]);
        let output = edge_betweenness_centrality(&graph, false, 50);
        let result = output.iter().map(|x| x.unwrap()).collect::<Vec<f64>>();
        let expected_values = [
            3.83333, 5.5, 5.33333, 3.5, 2.5, 3.0, 3.5, 4.0, 3.66667, 6.5, 3.5, 2.16667,
        ];
        for i in 0..12 {
            assert_almost_equal!(result[i], expected_values[i], 1e-4);
        }
    }

    #[test]
    fn test_directed_graph_normalized() {
        let graph = petgraph::graph::DiGraph::<(), ()>::from_edges([
            (0, 1),
            (1, 0),
            (1, 3),
            (1, 2),
            (1, 4),
            (2, 3),
            (2, 4),
            (2, 1),
            (3, 2),
            (4, 3),
        ]);
        let output = edge_betweenness_centrality(&graph, true, 50);
        let result = output.iter().map(|x| x.unwrap()).collect::<Vec<f64>>();
        let expected_values = [0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.1, 0.3, 0.35, 0.2];
        for i in 0..10 {
            assert_almost_equal!(result[i], expected_values[i], 1e-4);
        }
    }

    #[test]
    fn test_directed_graph_unnormalized() {
        let graph = petgraph::graph::DiGraph::<(), ()>::from_edges([
            (0, 4),
            (1, 0),
            (1, 3),
            (2, 3),
            (2, 4),
            (2, 0),
            (3, 4),
            (3, 2),
            (3, 1),
            (4, 1),
        ]);
        let output = edge_betweenness_centrality(&graph, false, 50);
        let result = output.iter().map(|x| x.unwrap()).collect::<Vec<f64>>();
        let expected_values = [4.5, 3.0, 6.5, 1.5, 1.5, 1.5, 1.5, 4.5, 2.0, 7.5];
        for i in 0..10 {
            assert_almost_equal!(result[i], expected_values[i], 1e-4);
        }
    }

    #[test]
    fn test_stable_graph_with_removed_edges() {
        let mut graph: StableGraph<(), (), Undirected> =
            StableGraph::from_edges([(0, 1), (1, 2), (2, 3), (3, 0)]);
        graph.remove_edge(edge_index(1));
        let result = edge_betweenness_centrality(&graph, false, 50);
        let expected_values = vec![Some(3.0), None, Some(3.0), Some(4.0)];
        assert_eq!(result, expected_values);
    }

    #[test]
    fn test_stable_graph_with_removed_nodes_and_edges() {
        let mut graph: StableGraph<(), (), Undirected> = StableGraph::default();
        let n0 = graph.add_node(());
        let d0 = graph.add_node(());
        let n1 = graph.add_node(());
        let d1 = graph.add_node(());
        let n2 = graph.add_node(());
        let d2 = graph.add_node(());
        let n3 = graph.add_node(());

        graph.remove_node(d0);
        graph.remove_node(d1);
        graph.remove_node(d2);

        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n3, ());
        graph.add_edge(n3, n0, ());

        graph.remove_edge(edge_index(1));
        let result = edge_betweenness_centrality(&graph, false, 50);
        let expected_values = vec![Some(3.0), None, Some(3.0), Some(4.0)];
        assert_eq!(result, expected_values);
    }
}

#[cfg(test)]
mod test_betweenness_centrality {
    use crate::centrality::betweenness_centrality;
    use petgraph::Undirected;
    use petgraph::graph::edge_index;
    use petgraph::prelude::StableGraph;

    #[test]
    fn test_stable_graph_with_removed_nodes_and_edges() {
        let mut graph: StableGraph<(), (), Undirected> = StableGraph::default();
        let n0 = graph.add_node(());
        let d0 = graph.add_node(());
        let n1 = graph.add_node(());
        let d1 = graph.add_node(());
        let n2 = graph.add_node(());
        let d2 = graph.add_node(());
        let n3 = graph.add_node(());

        graph.remove_node(d0);
        graph.remove_node(d1);
        graph.remove_node(d2);

        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n3, ());
        graph.add_edge(n3, n0, ());
        graph.remove_edge(edge_index(1));

        let result = betweenness_centrality(&graph, false, false, 50);
        let expected_values = vec![Some(2.0), None, Some(0.0), None, Some(0.0), None, Some(2.0)];
        assert_eq!(result, expected_values);
    }
}

/// Compute the eigenvector centrality of a graph
///
/// For details on the eigenvector centrality refer to:
///
/// Phillip Bonacich. “Power and Centrality: A Family of Measures.”
/// American Journal of Sociology 92(5):1170–1182, 1986
/// <https://doi.org/10.1086/228631>
///
/// This function uses a power iteration method to compute the eigenvector
/// and convergence is not guaranteed. The function will stop when `max_iter`
/// iterations is reached or when the computed vector between two iterations
/// is smaller than the error tolerance multiplied by the number of nodes.
/// The implementation of this algorithm is based on the NetworkX
/// [`eigenvector_centrality()`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.eigenvector_centrality.html)
/// function.
///
/// In the case of multigraphs the weights of any parallel edges will be
/// summed when computing the eigenvector centrality.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `weight_fn` - An input callable that will be passed the `EdgeRef` for
///   an edge in the graph and is expected to return a `Result<f64>` of
///   the weight of that edge.
/// * `max_iter` - The maximum number of iterations in the power method. If
///   set to `None` a default value of 100 is used.
/// * `tol` - The error tolerance used when checking for convergence in the
///   power method. If set to `None` a default value of 1e-6 is used.
///
/// # Example
/// ```rust
/// use rustworkx_core::Result;
/// use rustworkx_core::petgraph;
/// use rustworkx_core::petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
/// use rustworkx_core::centrality::eigenvector_centrality;
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2)
/// ]);
/// // Calculate the eigenvector centrality
/// let output: Result<Option<Vec<f64>>> = eigenvector_centrality(&g, |_| {Ok(1.)}, None, None);
/// ```
pub fn eigenvector_centrality<G, F, E>(
    graph: G,
    mut weight_fn: F,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> Result<Option<Vec<f64>>, E>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoNeighbors + IntoEdges + NodeCount,
    G::NodeId: Eq,
    F: FnMut(G::EdgeRef) -> Result<f64, E>,
{
    let tol: f64 = tol.unwrap_or(1e-6);
    let max_iter = max_iter.unwrap_or(100);
    let mut x: Vec<f64> = vec![1.; graph.node_bound()];
    let node_count = graph.node_count();
    for _ in 0..max_iter {
        let x_last = x.clone();
        for node_index in graph.node_identifiers() {
            let node = graph.to_index(node_index);
            for edge in graph.edges(node_index) {
                let w = weight_fn(edge)?;
                let neighbor = edge.target();
                x[graph.to_index(neighbor)] += x_last[node] * w;
            }
        }
        let norm: f64 = x.iter().map(|val| val.powi(2)).sum::<f64>().sqrt();
        if norm == 0. {
            return Ok(None);
        }
        for v in x.iter_mut() {
            *v /= norm;
        }
        if (0..x.len())
            .map(|node| (x[node] - x_last[node]).abs())
            .sum::<f64>()
            < node_count as f64 * tol
        {
            return Ok(Some(x));
        }
    }
    Ok(None)
}

/// Compute the Katz centrality of a graph
///
/// For details on the Katz centrality refer to:
///
/// Leo Katz. “A New Status Index Derived from Sociometric Index.”
/// Psychometrika 18(1):39–43, 1953
/// <https://link.springer.com/content/pdf/10.1007/BF02289026.pdf>
///
/// This function uses a power iteration method to compute the eigenvector
/// and convergence is not guaranteed. The function will stop when `max_iter`
/// iterations is reached or when the computed vector between two iterations
/// is smaller than the error tolerance multiplied by the number of nodes.
/// The implementation of this algorithm is based on the NetworkX
/// [`katz_centrality()`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.katz_centrality.html)
/// function.
///
/// In the case of multigraphs the weights of any parallel edges will be
/// summed when computing the eigenvector centrality.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `weight_fn` - An input callable that will be passed the `EdgeRef` for
///   an edge in the graph and is expected to return a `Result<f64>` of
///   the weight of that edge.
/// * `alpha` - Attenuation factor. If set to `None`, a default value of 0.1 is used.
/// * `beta_map` - Immediate neighbourhood weights. Must contain all node indices or be `None`.
/// * `beta_scalar` - Immediate neighbourhood scalar that replaces `beta_map` in case `beta_map` is None.
///   Defaults to 1.0 in case `None` is provided.
/// * `max_iter` - The maximum number of iterations in the power method. If
///   set to `None` a default value of 100 is used.
/// * `tol` - The error tolerance used when checking for convergence in the
///   power method. If set to `None` a default value of 1e-6 is used.
///
/// # Example
/// ```rust
/// use rustworkx_core::Result;
/// use rustworkx_core::petgraph;
/// use rustworkx_core::petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
/// use rustworkx_core::centrality::katz_centrality;
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2)
/// ]);
/// // Calculate the eigenvector centrality
/// let output: Result<Option<Vec<f64>>> = katz_centrality(&g, |_| {Ok(1.)}, None, None, None, None, None);
/// let centralities = output.unwrap().unwrap();
/// assert!(centralities[1] > centralities[0], "Node 1 is more central than node 0");
/// assert!(centralities[1] > centralities[2], "Node 1 is more central than node 2");
/// ```
pub fn katz_centrality<G, F, E>(
    graph: G,
    mut weight_fn: F,
    alpha: Option<f64>,
    beta_map: Option<HashMap<usize, f64>>,
    beta_scalar: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> Result<Option<Vec<f64>>, E>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoNeighbors + IntoEdges + NodeCount,
    G::NodeId: Eq,
    F: FnMut(G::EdgeRef) -> Result<f64, E>,
{
    let alpha: f64 = alpha.unwrap_or(0.1);

    let beta: HashMap<usize, f64> = beta_map.unwrap_or_default();
    //Initialize the beta vector in case a beta map was not provided
    let mut beta_v = vec![beta_scalar.unwrap_or(1.0); graph.node_bound()];

    if !beta.is_empty() {
        // Check if beta contains all node indices
        for node_index in graph.node_identifiers() {
            let node = graph.to_index(node_index);
            if !beta.contains_key(&node) {
                return Ok(None); // beta_map was provided but did not include all nodes
            }
            beta_v[node] = *beta.get(&node).unwrap(); //Initialize the beta vector with the provided values
        }
    }

    let tol: f64 = tol.unwrap_or(1e-6);
    let max_iter = max_iter.unwrap_or(1000);

    let mut x: Vec<f64> = vec![0.; graph.node_bound()];
    let node_count = graph.node_count();
    for _ in 0..max_iter {
        let x_last = x.clone();
        x = vec![0.; graph.node_bound()];
        for node_index in graph.node_identifiers() {
            let node = graph.to_index(node_index);
            for edge in graph.edges(node_index) {
                let w = weight_fn(edge)?;
                let neighbor = edge.target();
                x[graph.to_index(neighbor)] += x_last[node] * w;
            }
        }
        for node_index in graph.node_identifiers() {
            let node = graph.to_index(node_index);
            x[node] = alpha * x[node] + beta_v[node];
        }
        if (0..x.len())
            .map(|node| (x[node] - x_last[node]).abs())
            .sum::<f64>()
            < node_count as f64 * tol
        {
            // Normalize vector
            let norm: f64 = x.iter().map(|val| val.powi(2)).sum::<f64>().sqrt();
            if norm == 0. {
                return Ok(None);
            }
            for v in x.iter_mut() {
                *v /= norm;
            }

            return Ok(Some(x));
        }
    }

    Ok(None)
}

#[cfg(test)]
mod test_eigenvector_centrality {

    use crate::Result;
    use crate::centrality::eigenvector_centrality;
    use crate::petgraph;

    macro_rules! assert_almost_equal {
        ($x:expr, $y:expr, $d:expr) => {
            if ($x - $y).abs() >= $d {
                panic!("{} != {} within delta of {}", $x, $y, $d);
            }
        };
    }
    #[test]
    fn test_no_convergence() {
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges([(0, 1), (1, 2)]);
        let output: Result<Option<Vec<f64>>> =
            eigenvector_centrality(&g, |_| Ok(1.), Some(0), None);
        let result = output.unwrap();
        assert_eq!(None, result);
    }

    #[test]
    fn test_undirected_complete_graph() {
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges([
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
        ]);
        let output: Result<Option<Vec<f64>>> = eigenvector_centrality(&g, |_| Ok(1.), None, None);
        let result = output.unwrap().unwrap();
        let expected_value: f64 = (1_f64 / 5_f64).sqrt();
        let expected_values: Vec<f64> = vec![expected_value; 5];
        for i in 0..5 {
            assert_almost_equal!(expected_values[i], result[i], 1e-4);
        }
    }

    #[test]
    fn test_undirected_path_graph() {
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges([(0, 1), (1, 2)]);
        let output: Result<Option<Vec<f64>>> = eigenvector_centrality(&g, |_| Ok(1.), None, None);
        let result = output.unwrap().unwrap();
        let expected_values: Vec<f64> = vec![0.5, std::f64::consts::FRAC_1_SQRT_2, 0.5];
        for i in 0..3 {
            assert_almost_equal!(expected_values[i], result[i], 1e-4);
        }
    }

    #[test]
    fn test_directed_graph() {
        let g = petgraph::graph::DiGraph::<i32, ()>::from_edges([
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 1),
            (2, 4),
            (3, 1),
            (3, 4),
            (3, 5),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 7),
            (6, 0),
            (6, 4),
            (6, 7),
            (7, 5),
            (7, 6),
        ]);
        let output: Result<Option<Vec<f64>>> = eigenvector_centrality(&g, |_| Ok(2.), None, None);
        let result = output.unwrap().unwrap();
        let expected_values: Vec<f64> = vec![
            0.2140437, 0.2009269, 0.1036383, 0.0972886, 0.3113323, 0.4891686, 0.4420605, 0.6016448,
        ];
        for i in 0..8 {
            assert_almost_equal!(expected_values[i], result[i], 1e-4);
        }
    }
}

#[cfg(test)]
mod test_katz_centrality {

    use crate::Result;
    use crate::centrality::katz_centrality;
    use crate::petgraph;
    use hashbrown::HashMap;

    macro_rules! assert_almost_equal {
        ($x:expr, $y:expr, $d:expr) => {
            if ($x - $y).abs() >= $d {
                panic!("{} != {} within delta of {}", $x, $y, $d);
            }
        };
    }
    #[test]
    fn test_no_convergence() {
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges([(0, 1), (1, 2)]);
        let output: Result<Option<Vec<f64>>> =
            katz_centrality(&g, |_| Ok(1.), None, None, None, Some(0), None);
        let result = output.unwrap();
        assert_eq!(None, result);
    }

    #[test]
    fn test_incomplete_beta() {
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges([(0, 1), (1, 2)]);
        let beta_map: HashMap<usize, f64> = [(0, 1.0)].iter().cloned().collect();
        let output: Result<Option<Vec<f64>>> =
            katz_centrality(&g, |_| Ok(1.), None, Some(beta_map), None, None, None);
        let result = output.unwrap();
        assert_eq!(None, result);
    }

    #[test]
    fn test_complete_beta() {
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges([(0, 1), (1, 2)]);
        let beta_map: HashMap<usize, f64> =
            [(0, 0.5), (1, 1.0), (2, 0.5)].iter().cloned().collect();
        let output: Result<Option<Vec<f64>>> =
            katz_centrality(&g, |_| Ok(1.), None, Some(beta_map), None, None, None);
        let result = output.unwrap().unwrap();
        let expected_values: Vec<f64> =
            vec![0.4318894504492167, 0.791797325823564, 0.4318894504492167];
        for i in 0..3 {
            assert_almost_equal!(expected_values[i], result[i], 1e-4);
        }
    }

    #[test]
    fn test_undirected_complete_graph() {
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges([
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
        ]);
        let output: Result<Option<Vec<f64>>> =
            katz_centrality(&g, |_| Ok(1.), Some(0.2), None, Some(1.1), None, None);
        let result = output.unwrap().unwrap();
        let expected_value: f64 = (1_f64 / 5_f64).sqrt();
        let expected_values: Vec<f64> = vec![expected_value; 5];
        for i in 0..5 {
            assert_almost_equal!(expected_values[i], result[i], 1e-4);
        }
    }

    #[test]
    fn test_directed_graph() {
        let g = petgraph::graph::DiGraph::<i32, ()>::from_edges([
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 1),
            (2, 4),
            (3, 1),
            (3, 4),
            (3, 5),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 7),
            (6, 0),
            (6, 4),
            (6, 7),
            (7, 5),
            (7, 6),
        ]);
        let output: Result<Option<Vec<f64>>> =
            katz_centrality(&g, |_| Ok(1.), None, None, None, None, None);
        let result = output.unwrap().unwrap();
        let expected_values: Vec<f64> = vec![
            0.3135463087489011,
            0.3719056758615039,
            0.3094350787809586,
            0.31527101632646026,
            0.3760169058294464,
            0.38618584417917906,
            0.35465874858087904,
            0.38976653416801743,
        ];

        for i in 0..8 {
            assert_almost_equal!(expected_values[i], result[i], 1e-4);
        }
    }
}

/// Compute the closeness centrality of each node in the graph.
///
/// The closeness centrality of a node `u` is the reciprocal of the average
/// shortest path distance to `u` over all `n-1` reachable nodes.
///
/// In the case of a graphs with more than one connected component there is
/// an alternative improved formula that calculates the closeness centrality
/// as "a ratio of the fraction of actors in the group who are reachable, to
/// the average distance".[^WF]
/// You can enable this by setting `wf_improved` to `true`.
///
/// [^WF]: Wasserman, S., & Faust, K. (1994). Social Network Analysis:
///     Methods and Applications (Structural Analysis in the Social Sciences).
///     Cambridge: Cambridge University Press.
///     <https://doi.org/10.1017/CBO9780511815478>
///
/// This function is multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold``. If the
/// function will be running in parallel the env var ``RAYON_NUM_THREADS`` can
/// be used to adjust how many threads will be used.
///
/// # Arguments
///
/// * `graph` - The graph object to run the algorithm on
/// * `wf_improved` - If `true`, scale by the fraction of nodes reachable.
/// * `parallel_threshold` - The number of nodes to calculate the betweenness
///   centrality in parallel at, if the number of nodes in `graph` is less
///   than this value it will run in a single thread. The suggested default to use
///   here is `50`.
///
/// # Example
///
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::centrality::closeness_centrality;
///
/// // Calculate the closeness centrality of Graph
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 4), (1, 2), (2, 3), (3, 4), (1, 4)
/// ]);
/// let output = closeness_centrality(&g, true, 200);
/// assert_eq!(
///     vec![Some(1./2.), Some(2./3.), Some(4./7.), Some(2./3.), Some(4./5.)],
///     output
/// );
///
/// // Calculate the closeness centrality of DiGraph
/// let dg = petgraph::graph::DiGraph::<i32, ()>::from_edges(&[
///     (0, 4), (1, 2), (2, 3), (3, 4), (1, 4)
/// ]);
/// let output = closeness_centrality(&dg, true, 200);
/// assert_eq!(
///     vec![Some(0.), Some(0.), Some(1./4.), Some(1./3.), Some(4./5.)],
///     output
/// );
/// ```
pub fn closeness_centrality<G>(
    graph: G,
    wf_improved: bool,
    parallel_threshold: usize,
) -> Vec<Option<f64>>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + GraphBase
        + IntoEdges
        + Visitable
        + NodeCount
        + IntoEdgesDirected
        + std::marker::Sync,
    G::NodeId: Eq + Hash + Send,
    G::EdgeId: Eq + Hash + Send,
{
    let max_index = graph.node_bound();
    let mut node_indices: Vec<Option<G::NodeId>> = vec![None; max_index];
    graph.node_identifiers().for_each(|node| {
        let index = graph.to_index(node);
        node_indices[index] = Some(node);
    });

    let unweighted_shortest_path = |g: Reversed<&G>, s: G::NodeId| -> HashMap<G::NodeId, usize> {
        let mut distances = HashMap::new();
        let mut bfs = Bfs::new(g, s);
        distances.insert(s, 0);
        while let Some(node) = bfs.next(g) {
            let distance = distances[&node];
            for edge in g.edges(node) {
                let target = edge.target();
                distances.entry(target).or_insert(distance + 1);
            }
        }
        distances
    };

    let closeness: Vec<Option<f64>> =
        CondIterator::new(node_indices, graph.node_count() >= parallel_threshold)
            .map(|node_s| {
                let node_s = node_s?;
                let map = unweighted_shortest_path(Reversed(&graph), node_s);
                let reachable_nodes_count = map.len();
                let dists_sum: usize = map.into_values().sum();
                if reachable_nodes_count == 1 {
                    return Some(0.0);
                }
                let mut centrality_s = Some((reachable_nodes_count - 1) as f64 / dists_sum as f64);
                if wf_improved {
                    let node_count = graph.node_count();
                    centrality_s = centrality_s
                        .map(|c| c * (reachable_nodes_count - 1) as f64 / (node_count - 1) as f64);
                }
                centrality_s
            })
            .collect();
    closeness
}

/// Compute the weighted closeness centrality of each node in the graph.
///
/// The weighted closeness centrality is an extension of the standard closeness
/// centrality measure where edge weights represent connection strength rather
/// than distance. To properly compute shortest paths, weights are inverted
/// so that stronger connections correspond to shorter effective distances.
/// The algorithm follows the method described by Newman (2001) in analyzing
/// weighted graphs.[^Newman]
///
/// The edges originally represent connection strength between nodes.
/// The idea is that if two nodes have a strong connection, the computed
/// distance between them should be small (shorter), and vice versa.
/// Note that this assume that the graph is modelling a measure of
/// connection strength (e.g. trust, collaboration, or similarity).
/// If the graph is not modelling a measure of connection strength,
/// the function `weight_fn` should invert the weights before calling this
/// function, if not it is considered as a logical error.
///
/// In the case of a graphs with more than one connected component there is
/// an alternative improved formula that calculates the closeness centrality
/// as "a ratio of the fraction of actors in the group who are reachable, to
/// the average distance".[^WF]
/// You can enable this by setting `wf_improved` to `true`.
///
/// [^Newman]: Newman, M. E. J. (2001). Scientific collaboration networks.
///     II. Shortest paths, weighted networks, and centrality.
///     Physical Review E, 64(1), 016132.
///
/// [^WF]: Wasserman, S., & Faust, K. (1994). Social Network Analysis:
///     Methods and Applications (Structural Analysis in the Social Sciences).
///     Cambridge: Cambridge University Press.
///     <https://doi.org/10.1017/CBO9780511815478>
///
/// This function is multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold``. If the
/// function will be running in parallel the env var ``RAYON_NUM_THREADS`` can
/// be used to adjust how many threads will be used.
///
/// # Arguments
/// * `graph` - The graph object to run the algorithm on
/// * `wf_improved` - If `true`, scale by the fraction of nodes reachable.
/// * `weight_fn` - An input callable that will be passed the
///   `ReversedEdgeReference<<G as IntoEdgeReferences>::EdgeRef>` for
///   an edge in the graph and is expected to return a `f64` of
///   the weight of that edge.
/// * `parallel_threshold` - The number of nodes to calculate the betweenness
///   centrality in parallel at, if the number of nodes in `graph` is less
///   than this value it will run in a single thread. The suggested default to use
///   here is `50`.
///
/// # Example
///
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::centrality::newman_weighted_closeness_centrality;
/// use crate::rustworkx_core::petgraph::visit::EdgeRef;
///
/// // Calculate the closeness centrality of Graph
/// let g = petgraph::graph::UnGraph::<i32, f64>::from_edges(&[
///     (0, 1, 0.7), (1, 2, 0.2), (2, 3, 0.5),
/// ]);
/// let output = newman_weighted_closeness_centrality(&g, false, |x| *x.weight(), 200);
/// assert!(output[1] > output[3]);
///
/// // Calculate the closeness centrality of DiGraph
/// let g = petgraph::graph::DiGraph::<i32, f64>::from_edges(&[
///     (0, 1, 0.7), (1, 2, 0.2), (2, 3, 0.5),
/// ]);
/// let output = newman_weighted_closeness_centrality(&g, false, |x| *x.weight(), 200);
/// assert!(output[1] > output[3]);
/// ```
pub fn newman_weighted_closeness_centrality<G, F>(
    graph: G,
    wf_improved: bool,
    weight_fn: F,
    parallel_threshold: usize,
) -> Vec<Option<f64>>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + GraphBase
        + IntoEdges
        + Visitable
        + NodeCount
        + IntoEdgesDirected
        + std::marker::Sync,
    G::NodeId: Eq + Hash + Send,
    G::EdgeId: Eq + Hash + Send,
    F: Fn(ReversedEdgeReference<<G as IntoEdgeReferences>::EdgeRef>) -> f64 + Sync,
{
    // The edges originally represent `connection strength` between nodes.
    // As shown in the paper, the weight of the edges should be inverted to
    // ensure that stronger ties correspond to shorter effective distances.
    // The idea is that if two nodes have a strong connection, the computed
    // distance between them should be small (shorter), and vice versa.
    //
    // Note that this assume that the graph is modelling a measure of
    // connection strength (e.g. trust, collaboration, or similarity).
    // If the graph is not modelling a measure of connection strength,
    // the user should invert the weights before calling this function,
    // if not it is considered as a logical error.
    let inverted_weight_fn =
        |x: ReversedEdgeReference<<G as IntoEdgeReferences>::EdgeRef>| 1.0 / weight_fn(x);

    let max_index = graph.node_bound();
    let mut node_indices: Vec<Option<G::NodeId>> = vec![None; max_index];
    graph.node_identifiers().for_each(|node| {
        let index = graph.to_index(node);
        node_indices[index] = Some(node);
    });

    let closeness: Vec<Option<f64>> =
        CondIterator::new(node_indices, graph.node_count() >= parallel_threshold)
            .map(|node_s| {
                let node_s = node_s?;
                let map = dijkstra(Reversed(&graph), node_s, None, &inverted_weight_fn);
                let reachable_nodes_count = map.len();
                let dists_sum: f64 = map.into_values().sum();
                if reachable_nodes_count == 1 {
                    return Some(0.0);
                }
                let mut centrality_s = Some((reachable_nodes_count - 1) as f64 / dists_sum as f64);
                if wf_improved {
                    let node_count = graph.node_count();
                    centrality_s = centrality_s
                        .map(|c| c * (reachable_nodes_count - 1) as f64 / (node_count - 1) as f64);
                }
                centrality_s
            })
            .collect();
    closeness
}

#[cfg(test)]
mod test_newman_weighted_closeness_centrality {
    use crate::centrality::closeness_centrality;

    use super::newman_weighted_closeness_centrality;
    use petgraph::visit::EdgeRef;

    macro_rules! assert_almost_equal {
        ($x:expr, $y:expr, $d:expr) => {
            if ($x - $y).abs() >= $d {
                panic!("{} != {} within delta of {}", $x, $y, $d);
            }
        };
    }

    macro_rules! assert_almost_equal_iter {
        ($expected:expr, $computed:expr, $tolerance:expr) => {
            for (&expected, &computed) in $expected.iter().zip($computed.iter()) {
                assert_almost_equal!(expected.unwrap(), computed.unwrap(), $tolerance);
            }
        };
    }

    #[test]
    fn test_weighted_closeness_graph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::UnGraph::<u32, f64>::from_edges([
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 3, 1.0),
                (3, 4, 1.0),
                (4, 5, 1.0),
                (5, 6, 1.0),
            ]);
            let classic_closeness = closeness_centrality(&g, false, parallel_threshold);
            let weighted_closeness = newman_weighted_closeness_centrality(
                &g,
                false,
                |x| *x.weight(),
                parallel_threshold,
            );

            assert_eq!(classic_closeness, weighted_closeness);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_the_same_as_closeness_centrality_when_weights_are_1_not_improved_digraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::DiGraph::<u32, f64>::from_edges([
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 3, 1.0),
                (3, 4, 1.0),
                (4, 5, 1.0),
                (5, 6, 1.0),
            ]);
            let classic_closeness = closeness_centrality(&g, false, parallel_threshold);
            let weighted_closeness = newman_weighted_closeness_centrality(
                &g,
                false,
                |x| *x.weight(),
                parallel_threshold,
            );

            assert_eq!(classic_closeness, weighted_closeness);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_the_same_as_closeness_centrality_when_weights_are_1_improved_digraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::DiGraph::<u32, f64>::from_edges([
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 3, 1.0),
                (3, 4, 1.0),
                (4, 5, 1.0),
                (5, 6, 1.0),
            ]);
            let classic_closeness = closeness_centrality(&g, true, parallel_threshold);
            let weighted_closeness =
                newman_weighted_closeness_centrality(&g, true, |x| *x.weight(), parallel_threshold);

            assert_eq!(classic_closeness, weighted_closeness);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_weighted_closeness_two_connected_components_not_improved_digraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::DiGraph::<u32, f64>::from_edges([
                (0, 1, 1.0),
                (1, 2, 0.5),
                (2, 3, 0.25),
                (4, 5, 1.0),
                (5, 6, 0.5),
                (6, 7, 0.25),
            ]);
            let c = newman_weighted_closeness_centrality(
                &g,
                false,
                |x| *x.weight(),
                parallel_threshold,
            );
            let result = [
                Some(0.0),
                Some(1.0),
                Some(0.4),
                Some(0.176470),
                Some(0.0),
                Some(1.0),
                Some(0.4),
                Some(0.176470),
            ];

            assert_almost_equal_iter!(result, c, 1e-4);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_weighted_closeness_two_connected_components_improved_digraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::DiGraph::<u32, f64>::from_edges([
                (0, 1, 1.0),
                (1, 2, 0.5),
                (2, 3, 0.25),
                (4, 5, 1.0),
                (5, 6, 0.5),
                (6, 7, 0.25),
            ]);
            let c =
                newman_weighted_closeness_centrality(&g, true, |x| *x.weight(), parallel_threshold);
            let result = [
                Some(0.0),
                Some(0.14285714),
                Some(0.11428571),
                Some(0.07563025),
                Some(0.0),
                Some(0.14285714),
                Some(0.11428571),
                Some(0.07563025),
            ];

            assert_almost_equal_iter!(result, c, 1e-4);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_weighted_closeness_two_connected_components_improved_different_cardinality_digraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::DiGraph::<u32, f64>::from_edges([
                (0, 1, 1.0),
                (1, 2, 0.5),
                (2, 3, 0.25),
                (4, 5, 1.0),
                (5, 6, 0.5),
                (6, 7, 0.25),
                (7, 8, 0.125),
            ]);
            let c =
                newman_weighted_closeness_centrality(&g, true, |x| *x.weight(), parallel_threshold);
            let result = [
                Some(0.0),
                Some(0.125),
                Some(0.1),
                Some(0.06617647),
                Some(0.0),
                Some(0.125),
                Some(0.1),
                Some(0.06617647),
                Some(0.04081632),
            ];

            assert_almost_equal_iter!(result, c, 1e-4);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_weighted_closeness_small_ungraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::UnGraph::<u32, f64>::from_edges([
                (0, 1, 0.7),
                (1, 2, 0.2),
                (2, 3, 0.5),
            ]);
            let c = newman_weighted_closeness_centrality(
                &g,
                false,
                |x| *x.weight(),
                parallel_threshold,
            );
            let result = [
                Some(0.1842105),
                Some(0.2234042),
                Some(0.2234042),
                Some(0.1721311),
            ];

            assert_almost_equal_iter!(result, c, 1e-4);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }
    #[test]
    fn test_weighted_closeness_small_digraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::DiGraph::<u32, f64>::from_edges([
                (0, 1, 0.7),
                (1, 2, 0.2),
                (2, 3, 0.5),
            ]);
            let c = newman_weighted_closeness_centrality(
                &g,
                false,
                |x| *x.weight(),
                parallel_threshold,
            );
            let result = [Some(0.0), Some(0.7), Some(0.175), Some(0.172131)];

            assert_almost_equal_iter!(result, c, 1e-4);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_weighted_closeness_many_to_one_connected_digraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::DiGraph::<u32, f64>::from_edges([
                (1, 0, 0.1),
                (2, 0, 0.1),
                (3, 0, 0.1),
                (4, 0, 0.1),
                (5, 0, 0.1),
                (6, 0, 0.1),
                (7, 0, 0.1),
                (0, 8, 1.0),
            ]);
            let c = newman_weighted_closeness_centrality(
                &g,
                false,
                |x| *x.weight(),
                parallel_threshold,
            );
            let result = [
                Some(0.1),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.10256),
            ];

            assert_almost_equal_iter!(result, c, 1e-4);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_weighted_closeness_many_to_one_connected_ungraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::UnGraph::<u32, f64>::from_edges([
                (1, 0, 0.1),
                (2, 0, 0.1),
                (3, 0, 0.1),
                (4, 0, 0.1),
                (5, 0, 0.1),
                (6, 0, 0.1),
                (7, 0, 0.1),
                (0, 8, 1.0),
            ]);
            let c = newman_weighted_closeness_centrality(
                &g,
                false,
                |x| *x.weight(),
                parallel_threshold,
            );
            let result = [
                Some(0.112676056),
                Some(0.056737588),
                Some(0.056737588),
                Some(0.056737588),
                Some(0.056737588),
                Some(0.056737588),
                Some(0.056737588),
                Some(0.056737588),
                Some(0.102564102),
            ];

            assert_almost_equal_iter!(result, c, 1e-4);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_weighted_closeness_many_to_one_not_connected_2_digraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::DiGraph::<u32, f64>::from_edges([
                (1, 0, 0.1),
                (2, 0, 0.1),
                (3, 0, 0.1),
                (4, 0, 0.1),
                (5, 0, 0.1),
                (6, 0, 0.1),
                (7, 0, 0.1),
                (1, 7, 1.0),
            ]);
            let c = newman_weighted_closeness_centrality(
                &g,
                false,
                |x| *x.weight(),
                parallel_threshold,
            );
            let result = [
                Some(0.1),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(1.0),
            ];

            assert_eq!(result, *c);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }

    #[test]
    fn test_weighted_closeness_many_to_one_not_connected_1_digraph() {
        let test_case = |parallel_threshold: usize| {
            let g = petgraph::graph::DiGraph::<u32, f64>::from_edges([
                (1, 0, 0.1),
                (2, 0, 0.1),
                (3, 0, 0.1),
                (4, 0, 0.1),
                (5, 0, 0.1),
                (6, 0, 0.1),
                (7, 0, 0.1),
                (8, 7, 1.0),
            ]);
            let c = newman_weighted_closeness_centrality(
                &g,
                false,
                |x| *x.weight(),
                parallel_threshold,
            );
            let result = [
                Some(0.098765),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(1.0),
                Some(0.0),
            ];

            assert_almost_equal_iter!(result, c, 1e-4);
        };
        test_case(200); // sequential
        test_case(1); // parallel
    }
}

/// Compute the group degree centrality of a set of nodes.
///
/// Group degree centrality measures the fraction of non-group nodes that are
/// connected to at least one member of the group. It is defined as:
///
/// C_D(S) = |N(S) \ S| / (|V| - |S|)
///
/// where N(S) is the union of neighborhoods of all nodes in S.
///
/// Based on: Everett, M. G., & Borgatti, S. P. (1999).
/// The centrality of groups and classes.
/// Journal of Mathematical Sociology, 23(3), 181-201.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `group` - A slice of node indices representing the group
/// * `direction` - Optional direction for directed graphs:
///     - `None` uses outgoing edges (default)
///     - `Some(Incoming)` counts nodes with edges into the group
///     - `Some(Outgoing)` counts nodes reachable from the group
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::centrality::group_degree_centrality;
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2), (2, 3), (3, 4)
/// ]);
/// let output = group_degree_centrality(&g, &[0, 1], None);
/// // Nodes 0,1 are the group. Neighbors of {0,1} outside the group = {2}.
/// // So centrality = 1 / (5 - 2) = 1/3.
/// assert!((output - 1.0 / 3.0).abs() < 1e-10);
/// ```
pub fn group_degree_centrality<G>(
    graph: G,
    group: &[usize],
    direction: Option<petgraph::Direction>,
) -> f64
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighbors
        + IntoNeighborsDirected
        + NodeCount
        + GraphProp,
    G::NodeId: Eq + Hash,
{
    let node_count = graph.node_count();
    let group_size = group.len();
    if group_size >= node_count {
        return 0.0;
    }

    let group_set: HashSet<usize> = group.iter().copied().collect();
    let mut reached: HashSet<usize> = HashSet::new();

    for &node_idx in group {
        let node_id = graph.from_index(node_idx);
        let neighbors: Box<dyn Iterator<Item = G::NodeId>> = match direction {
            Some(dir) => Box::new(graph.neighbors_directed(node_id, dir)),
            None => Box::new(graph.neighbors(node_id)),
        };
        for neighbor in neighbors {
            let neighbor_idx = graph.to_index(neighbor);
            if !group_set.contains(&neighbor_idx) {
                reached.insert(neighbor_idx);
            }
        }
    }

    reached.len() as f64 / (node_count - group_size) as f64
}

/// Compute the group closeness centrality of a set of nodes.
///
/// Group closeness centrality measures how close a group of nodes is to
/// all non-group nodes. It is defined as:
///
/// C_close(S) = |V \ S| / sum_{v in V\S} d(S, v)
///
/// where d(S, v) = min_{u in S} d(u, v) is the minimum distance from any
/// group member to node v.
///
/// Based on: Everett, M. G., & Borgatti, S. P. (1999).
/// The centrality of groups and classes.
/// Journal of Mathematical Sociology, 23(3), 181-201.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `group` - A slice of node indices representing the group
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::centrality::group_closeness_centrality;
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2), (2, 3), (3, 4)
/// ]);
/// let output = group_closeness_centrality(&g, &[0, 1]);
/// // Group = {0, 1}. Non-group = {2, 3, 4}.
/// // d({0,1}, 2) = 1, d({0,1}, 3) = 2, d({0,1}, 4) = 3. Sum = 6.
/// // Closeness = 3 / 6 = 0.5
/// assert!((output - 0.5).abs() < 1e-10);
/// ```
pub fn group_closeness_centrality<G>(graph: G, group: &[usize]) -> f64
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + GraphBase
        + IntoEdges
        + IntoEdgesDirected
        + Visitable
        + NodeCount,
    G::NodeId: Eq + Hash,
    G::EdgeId: Eq + Hash,
{
    let node_count = graph.node_count();
    let group_size = group.len();
    if group_size >= node_count {
        return 0.0;
    }

    let group_set: HashSet<usize> = group.iter().copied().collect();

    // Multi-source BFS on Reversed graph (incoming edges), matching the
    // convention used by NX and by per-node closeness_centrality: d(S,v)
    // is the distance from v to the nearest group member.
    let reversed = Reversed(&graph);
    let max_index = graph.node_bound();
    let mut distance: Vec<Option<usize>> = vec![None; max_index];
    let mut queue: VecDeque<G::NodeId> = VecDeque::new();

    for &node_idx in group {
        let node_id = graph.from_index(node_idx);
        distance[node_idx] = Some(0);
        queue.push_back(node_id);
    }

    while let Some(v) = queue.pop_front() {
        let v_idx = graph.to_index(v);
        let dist_v = distance[v_idx].unwrap();
        for edge in reversed.edges(v) {
            let w = edge.target();
            let w_idx = graph.to_index(w);
            if distance[w_idx].is_none() {
                distance[w_idx] = Some(dist_v + 1);
                queue.push_back(w);
            }
        }
    }

    let mut dist_sum: usize = 0;
    for node in graph.node_identifiers() {
        let idx = graph.to_index(node);
        if group_set.contains(&idx) {
            continue;
        }
        if let Some(d) = distance[idx] {
            dist_sum += d;
        }
    }

    if dist_sum == 0 {
        return 0.0;
    }

    (node_count - group_size) as f64 / dist_sum as f64
}

/// Compute the group betweenness centrality of a set of nodes.
///
/// Group betweenness centrality measures the fraction of shortest paths
/// between non-group node pairs that pass through at least one group member.
/// It is defined as:
///
/// C_B(S) = sum_{s,t in V\S} sigma(s,t|S) / sigma(s,t)
///
/// where sigma(s,t) is the number of shortest paths from s to t, and
/// sigma(s,t|S) is the number of those paths passing through at least
/// one node in S.
///
/// Based on: Everett, M. G., & Borgatti, S. P. (1999).
/// The centrality of groups and classes.
/// Journal of Mathematical Sociology, 23(3), 181-201.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `group` - A slice of node indices representing the group
/// * `normalized` - Whether to normalize the result
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::centrality::group_betweenness_centrality;
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2), (2, 3), (3, 4)
/// ]);
/// let output = group_betweenness_centrality(&g, &[2], true);
/// // Node 2 is on every shortest path between {0,1} and {3,4}.
/// assert!(output > 0.0);
/// ```
pub fn group_betweenness_centrality<G>(
    graph: G,
    group: &[usize],
    normalized: bool,
) -> f64
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + NodeCount
        + GraphProp
        + GraphBase,
    G::NodeId: Eq + Hash,
{
    let node_count = graph.node_count();
    let group_size = group.len();

    if group_size == 0 || node_count <= 1 {
        return 0.0;
    }

    let group_set: HashSet<usize> = group.iter().copied().collect();
    let max_index = graph.node_bound();

    // For each non-group source, run BFS on the full graph and on the graph
    // with group nodes removed. The difference in path counts gives us the
    // fraction of shortest paths passing through the group.
    let mut group_betweenness: f64 = 0.0;

    let node_ids: Vec<G::NodeId> = graph.node_identifiers().collect();

    for &source_id in &node_ids {
        let source_idx = graph.to_index(source_id);
        if group_set.contains(&source_idx) {
            continue;
        }

        // BFS on full graph from source
        let mut sigma_full = vec![0.0_f64; max_index];
        let mut dist_full: Vec<Option<usize>> = vec![None; max_index];
        let mut queue: VecDeque<G::NodeId> = VecDeque::new();

        sigma_full[source_idx] = 1.0;
        dist_full[source_idx] = Some(0);
        queue.push_back(source_id);

        while let Some(v) = queue.pop_front() {
            let v_idx = graph.to_index(v);
            let dist_v = dist_full[v_idx].unwrap();
            for w in graph.neighbors(v) {
                let w_idx = graph.to_index(w);
                if dist_full[w_idx].is_none() {
                    dist_full[w_idx] = Some(dist_v + 1);
                    queue.push_back(w);
                }
                if dist_full[w_idx] == Some(dist_v + 1) {
                    sigma_full[w_idx] += sigma_full[v_idx];
                }
            }
        }

        // BFS on graph with group nodes removed
        let mut sigma_no_group = vec![0.0_f64; max_index];
        let mut dist_no_group: Vec<Option<usize>> = vec![None; max_index];
        let mut queue2: VecDeque<G::NodeId> = VecDeque::new();

        sigma_no_group[source_idx] = 1.0;
        dist_no_group[source_idx] = Some(0);
        queue2.push_back(source_id);

        while let Some(v) = queue2.pop_front() {
            let v_idx = graph.to_index(v);
            let dist_v = dist_no_group[v_idx].unwrap();
            for w in graph.neighbors(v) {
                let w_idx = graph.to_index(w);
                if group_set.contains(&w_idx) {
                    continue;
                }
                if dist_no_group[w_idx].is_none() {
                    dist_no_group[w_idx] = Some(dist_v + 1);
                    queue2.push_back(w);
                }
                if dist_no_group[w_idx] == Some(dist_v + 1) {
                    sigma_no_group[w_idx] += sigma_no_group[v_idx];
                }
            }
        }

        // For each non-group target, accumulate the fraction of shortest paths
        // that pass through at least one group member.
        for &target_id in &node_ids {
            let target_idx = graph.to_index(target_id);
            if target_idx == source_idx || group_set.contains(&target_idx) {
                continue;
            }
            if sigma_full[target_idx] == 0.0 {
                continue;
            }

            // Paths through group = total - paths avoiding group,
            // but only if the shortest path length is the same. If it differs,
            // none of the shortest paths avoid the group.
            let paths_avoiding = if dist_no_group[target_idx] == dist_full[target_idx] {
                sigma_no_group[target_idx]
            } else {
                0.0
            };

            let fraction_through_group =
                (sigma_full[target_idx] - paths_avoiding) / sigma_full[target_idx];
            group_betweenness += fraction_through_group;
        }
    }

    if !graph.is_directed() {
        group_betweenness /= 2.0;
    }

    if normalized {
        let non_group = node_count - group_size;
        if non_group > 1 {
            let norm = if graph.is_directed() {
                (non_group * (non_group - 1)) as f64
            } else {
                ((non_group * (non_group - 1)) / 2) as f64
            };
            group_betweenness /= norm;
        }
    }

    group_betweenness
}

#[cfg(test)]
mod test_group_degree_centrality {
    use crate::centrality::group_degree_centrality;
    use crate::petgraph;

    #[test]
    fn test_undirected_path() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ]);
        let result = group_degree_centrality(&g, &[0, 1], None);
        // Neighbors of {0,1} outside group = {2}. Centrality = 1/3.
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_undirected_complete() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ]);
        let result = group_degree_centrality(&g, &[0], None);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_directed_out() {
        let g = petgraph::graph::DiGraph::<(), ()>::from_edges(&[
            (0, 1),
            (1, 2),
            (2, 3),
        ]);
        let result =
            group_degree_centrality(&g, &[0, 1], Some(petgraph::Direction::Outgoing));
        // Out-neighbors of {0,1} outside group = {2}. Centrality = 1/2.
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_directed_in() {
        let g = petgraph::graph::DiGraph::<(), ()>::from_edges(&[
            (0, 1),
            (1, 2),
            (2, 3),
        ]);
        let result =
            group_degree_centrality(&g, &[2, 3], Some(petgraph::Direction::Incoming));
        // In-neighbors of {2,3} outside group = {1}. Centrality = 1/2.
        assert!((result - 0.5).abs() < 1e-10);
    }
}

#[cfg(test)]
mod test_group_closeness_centrality {
    use crate::centrality::group_closeness_centrality;
    use crate::petgraph;

    macro_rules! assert_almost_equal {
        ($x:expr, $y:expr, $d:expr) => {
            if ($x - $y).abs() >= $d {
                panic!("{} != {} within delta of {}", $x, $y, $d);
            }
        };
    }

    #[test]
    fn test_undirected_path() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ]);
        let result = group_closeness_centrality(&g, &[0, 1]);
        // Non-group = {2,3,4}. d(S,2)=1, d(S,3)=2, d(S,4)=3. Sum=6.
        // Closeness = 3/6 = 0.5
        assert_almost_equal!(result, 0.5, 1e-10);
    }

    #[test]
    fn test_undirected_center_node() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
        ]);
        let result = group_closeness_centrality(&g, &[2]);
        // Non-group = {0,1,3,4}. All at distance 1. Sum=4.
        // Closeness = 4/4 = 1.0
        assert_almost_equal!(result, 1.0, 1e-10);
    }

    #[test]
    fn test_disconnected() {
        // Two disconnected components
        let mut g = petgraph::graph::UnGraph::<(), ()>::new_undirected();
        g.add_node(());
        g.add_node(());
        g.add_node(());
        g.add_edge(
            petgraph::graph::NodeIndex::new(0),
            petgraph::graph::NodeIndex::new(1),
            (),
        );
        // Node 2 is disconnected
        let result = group_closeness_centrality(&g, &[0]);
        // |V-S|=2, only node 1 reachable at distance 1. Node 2 unreachable.
        // dist_sum=1. closeness = 2/1 = 2.0
        assert_almost_equal!(result, 2.0, 1e-10);
    }
}

#[cfg(test)]
mod test_group_betweenness_centrality {
    use crate::centrality::group_betweenness_centrality;
    use crate::petgraph;

    macro_rules! assert_almost_equal {
        ($x:expr, $y:expr, $d:expr) => {
            if ($x - $y).abs() >= $d {
                panic!("{} != {} within delta of {}", $x, $y, $d);
            }
        };
    }

    #[test]
    fn test_undirected_path_center() {
        // Path: 0-1-2-3-4
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ]);
        // Group = {2}. Node 2 is on all shortest paths between {0,1} and {3,4}.
        let result = group_betweenness_centrality(&g, &[2], false);
        // Pairs through node 2: (0,3), (0,4), (1,3), (1,4) = 4 paths
        assert_almost_equal!(result, 4.0, 1e-10);
    }

    #[test]
    fn test_undirected_path_center_normalized() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ]);
        let result = group_betweenness_centrality(&g, &[2], true);
        // Non-group size = 4. Normalization = C(4,2) = 6.
        // Normalized = 4/6 = 2/3
        assert_almost_equal!(result, 2.0 / 3.0, 1e-10);
    }

    #[test]
    fn test_empty_group() {
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[(0, 1), (1, 2)]);
        let result = group_betweenness_centrality(&g, &[], false);
        assert_almost_equal!(result, 0.0, 1e-10);
    }

    #[test]
    fn test_single_node_group() {
        // Star graph: center=0, leaves=1,2,3,4
        let g = petgraph::graph::UnGraph::<(), ()>::from_edges(&[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
        ]);
        let result = group_betweenness_centrality(&g, &[0], false);
        // Node 0 is on all 6 shortest paths between leaf pairs
        assert_almost_equal!(result, 6.0, 1e-10);
    }
}
