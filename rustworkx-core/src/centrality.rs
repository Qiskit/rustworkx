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

use hashbrown::HashMap;
use petgraph::algo::dijkstra;
use petgraph::visit::{
    EdgeCount,
    EdgeIndexable,
    EdgeRef,
    GraphBase,
    GraphProp, // allows is_directed
    IntoEdges,
    IntoEdgesDirected,
    IntoNeighbors,
    IntoNeighborsDirected,
    IntoNodeIdentifiers,
    NodeCount,
    NodeIndexable,
    Reversed,
    Visitable,
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
///     lengths used to compute the betweenness
/// * `normalized` - Whether to normalize the betweenness scores by the number
///     of distinct paths between all pairs of nodes
/// * `parallel_threshold` - The number of nodes to calculate the betweenness
///     centrality in parallel at, if the number of nodes in `graph` is less
///     than this value it will run in a single thread. A good default to use
///     here if you're not sure is `50` as that was found to be roughly the
///     number of nodes where parallelism improves performance
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
    <G as GraphBase>::NodeId: std::cmp::Eq + Hash + Send,
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
///     of distinct paths between all pairs of nodes
/// * `parallel_threshold` - The number of nodes to calculate the betweenness
///     centrality in parallel at, if the number of nodes in `graph` is less
///     than this value it will run in a single thread. A good default to use
///     here if you're not sure is `50` as that was found to be roughly the
///     number of nodes where parallelism improves performance
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
    G::NodeId: Eq + Hash + Send,
    G::EdgeId: Eq + Hash + Send,
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
    <G as GraphBase>::NodeId: std::cmp::Eq + Hash,
{
    let mut delta = vec![0.0; max_index];
    for w in &path_calc.verts_sorted_by_distance {
        let iw = graph.to_index(*w);
        let coeff = (1.0 + delta[iw]) / path_calc.sigma[w];
        let p_w = path_calc.predecessors.get(w).unwrap();
        for v in p_w {
            let iv = graph.to_index(*v);
            delta[iv] += path_calc.sigma[v] * coeff;
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
    G::NodeId: Eq + Hash,
    G::EdgeId: Eq + Hash,
{
    let mut delta = vec![0.0; max_index];
    for w in &path_calc.verts_sorted_by_distance {
        let iw = NodeIndexable::to_index(&graph, *w);
        let coeff = (1.0 + delta[iw]) / path_calc.sigma[w];
        let p_w = path_calc.predecessors.get(w).unwrap();
        let e_w = path_calc.predecessor_edges.get(w).unwrap();
        let mut betweenness = locked_betweenness.write().unwrap();
        for i in 0..p_w.len() {
            let v = p_w[i];
            let iv = NodeIndexable::to_index(&graph, v);
            let ie = EdgeIndexable::to_index(&graph, e_w[i]);
            let c = path_calc.sigma[&v] * coeff;
            betweenness[ie] = betweenness[ie].map(|x| x + c);
            delta[iv] += c;
        }
    }
}

struct ShortestPathData<G>
where
    G: GraphBase,
    <G as GraphBase>::NodeId: std::cmp::Eq + Hash,
{
    verts_sorted_by_distance: Vec<G::NodeId>,
    predecessors: HashMap<G::NodeId, Vec<G::NodeId>>,
    sigma: HashMap<G::NodeId, f64>,
}

fn shortest_path_for_centrality<G>(graph: G, node_s: &G::NodeId) -> ShortestPathData<G>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoNeighborsDirected + NodeCount + GraphBase,
    <G as GraphBase>::NodeId: std::cmp::Eq + Hash,
{
    let mut verts_sorted_by_distance: Vec<G::NodeId> = Vec::new(); // a stack
    let c = graph.node_count();
    let mut predecessors: Vec<Vec<G::NodeId>> = vec![Vec::new(); c];
    let mut sigma: Vec<f64> = vec![0.; c];
    let mut distance: Vec<i64> = vec![-1; c];
    #[allow(non_snake_case)]
    let mut Q: VecDeque<G::NodeId> = VecDeque::with_capacity(c);
    let node_s_index = graph.to_index(*node_s);
    sigma[node_s_index] = 1.0;
    distance[node_s_index] = 0;
    Q.push_back(*node_s);
    while let Some(v) = Q.pop_front() {
        verts_sorted_by_distance.push(v);
        let v_idx = graph.to_index(v);
        let distance_v = distance[v_idx];
        for w in graph.neighbors(v) {
            let w_idx = graph.to_index(w);
            if distance[w_idx] < 0 {
                Q.push_back(w);
                distance[w_idx] = distance_v + 1;
            }
            if distance[w_idx] == distance_v + 1 {
                sigma[w_idx] += sigma[v_idx];
                predecessors[w_idx].push(v);
            }
        }
    }
    let mut sigma_h = HashMap::with_capacity(c);
    for (idx, s) in sigma.iter().enumerate() {
        sigma_h.insert(graph.from_index(idx), *s);
    }
    let mut pred_h = HashMap::with_capacity(c);
    let mut idx = predecessors.len();

    while let Some(p) = predecessors.pop() {
        idx -= 1;
        pred_h.insert(graph.from_index(idx), p);
    }
    let sigma = sigma_h;
    let predecessors = pred_h;
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
    G::NodeId: Eq + Hash,
    G::EdgeId: Eq + Hash,
{
    verts_sorted_by_distance: Vec<G::NodeId>,
    predecessors: HashMap<G::NodeId, Vec<G::NodeId>>,
    predecessor_edges: HashMap<G::NodeId, Vec<G::EdgeId>>,
    sigma: HashMap<G::NodeId, f64>,
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
    G::NodeId: Eq + Hash,
    G::EdgeId: Eq + Hash,
{
    let mut verts_sorted_by_distance: Vec<G::NodeId> = Vec::new(); // a stack
    let c = graph.node_count();
    let mut predecessors = HashMap::<G::NodeId, Vec<G::NodeId>>::with_capacity(c);
    let mut predecessor_edges = HashMap::<G::NodeId, Vec<G::EdgeId>>::with_capacity(c);
    let mut sigma = HashMap::<G::NodeId, f64>::with_capacity(c);
    let mut distance = HashMap::<G::NodeId, i64>::with_capacity(c);
    #[allow(non_snake_case)]
    let mut Q: VecDeque<G::NodeId> = VecDeque::with_capacity(c);

    for node in graph.node_identifiers() {
        predecessors.insert(node, Vec::new());
        predecessor_edges.insert(node, Vec::new());
        sigma.insert(node, 0.0);
        distance.insert(node, -1);
    }
    sigma.insert(*node_s, 1.0);
    distance.insert(*node_s, 0);
    Q.push_back(*node_s);
    while let Some(v) = Q.pop_front() {
        verts_sorted_by_distance.push(v);
        let distance_v = distance[&v];
        for edge in graph.edges(v) {
            let w = edge.target();
            if distance[&w] < 0 {
                Q.push_back(w);
                distance.insert(w, distance_v + 1);
            }
            if distance[&w] == distance_v + 1 {
                sigma.insert(w, sigma[&w] + sigma[&v]);
                let e_p = predecessors.get_mut(&w).unwrap();
                e_p.push(v);
                predecessor_edges.get_mut(&w).unwrap().push(edge.id());
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
    use petgraph::graph::edge_index;
    use petgraph::prelude::StableGraph;
    use petgraph::Undirected;

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
///     an edge in the graph and is expected to return a `Result<f64>` of
///     the weight of that edge.
/// * `max_iter` - The maximum number of iterations in the power method. If
///     set to `None` a default value of 100 is used.
/// * `tol` - The error tolerance used when checking for convergence in the
///     power method. If set to `None` a default value of 1e-6 is used.
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
    G::NodeId: Eq + std::hash::Hash,
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
///     an edge in the graph and is expected to return a `Result<f64>` of
///     the weight of that edge.
/// * `alpha` - Attenuation factor. If set to `None`, a default value of 0.1 is used.
/// * `beta_map` - Immediate neighbourhood weights. Must contain all node indices or be `None`.
/// * `beta_scalar` - Immediate neighbourhood scalar that replaces `beta_map` in case `beta_map` is None.
///     Defaults to 1.0 in case `None` is provided.
/// * `max_iter` - The maximum number of iterations in the power method. If
///     set to `None` a default value of 100 is used.
/// * `tol` - The error tolerance used when checking for convergence in the
///     power method. If set to `None` a default value of 1e-6 is used.
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
    G::NodeId: Eq + std::hash::Hash,
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

    use crate::centrality::eigenvector_centrality;
    use crate::petgraph;
    use crate::Result;

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

    use crate::centrality::katz_centrality;
    use crate::petgraph;
    use crate::Result;
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
/// the average distance" [^WF]. You can enable this by setting `wf_improved` to `true`.
///
/// [^WF] Wasserman, S., & Faust, K. (1994). Social Network Analysis:
///     Methods and Applications (Structural Analysis in the Social Sciences).
///     Cambridge: Cambridge University Press. doi:10.1017/CBO9780511815478
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `wf_improved` - If `true`, scale by the fraction of nodes reachable.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::centrality::closeness_centrality;
///
/// // Calculate the closeness centrality of Graph
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 4), (1, 2), (2, 3), (3, 4), (1, 4)
/// ]);
/// let output = closeness_centrality(&g, true);
/// assert_eq!(
///     vec![Some(1./2.), Some(2./3.), Some(4./7.), Some(2./3.), Some(4./5.)],
///     output
/// );
///
/// // Calculate the closeness centrality of DiGraph
/// let dg = petgraph::graph::DiGraph::<i32, ()>::from_edges(&[
///     (0, 4), (1, 2), (2, 3), (3, 4), (1, 4)
/// ]);
/// let output = closeness_centrality(&dg, true);
/// assert_eq!(
///     vec![Some(0.), Some(0.), Some(1./4.), Some(1./3.), Some(4./5.)],
///     output
/// );
/// ```
pub fn closeness_centrality<G>(graph: G, wf_improved: bool) -> Vec<Option<f64>>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + GraphBase
        + IntoEdges
        + Visitable
        + NodeCount
        + IntoEdgesDirected,
    G::NodeId: std::hash::Hash + Eq,
{
    let max_index = graph.node_bound();
    let mut closeness: Vec<Option<f64>> = vec![None; max_index];
    for node_s in graph.node_identifiers() {
        let is = graph.to_index(node_s);
        let map = dijkstra(Reversed(&graph), node_s, None, |_| 1);
        let reachable_nodes_count = map.len();
        let dists_sum: usize = map.into_values().sum();
        if reachable_nodes_count == 1 {
            closeness[is] = Some(0.0);
            continue;
        }
        closeness[is] = Some((reachable_nodes_count - 1) as f64 / dists_sum as f64);
        if wf_improved {
            let node_count = graph.node_count();
            closeness[is] = closeness[is]
                .map(|c| c * (reachable_nodes_count - 1) as f64 / (node_count - 1) as f64);
        }
    }
    closeness
}
