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
use petgraph::visit::{
    EdgeRef,
    GraphBase,
    GraphProp, // allows is_directed
    IntoEdges,
    IntoNeighbors,
    IntoNeighborsDirected,
    IntoNodeIdentifiers,
    NodeCount,
    NodeIndexable,
};
use rayon::prelude::*;

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
/// * `endpoints` - Whether to include the endpoints of paths in the path
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
pub fn betweenness_centrality<G>(
    graph: G,
    endpoints: bool,
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
    let node_indices: Vec<usize> = graph
        .node_identifiers()
        .map(|i| graph.to_index(i))
        .collect();
    if graph.node_count() < parallel_threshold {
        node_indices
            .iter()
            .map(|node_s| {
                (
                    shortest_path_for_centrality(&graph, &graph.from_index(*node_s)),
                    *node_s,
                )
            })
            .for_each(|(mut shortest_path_calc, is)| {
                if endpoints {
                    _accumulate_endpoints(
                        &locked_betweenness,
                        max_index,
                        &mut shortest_path_calc,
                        is,
                        &graph,
                    );
                } else {
                    _accumulate_basic(
                        &locked_betweenness,
                        max_index,
                        &mut shortest_path_calc,
                        is,
                        &graph,
                    );
                }
            });
    } else {
        node_indices
            .par_iter()
            .map(|node_s| {
                (
                    shortest_path_for_centrality(&graph, &graph.from_index(*node_s)),
                    node_s,
                )
            })
            .for_each(|(mut shortest_path_calc, is)| {
                if endpoints {
                    _accumulate_endpoints(
                        &locked_betweenness,
                        max_index,
                        &mut shortest_path_calc,
                        *is,
                        &graph,
                    );
                } else {
                    _accumulate_basic(
                        &locked_betweenness,
                        max_index,
                        &mut shortest_path_calc,
                        *is,
                        &graph,
                    );
                }
            });
    }
    _rescale(
        &mut betweenness,
        graph.node_count(),
        normalized,
        graph.is_directed(),
        endpoints,
    );

    betweenness
}

fn _rescale(
    betweenness: &mut [Option<f64>],
    node_count: usize,
    normalized: bool,
    directed: bool,
    endpoints: bool,
) {
    let mut do_scale = true;
    let mut scale = 1.0;
    if normalized {
        if endpoints {
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

fn _accumulate_basic<G>(
    locked_betweenness: &RwLock<&mut Vec<Option<f64>>>,
    max_index: usize,
    path_calc: &mut ShortestPathData<G>,
    is: usize,
    graph: G,
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
    for w in &path_calc.verts_sorted_by_distance {
        let iw = graph.to_index(*w);
        if iw != is {
            betweenness[iw] = betweenness[iw].map(|x| x + delta[iw]);
        }
    }
}

fn _accumulate_endpoints<G>(
    locked_betweenness: &RwLock<&mut Vec<Option<f64>>>,
    max_index: usize,
    path_calc: &mut ShortestPathData<G>,
    is: usize,
    graph: G,
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
    betweenness[is] =
        betweenness[is].map(|x| x + ((path_calc.verts_sorted_by_distance.len() - 1) as f64));
    for w in &path_calc.verts_sorted_by_distance {
        let iw = graph.to_index(*w);
        if iw != is {
            betweenness[iw] = betweenness[iw].map(|x| x + delta[iw] + 1.0);
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
    let mut predecessors = HashMap::<G::NodeId, Vec<G::NodeId>>::with_capacity(c);
    let mut sigma = HashMap::<G::NodeId, f64>::with_capacity(c);
    let mut distance = HashMap::<G::NodeId, i64>::with_capacity(c);
    #[allow(non_snake_case)]
    let mut Q: VecDeque<G::NodeId> = VecDeque::with_capacity(c);

    for node in graph.node_identifiers() {
        predecessors.insert(node, Vec::new());
        sigma.insert(node, 0.0);
        distance.insert(node, -1);
    }
    sigma.insert(*node_s, 1.0);
    distance.insert(*node_s, 0);
    Q.push_back(*node_s);
    while let Some(v) = Q.pop_front() {
        verts_sorted_by_distance.push(v);
        let distance_v = distance[&v];
        for w in graph.neighbors(v) {
            if distance[&w] < 0 {
                Q.push_back(w);
                distance.insert(w, distance_v + 1);
            }
            if distance[&w] == distance_v + 1 {
                sigma.insert(w, sigma[&w] + sigma[&v]);
                let e_p = predecessors.get_mut(&w).unwrap();
                e_p.push(v);
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

#[cfg(test)]
mod test_eigenvector_centrality {

    use crate::centrality::eigenvector_centrality;
    use crate::petgraph;
    use crate::Result;

    macro_rules! assert_almost_equal {
        ($x:expr, $y:expr, $d:expr) => {
            if !($x - $y < $d || $y - $x < $d) {
                panic!("{} != {} within delta of {}", $x, $y, $d);
            }
        };
    }
    #[test]
    fn test_no_convergence() {
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[(0, 1), (1, 2)]);
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
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[(0, 1), (1, 2)]);
        let output: Result<Option<Vec<f64>>> = eigenvector_centrality(&g, |_| Ok(1.), None, None);
        let result = output.unwrap().unwrap();
        let expected_values: Vec<f64> = vec![0.5, 0.7071, 0.5];
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
            0.25368793, 0.19576478, 0.32817092, 0.40430835, 0.48199885, 0.15724483, 0.51346196,
            0.32475403,
        ];
        for i in 0..8 {
            assert_almost_equal!(expected_values[i], result[i], 1e-4);
        }
    }
}
