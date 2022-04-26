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
use std::sync::RwLock;

use hashbrown::HashMap;
use petgraph::algo::dijkstra;
use petgraph::graph::NodeIndex;
use petgraph::visit::{
    GraphBase, GraphProp, IntoEdges, IntoEdgesDirected, IntoNeighborsDirected, IntoNodeIdentifiers,
    NodeCount, NodeIndexable, Reversed, Visitable,
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
/// use retworkx_core::petgraph;
/// use retworkx_core::centrality::betweenness_centrality;
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 4), (1, 2), (2, 3), (3, 4), (1, 4)
/// ]);
/// // Calculate the betweeness centrality
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
        + GraphBase<NodeId = NodeIndex>
        + std::marker::Sync,
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
    let node_indices: Vec<NodeIndex> = graph.node_identifiers().collect();
    if graph.node_count() < parallel_threshold {
        node_indices
            .iter()
            .map(|node_s| {
                (
                    shortest_path_for_centrality(&graph, node_s),
                    graph.to_index(*node_s),
                )
            })
            .for_each(|(mut shortest_path_calc, is)| {
                if endpoints {
                    _accumulate_endpoints(
                        &locked_betweenness,
                        max_index,
                        &mut shortest_path_calc,
                        is,
                    );
                } else {
                    _accumulate_basic(&locked_betweenness, max_index, &mut shortest_path_calc, is);
                }
            });
    } else {
        node_indices
            .par_iter()
            .map(|node_s| {
                (
                    shortest_path_for_centrality(&graph, node_s),
                    graph.to_index(*node_s),
                )
            })
            .for_each(|(mut shortest_path_calc, is)| {
                if endpoints {
                    _accumulate_endpoints(
                        &locked_betweenness,
                        max_index,
                        &mut shortest_path_calc,
                        is,
                    );
                } else {
                    _accumulate_basic(&locked_betweenness, max_index, &mut shortest_path_calc, is);
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

fn _accumulate_basic(
    locked_betweenness: &RwLock<&mut Vec<Option<f64>>>,
    max_index: usize,
    path_calc: &mut ShortestPathData,
    is: usize,
) {
    let mut delta = vec![0.0; max_index];
    for w in &path_calc.verts_sorted_by_distance {
        let iw = w.index();
        let coeff = (1.0 + delta[iw]) / path_calc.sigma[w];
        let p_w = path_calc.predecessors.get(w).unwrap();
        for v in p_w {
            let iv = (*v).index();
            delta[iv] += path_calc.sigma[v] * coeff;
        }
    }
    let mut betweenness = locked_betweenness.write().unwrap();
    for w in &path_calc.verts_sorted_by_distance {
        let iw = w.index();
        if iw != is {
            betweenness[iw] = betweenness[iw].map(|x| x + delta[iw]);
        }
    }
}

fn _accumulate_endpoints(
    locked_betweenness: &RwLock<&mut Vec<Option<f64>>>,
    max_index: usize,
    path_calc: &mut ShortestPathData,
    is: usize,
) {
    let mut delta = vec![0.0; max_index];
    for w in &path_calc.verts_sorted_by_distance {
        let iw = w.index();
        let coeff = (1.0 + delta[iw]) / path_calc.sigma[w];
        let p_w = path_calc.predecessors.get(w).unwrap();
        for v in p_w {
            let iv = (*v).index();
            delta[iv] += path_calc.sigma[v] * coeff;
        }
    }
    let mut betweenness = locked_betweenness.write().unwrap();
    betweenness[is] =
        betweenness[is].map(|x| x + ((path_calc.verts_sorted_by_distance.len() - 1) as f64));
    for w in &path_calc.verts_sorted_by_distance {
        let iw = w.index();
        if iw != is {
            betweenness[iw] = betweenness[iw].map(|x| x + delta[iw] + 1.0);
        }
    }
}

struct ShortestPathData {
    verts_sorted_by_distance: Vec<NodeIndex>,
    predecessors: HashMap<NodeIndex, Vec<NodeIndex>>,
    sigma: HashMap<NodeIndex, f64>,
}

fn shortest_path_for_centrality<G>(graph: G, node_s: &G::NodeId) -> ShortestPathData
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + NodeCount
        + GraphBase<NodeId = NodeIndex>, // for get() and get_mut()
{
    let mut verts_sorted_by_distance: Vec<NodeIndex> = Vec::new(); // a stack
    let c = graph.node_count();
    let mut predecessors = HashMap::<G::NodeId, Vec<G::NodeId>>::with_capacity(c);
    let mut sigma = HashMap::<G::NodeId, f64>::with_capacity(c);
    let mut distance = HashMap::<G::NodeId, i64>::with_capacity(c);
    #[allow(non_snake_case)]
    let mut Q: VecDeque<NodeIndex> = VecDeque::with_capacity(c);

    let i_s = graph.to_index(*node_s);
    let index_s = NodeIndex::new(i_s);

    for node in graph.node_identifiers() {
        predecessors.insert(node, Vec::new());
        sigma.insert(node, 0.0);
        distance.insert(node, -1);
    }
    sigma.insert(index_s, 1.0);
    distance.insert(index_s, 0);
    Q.push_back(index_s);
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

/// Compute the closeness centrality of all nodes in a graph.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `wf_improved` - If True, scale by the fraction of nodes reachable.
///
/// # Example
/// ```rust
/// use retworkx_core::petgraph;
/// use retworkx_core::centrality::closeness_centrality;
///
/// // Calculate the betweeness centrality of Graph
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 4), (1, 2), (2, 3), (3, 4), (1, 4)
/// ]);
/// let output = closeness_centrality(&g, true);
/// assert_eq!(
///     vec![Some(1./2.), Some(2./3.), Some(4./7.), Some(2./3.), Some(4./5.)],
///     output
/// );
///
/// // Calculate the betweeness centrality of DiGraph
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
        + GraphBase<NodeId = NodeIndex>
        + IntoEdges
        + Visitable
        + NodeCount
        + IntoEdgesDirected,
{
    let max_index = graph.node_bound();
    let mut betweenness: Vec<Option<f64>> = vec![None; max_index];
    for node_s in graph.node_identifiers() {
        let is = graph.to_index(node_s);
        let map = dijkstra(Reversed(&graph), node_s, None, |_| 1);
        let mut reachable_nodes_count = 0;
        let mut dists_sum = 0;
        for (_, &value) in map.iter() {
            reachable_nodes_count += 1;
            dists_sum += value;
        }
        if reachable_nodes_count == 1 {
            betweenness[is] = Some(0.0);
            continue;
        }
        betweenness[is] = Some((reachable_nodes_count - 1) as f64 / dists_sum as f64);
        if wf_improved {
            let node_count = graph.node_count();
            betweenness[is] = betweenness[is]
                .map(|c| c * (reachable_nodes_count - 1) as f64 / (node_count - 1) as f64);
        }
    }
    betweenness
}
