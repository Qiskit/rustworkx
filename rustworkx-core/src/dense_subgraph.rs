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

use hashbrown::{HashMap, HashSet};
use std::hash::Hash;

use petgraph::prelude::*;
use petgraph::visit::{
    EdgeCount, GraphProp, IntoEdgeReferences, IntoNeighbors, IntoNodeIdentifiers, NodeCount,
    Visitable,
};

use rayon::prelude::*;

struct SubsetResult<N> {
    pub count: usize,
    pub error: f64,
    pub map: Vec<N>,
}

/// Find the most densely connected k-subgraph
///
/// This function will return the node indices of the subgraph of `num_nodes` that is the
/// most densely connected.
///
/// This method does not provide any guarantees on the approximation as it
/// does a naive search using BFS traversal.
///
/// # Arguments
///
/// * `graph` - The graph to find densest subgraph in.
/// * `num_nodes` - The number of nodes in the subgraph to find
/// * `edge_weight_callback` - An optional callable that if specified will be
///     passed the node indices of each edge in the graph and it is expected to
///     return a float value. If specified the lowest avg weight for edges in
///     a found subgraph will be a criteria for selection in addition to the
///     connectivity of the subgraph.
/// * `node_weight_callback` - An optional callable that if specified will be
///     passed the node indices of each node in the graph and it is expected to
///     return a float value. If specified the lowest avg weight for node of
///     a found subgraph will be a criteria for selection in addition to the
///     connectivity of the subgraph.
///
/// # Example:
///
/// ```rust
/// use std::convert::Infallible;
/// use rustworkx_core::petgraph::stable_graph::{StableDiGraph, NodeIndex};
/// use rustworkx_core::petgraph::visit::IntoEdgeReferences;
/// use rustworkx_core::generators::grid_graph;
/// use rustworkx_core::dense_subgraph::densest_subgraph;
///
/// type EdgeWeightType = Box<dyn FnMut(<&StableDiGraph<(), ()> as IntoEdgeReferences>::EdgeRef) -> Result<f64, Infallible>>;
/// type NodeWeightType = Box<dyn FnMut(NodeIndex) -> Result<f64, Infallible>>;
///
/// let graph: StableDiGraph<(), ()> = grid_graph(
///     Some(10),
///     Some(10),
///     None,
///     || {()},
///     || {()},
///     false
/// ).unwrap();
/// let subgraph_nodes = densest_subgraph(&graph, 10, None::<EdgeWeightType>, None::<NodeWeightType>).unwrap();
///
/// let expected = vec![
///     NodeIndex::new(7), NodeIndex::new(8), NodeIndex::new(17), NodeIndex::new(9),
///     NodeIndex::new(18), NodeIndex::new(27), NodeIndex::new(19), NodeIndex::new(28),
///     NodeIndex::new(37), NodeIndex::new(29)
/// ];
///
/// assert_eq!(subgraph_nodes, expected);
/// ```
pub fn densest_subgraph<G, H, F, E>(
    graph: G,
    num_nodes: usize,
    edge_weight_callback: Option<H>,
    node_weight_callback: Option<F>,
) -> Result<Vec<G::NodeId>, E>
where
    G: IntoNodeIdentifiers
        + IntoEdgeReferences
        + EdgeCount
        + GraphProp
        + NodeCount
        + IntoNeighbors
        + Visitable
        + Sync,
    G::NodeId: Eq + Hash + Send + Sync,
    F: FnMut(G::NodeId) -> Result<f64, E>,
    H: FnMut(G::EdgeRef) -> Result<f64, E>,
{
    let node_indices: Vec<G::NodeId> = graph.node_identifiers().collect();
    let mut edge_weight_map: Option<HashMap<[G::NodeId; 2], f64>> = None;
    let mut node_weight_map: Option<HashMap<G::NodeId, f64>> = None;

    if edge_weight_callback.is_some() {
        let mut inner_weight_map: HashMap<[G::NodeId; 2], f64> =
            HashMap::with_capacity(graph.edge_count());
        let mut callback = edge_weight_callback.unwrap();
        for edge in graph.edge_references() {
            let source = edge.source();
            let target = edge.target();
            let weight = callback(edge)?;
            inner_weight_map.insert([source, target], weight);
            if !graph.is_directed() {
                inner_weight_map.insert([target, source], weight);
            }
        }
        edge_weight_map = Some(inner_weight_map);
    }
    let mut avg_node_error: f64 = 0.;
    if node_weight_callback.is_some() {
        let mut callback = node_weight_callback.unwrap();
        let mut inner_weight_map: HashMap<G::NodeId, f64> =
            HashMap::with_capacity(graph.node_count());
        for node in graph.node_identifiers() {
            let weight = callback(node)?;
            avg_node_error += weight;
            inner_weight_map.insert(node, weight);
        }
        avg_node_error /= graph.node_count() as f64;
        node_weight_map = Some(inner_weight_map);
    }
    let reduce_identity_fn = || -> SubsetResult<G::NodeId> {
        SubsetResult {
            count: 0,
            map: Vec::new(),
            error: f64::INFINITY,
        }
    };

    let reduce_fn =
        |best: SubsetResult<G::NodeId>, curr: SubsetResult<G::NodeId>| -> SubsetResult<G::NodeId> {
            if edge_weight_map.is_some() || node_weight_map.is_some() {
                if curr.count >= best.count && curr.error <= best.error {
                    curr
                } else {
                    best
                }
            } else if curr.count > best.count {
                curr
            } else {
                best
            }
        };

    let best_result = node_indices
        .into_par_iter()
        .filter_map(|index| {
            let mut subgraph: Vec<[G::NodeId; 2]> = Vec::with_capacity(num_nodes);
            let mut bfs = Bfs::new(&graph, index);
            let mut bfs_vec: Vec<G::NodeId> = Vec::with_capacity(num_nodes);
            let mut bfs_set: HashSet<G::NodeId> = HashSet::with_capacity(num_nodes);

            let mut count = 0;
            while let Some(node) = bfs.next(&graph) {
                bfs_vec.push(node);
                bfs_set.insert(node);
                count += 1;
                if count >= num_nodes {
                    break;
                }
            }
            if bfs_vec.len() < num_nodes {
                return None;
            }
            let mut connection_count = 0;
            for node in &bfs_vec {
                for nbr in graph.neighbors(*node).filter(|j| bfs_set.contains(j)) {
                    connection_count += 1;
                    subgraph.push([*node, nbr]);
                }
            }
            let mut error = match &edge_weight_map {
                Some(map) => {
                    subgraph.iter().map(|edge| map[edge]).sum::<f64>() / subgraph.len() as f64
                }
                None => 0.,
            };
            error *= match &node_weight_map {
                Some(map) => {
                    let subgraph_node_error_avg =
                        bfs_vec.iter().map(|node| map[node]).sum::<f64>() / num_nodes as f64;
                    let node_error_diff = subgraph_node_error_avg - avg_node_error;
                    if node_error_diff > 0. {
                        num_nodes as f64 * node_error_diff
                    } else {
                        1.
                    }
                }
                None => 1.,
            };

            Some(SubsetResult {
                count: connection_count,
                error,
                map: bfs_vec,
            })
        })
        .reduce(reduce_identity_fn, reduce_fn);
    Ok(best_result.map)
}
