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

use std::cmp::{Eq, Ordering};
use std::convert::Infallible;
use std::hash::Hash;

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableGraph};
use petgraph::unionfind::UnionFind;
use petgraph::visit::{
    EdgeCount, EdgeIndexable, EdgeRef, GraphProp, IntoEdgeReferences, IntoEdges,
    IntoNodeIdentifiers, IntoNodeReferences, NodeCount, NodeIndexable, NodeRef, Visitable,
};
use petgraph::Undirected;

use crate::dictmap::*;
use crate::shortest_path::dijkstra;
use crate::utils::pairwise;

type AllPairsDijkstraReturn = HashMap<usize, (DictMap<usize, Vec<usize>>, DictMap<usize, f64>)>;

fn all_pairs_dijkstra_shortest_paths<G, F, E>(
    graph: G,
    mut weight_fn: F,
) -> Result<AllPairsDijkstraReturn, E>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + EdgeCount
        + NodeCount
        + EdgeIndexable
        + Visitable
        + Sync
        + IntoEdges,
    G::NodeId: Eq + Hash + Send,
    G::EdgeId: Eq + Hash + Send,
    F: FnMut(G::EdgeRef) -> Result<f64, E>,
{
    if graph.node_count() == 0 {
        return Ok(HashMap::new());
    } else if graph.edge_count() == 0 {
        return Ok(graph
            .node_identifiers()
            .map(|x| {
                (
                    NodeIndexable::to_index(&graph, x),
                    (DictMap::new(), DictMap::new()),
                )
            })
            .collect());
    }
    let mut edge_weights: Vec<Option<f64>> = vec![None; graph.edge_bound()];
    for edge in graph.edge_references() {
        let index = EdgeIndexable::to_index(&graph, edge.id());
        edge_weights[index] = Some(weight_fn(edge)?);
    }
    let edge_cost = |e: G::EdgeRef| -> Result<f64, Infallible> {
        Ok(edge_weights[EdgeIndexable::to_index(&graph, e.id())].unwrap())
    };

    let node_indices: Vec<usize> = graph
        .node_identifiers()
        .map(|n| NodeIndexable::to_index(&graph, n))
        .collect();
    Ok(node_indices
        .into_par_iter()
        .map(|x| {
            let mut paths: DictMap<G::NodeId, Vec<G::NodeId>> =
                DictMap::with_capacity(graph.node_count());
            let distances: DictMap<G::NodeId, f64> = dijkstra(
                graph,
                NodeIndexable::from_index(&graph, x),
                None,
                edge_cost,
                Some(&mut paths),
            )
            .unwrap();
            (
                x,
                (
                    paths
                        .into_iter()
                        .map(|(k, v)| {
                            (
                                NodeIndexable::to_index(&graph, k),
                                v.into_iter()
                                    .map(|n| NodeIndexable::to_index(&graph, n))
                                    .collect(),
                            )
                        })
                        .collect(),
                    distances
                        .into_iter()
                        .map(|(k, v)| (NodeIndexable::to_index(&graph, k), v))
                        .collect(),
                ),
            )
        })
        .collect())
}

struct MetricClosureEdge {
    source: usize,
    target: usize,
    distance: f64,
    path: Vec<usize>,
}

/// Return the metric closure of a graph
///
/// The metric closure of a graph is the complete graph in which each edge is
/// weighted by the shortest path distance between the nodes in the graph.
///
/// Arguments:
///     `graph`: The input graph to compute the metric closure for
///     `weight_fn`: A callable weight function that will be passed an edge reference
///         for each edge in the graph and it is expected to return a `Result<f64>`
///         which if it doesn't error represents the weight of that edge.
///     `default_weight`: A blind callable that returns a default weight to use for
///         edges added to the output
///
/// Returns a `StableGraph` with the input graph node ids for node weights and edge weights with a
///     tuple of the numeric weight (found via `weight_fn`) and the path. The output will be `None`
///     if `graph` is disconnected.
///
/// # Example
/// ```rust
/// use std::convert::Infallible;
///
/// use rustworkx_core::petgraph::Graph;
/// use rustworkx_core::petgraph::Undirected;
/// use rustworkx_core::petgraph::graph::EdgeReference;
/// use rustworkx_core::petgraph::visit::{IntoEdgeReferences, EdgeRef};
///
/// use rustworkx_core::steiner_tree::metric_closure;
///
/// let input_graph = Graph::<(), u8, Undirected>::from_edges(&[
///     (0, 1, 10),
///     (1, 2, 10),
///     (2, 3, 10),
///     (3, 4, 10),
///     (4, 5, 10),
///     (1, 6, 1),
///     (6, 4, 1),
/// ]);
///
/// let weight_fn = |e: EdgeReference<u8>| -> Result<f64, Infallible> {
///    Ok(*e.weight() as f64)
/// };
///
/// let closure = metric_closure(&input_graph, weight_fn).unwrap().unwrap();
/// let mut output_edge_list: Vec<(usize, usize, (f64, Vec<usize>))> = closure.edge_references().map(|edge| (edge.source().index(), edge.target().index(), edge.weight().clone())).collect();
/// let mut expected_edges: Vec<(usize, usize, (f64, Vec<usize>))> = vec![
///     (0, 1, (10.0, vec![0, 1])),
///     (0, 2, (20.0, vec![0, 1, 2])),
///     (0, 3, (22.0, vec![0, 1, 6, 4, 3])),
///     (0, 4, (12.0, vec![0, 1, 6, 4])),
///     (0, 5, (22.0, vec![0, 1, 6, 4, 5])),
///     (0, 6, (11.0, vec![0, 1, 6])),
///     (1, 2, (10.0, vec![1, 2])),
///     (1, 3, (12.0, vec![1, 6, 4, 3])),
///     (1, 4, (2.0, vec![1, 6, 4])),
///     (1, 5, (12.0, vec![1, 6, 4, 5])),
///     (1, 6, (1.0, vec![1, 6])),
///     (2, 3, (10.0, vec![2, 3])),
///     (2, 4, (12.0, vec![2, 1, 6, 4])),
///     (2, 5, (22.0, vec![2, 1, 6, 4, 5])),
///     (2, 6, (11.0, vec![2, 1, 6])),
///     (3, 4, (10.0, vec![3, 4])),
///     (3, 5, (20.0, vec![3, 4, 5])),
///     (3, 6, (11.0, vec![3, 4, 6])),
///     (4, 5, (10.0, vec![4, 5])),
///     (4, 6, (1.0, vec![4, 6])),
///     (5, 6, (11.0, vec![5, 4, 6])),
/// ];
/// output_edge_list.sort_by_key(|x| [x.0, x.1]);
/// expected_edges.sort_by_key(|x| [x.0, x.1]);
/// assert_eq!(output_edge_list, expected_edges);
///
/// ```
#[allow(clippy::type_complexity)]
pub fn metric_closure<G, F, E>(
    graph: G,
    weight_fn: F,
) -> Result<Option<StableGraph<G::NodeId, (f64, Vec<usize>), Undirected>>, E>
where
    G: NodeIndexable
        + EdgeIndexable
        + Sync
        + EdgeCount
        + NodeCount
        + Visitable
        + IntoNodeReferences
        + IntoEdges
        + Visitable
        + GraphProp<EdgeType = Undirected>,
    G::NodeId: Eq + Hash + NodeRef + Send,
    G::EdgeId: Eq + Hash + Send,
    G::NodeWeight: Clone,
    F: FnMut(G::EdgeRef) -> Result<f64, E>,
{
    let mut out_graph: StableGraph<G::NodeId, (f64, Vec<usize>), Undirected> =
        StableGraph::with_capacity(graph.node_count(), graph.edge_count());
    let node_map: HashMap<usize, NodeIndex> = graph
        .node_references()
        .map(|node| {
            (
                NodeIndexable::to_index(&graph, node.id()),
                out_graph.add_node(node.id()),
            )
        })
        .collect();
    let edges = metric_closure_edges(graph, weight_fn)?;
    if edges.is_none() {
        return Ok(None);
    }
    for edge in edges.unwrap() {
        out_graph.add_edge(
            node_map[&edge.source],
            node_map[&edge.target],
            (edge.distance, edge.path),
        );
    }
    Ok(Some(out_graph))
}

fn metric_closure_edges<G, F, E>(
    graph: G,
    weight_fn: F,
) -> Result<Option<Vec<MetricClosureEdge>>, E>
where
    G: NodeIndexable
        + Sync
        + Visitable
        + IntoNodeReferences
        + IntoEdges
        + Visitable
        + NodeIndexable
        + NodeCount
        + EdgeCount
        + EdgeIndexable,
    G::NodeId: Eq + Hash + Send,
    G::EdgeId: Eq + Hash + Send,
    F: FnMut(G::EdgeRef) -> Result<f64, E>,
{
    let node_count = graph.node_count();
    if node_count == 0 {
        return Ok(Some(Vec::new()));
    }
    let mut out_vec = Vec::with_capacity(node_count * (node_count - 1) / 2);
    let paths = all_pairs_dijkstra_shortest_paths(graph, weight_fn)?;
    let mut nodes: HashSet<usize> = graph
        .node_identifiers()
        .map(|x| NodeIndexable::to_index(&graph, x))
        .collect();
    let first_node = graph
        .node_identifiers()
        .map(|x| NodeIndexable::to_index(&graph, x))
        .next()
        .unwrap();
    let path_keys: HashSet<usize> = paths[&first_node].0.keys().copied().collect();
    // first_node will always be missing from path_keys so if the difference
    // is > 1 with nodes that means there is another node in the graph that
    // first_node doesn't have a path to.
    if nodes.difference(&path_keys).count() > 1 {
        return Ok(None);
    }
    // Iterate over node indices for a deterministic order
    for node in graph
        .node_identifiers()
        .map(|x| NodeIndexable::to_index(&graph, x))
    {
        let path_map = &paths[&node].0;
        nodes.remove(&node);
        let distance = &paths[&node].1;
        for v in &nodes {
            out_vec.push(MetricClosureEdge {
                source: node,
                target: *v,
                distance: distance[v],
                path: path_map[v].clone(),
            });
        }
    }
    Ok(Some(out_vec))
}

/// Computes the shortest path between all pairs `(s, t)` of the given `terminal_nodes`
/// *provided* that:
///   - there is an edge `(u, v)` in the graph and path pass through this edge.
///   - node `s` is the closest node to  `u` among all `terminal_nodes`
///   - node `t` is the closest node to `v` among all `terminal_nodes`
/// and wraps the result inside a `MetricClosureEdge`
///
/// For example, if all vertices are terminals, it returns the original edges of the graph.
fn fast_metric_edges<G, F, E>(
    in_graph: G,
    terminal_nodes: &[G::NodeId],
    mut weight_fn: F,
) -> Result<Vec<MetricClosureEdge>, E>
where
    G: IntoEdges
        + NodeIndexable
        + EdgeIndexable
        + Sync
        + EdgeCount
        + Visitable
        + IntoNodeReferences
        + NodeCount,
    G::NodeId: Eq + Hash + Send,
    G::EdgeId: Eq + Hash + Send,
    F: FnMut(G::EdgeRef) -> Result<f64, E>,
{
    let mut graph: StableGraph<(), (), Undirected> = StableGraph::with_capacity(
        in_graph.node_count() + 1,
        in_graph.edge_count() + terminal_nodes.len(),
    );
    let node_map: HashMap<G::NodeId, NodeIndex> = in_graph
        .node_references()
        .map(|n| (n.id(), graph.add_node(())))
        .collect();
    let reverse_node_map: HashMap<NodeIndex, G::NodeId> =
        node_map.iter().map(|(k, v)| (*v, *k)).collect();
    let edge_map: HashMap<EdgeIndex, G::EdgeRef> = in_graph
        .edge_references()
        .map(|e| {
            (
                graph.add_edge(node_map[&e.source()], node_map[&e.target()], ()),
                e,
            )
        })
        .collect();

    // temporarily add a ``dummy`` node, connect it with
    // all the terminal nodes and find all the shortest paths
    // starting from ``dummy`` node.
    let dummy = graph.add_node(());
    for node in terminal_nodes {
        graph.add_edge(dummy, node_map[node], ());
    }

    let mut paths = DictMap::with_capacity(graph.node_count());

    let mut wrapped_weight_fn =
        |e: <&StableGraph<(), ()> as IntoEdgeReferences>::EdgeRef| -> Result<f64, E> {
            if let Some(edge_ref) = edge_map.get(&e.id()) {
                weight_fn(*edge_ref)
            } else {
                Ok(0.0)
            }
        };

    let mut distance: DictMap<NodeIndex, f64> = dijkstra(
        &graph,
        dummy,
        None,
        &mut wrapped_weight_fn,
        Some(&mut paths),
    )?;
    paths.swap_remove(&dummy);
    distance.swap_remove(&dummy);

    // ``partition[u]`` holds the terminal node closest to node ``u``.
    let mut partition: Vec<usize> = vec![usize::MAX; graph.node_bound()];
    for (u, path) in paths.iter() {
        let u = NodeIndexable::to_index(&in_graph, reverse_node_map[u]);
        partition[u] = NodeIndexable::to_index(&in_graph, reverse_node_map[&path[1]]);
    }

    let mut out_edges: Vec<MetricClosureEdge> = Vec::with_capacity(graph.edge_count());

    for edge in graph.edge_references() {
        let source = edge.source();
        let target = edge.target();
        // assert that ``source`` is reachable from a terminal node.
        if distance.contains_key(&source) {
            let weight = distance[&source] + wrapped_weight_fn(edge)? + distance[&target];
            let mut path: Vec<usize> = paths[&source]
                .iter()
                .skip(1)
                .map(|x| NodeIndexable::to_index(&in_graph, reverse_node_map[x]))
                .collect();
            path.append(
                &mut paths[&target]
                    .iter()
                    .skip(1)
                    .rev()
                    .map(|x| NodeIndexable::to_index(&in_graph, reverse_node_map[x]))
                    .collect(),
            );

            let source = NodeIndexable::to_index(&in_graph, reverse_node_map[&source]);
            let target = NodeIndexable::to_index(&in_graph, reverse_node_map[&target]);

            let mut source = partition[source];
            let mut target = partition[target];

            match source.cmp(&target) {
                Ordering::Equal => continue,
                Ordering::Greater => std::mem::swap(&mut source, &mut target),
                _ => {}
            }

            out_edges.push(MetricClosureEdge {
                source,
                target,
                distance: weight,
                path,
            });
        }
    }

    // if parallel edges, keep the edge with minimum distance.
    out_edges.par_sort_unstable_by(|a, b| {
        let weight_a = (a.source, a.target, a.distance);
        let weight_b = (b.source, b.target, b.distance);
        weight_a.partial_cmp(&weight_b).unwrap_or(Ordering::Less)
    });

    out_edges.dedup_by(|edge_a, edge_b| {
        edge_a.source == edge_b.source && edge_a.target == edge_b.target
    });

    Ok(out_edges)
}

pub struct SteinerTreeResult {
    pub used_node_indices: HashSet<usize>,
    pub used_edge_endpoints: HashSet<(usize, usize)>,
}

/// Return an approximation to the minimum Steiner tree of a graph.
///
/// The minimum tree of ``graph`` with regard to a set of ``terminal_nodes``
/// is a tree within ``graph`` that spans those nodes and has a minimum size
/// (measured as the sum of edge weights) amoung all such trees.
///
/// The minimum steiner tree can be approximated by computing the minimum
/// spanning tree of the subgraph of the metric closure of ``graph`` induced
/// by the terminal nodes, where the metric closure of ``graph`` is the
/// complete graph in which each edge is weighted by the shortest path distance
/// between nodes in ``graph``.
///
/// This algorithm [1]_ produces a tree whose weight is within a
/// :math:`(2 - (2 / t))` factor of the weight of the optimal Steiner tree
/// where :math:`t` is the number of terminal nodes. The algorithm implemented
/// here is due to [2]_ . It avoids computing all pairs shortest paths but rather
/// reduces the problem to a single source shortest path and a minimum spanning tree
/// problem.
///
/// Arguments:
///     `graph`: The input graph to compute the steiner tree of
///     `terminal_nodes`: The terminal nodes of the steiner tree
///     `weight_fn`: A callable weight function that will be passed an edge reference
///         for each edge in the graph and it is expected to return a `Result<f64>`
///         which if it doesn't error represents the weight of that edge.
///
/// Returns a custom struct that contains a set of nodes and edges and `None`
/// if the graph is disconnected relative to the terminal nodes.
///
/// # Example
///
/// ```rust
/// use std::convert::Infallible;
///
/// use rustworkx_core::petgraph::Graph;
/// use rustworkx_core::petgraph::graph::NodeIndex;
/// use rustworkx_core::petgraph::Undirected;
/// use rustworkx_core::petgraph::graph::EdgeReference;
/// use rustworkx_core::petgraph::visit::{IntoEdgeReferences, EdgeRef};
///
/// use rustworkx_core::steiner_tree::steiner_tree;
///
/// let input_graph = Graph::<(), u8, Undirected>::from_edges(&[
///     (0, 1, 10),
///     (1, 2, 10),
///     (2, 3, 10),
///     (3, 4, 10),
///     (4, 5, 10),
///     (1, 6, 1),
///     (6, 4, 1),
/// ]);
///
/// let weight_fn = |e: EdgeReference<u8>| -> Result<f64, Infallible> {
///    Ok(*e.weight() as f64)
/// };
/// let terminal_nodes = vec![
///     NodeIndex::new(0),
///     NodeIndex::new(1),
///     NodeIndex::new(2),
///     NodeIndex::new(3),
///     NodeIndex::new(4),
///     NodeIndex::new(5),
/// ];
///
/// let tree = steiner_tree(&input_graph, &terminal_nodes, weight_fn).unwrap().unwrap();
/// ```
///
/// .. [1] Kou, Markowsky & Berman,
///    "A fast algorithm for Steiner trees"
///    Acta Informatica 15, 141–145 (1981).
///    https://link.springer.com/article/10.1007/BF00288961
/// .. [2] Kurt Mehlhorn,
///    "A faster approximation algorithm for the Steiner problem in graphs"
///    https://doi.org/10.1016/0020-0190(88)90066-X
pub fn steiner_tree<G, F, E>(
    graph: G,
    terminal_nodes: &[G::NodeId],
    weight_fn: F,
) -> Result<Option<SteinerTreeResult>, E>
where
    G: IntoEdges
        + NodeIndexable
        + Sync
        + EdgeCount
        + IntoNodeReferences
        + EdgeIndexable
        + Visitable
        + NodeCount,
    G::NodeId: Eq + Hash + Send,
    G::EdgeId: Eq + Hash + Send,
    F: FnMut(G::EdgeRef) -> Result<f64, E>,
{
    let node_bound = graph.node_bound();
    let mut edge_list = fast_metric_edges(graph, terminal_nodes, weight_fn)?;
    let mut subgraphs = UnionFind::<usize>::new(node_bound);
    edge_list.par_sort_unstable_by(|a, b| {
        let weight_a = (a.distance, a.source, a.target);
        let weight_b = (b.distance, b.source, b.target);
        weight_a.partial_cmp(&weight_b).unwrap_or(Ordering::Less)
    });
    let mut mst_edges: Vec<MetricClosureEdge> = Vec::new();
    for float_edge_pair in edge_list {
        let u = float_edge_pair.source;
        let v = float_edge_pair.target;
        if subgraphs.union(u, v) {
            mst_edges.push(float_edge_pair);
        }
    }
    // assert that the terminal nodes are connected.
    if !terminal_nodes.is_empty() && mst_edges.len() != terminal_nodes.len() - 1 {
        return Ok(None);
    }
    // Generate the output graph from the MST
    let out_edge_list: Vec<[usize; 2]> = mst_edges
        .into_iter()
        .flat_map(|edge| pairwise(edge.path))
        .filter_map(|x| x.0.map(|a| [a, x.1]))
        .collect();
    let out_edges: HashSet<(usize, usize)> = out_edge_list.iter().map(|x| (x[0], x[1])).collect();
    let out_nodes: HashSet<usize> = out_edge_list
        .iter()
        .flat_map(|x| x.iter())
        .copied()
        .collect();
    Ok(Some(SteinerTreeResult {
        used_node_indices: out_nodes,
        used_edge_endpoints: out_edges,
    }))
}
