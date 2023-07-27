use hashbrown::{HashMap, HashSet};
use num_traits::Float;
use petgraph::stable_graph::{EdgeReference, NodeIndex, StableGraph};
use petgraph::visit::NodeIndexable;
use petgraph::visit::{EdgeRef, GraphBase, IntoEdgeReferences};
use petgraph::Directed;
use rayon::prelude::ParallelSliceMut;
use std::cmp::Ordering;

use crate::dictmap::{DictMap, InitWithHasher};
use crate::petgraph::unionfind::UnionFind;
use crate::shortest_path::dijkstra;
use crate::utils::pairwise;

pub struct MetricClosureEdge<W> {
    source: usize,
    target: usize,
    distance: W,
    path: Vec<usize>,
}

/// Return the metric closure of a graph
///
/// The metric closure of a graph is the complete graph in which each edge is
/// weighted by the shortest path distance between the nodes in the graph.
///
/// :param PyGraph graph: The input graph to find the metric closure for
/// :param weight_fn: A callable object that will be passed an edge's
///     weight/data payload and expected to return a ``float``. For example,
///     you can use ``weight_fn=float`` to cast every weight as a float
///
/// :return: A metric closure graph from the input graph
/// :rtype: PyGraph
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
pub fn metric_closure<F, E, W>(
    graph: &StableGraph<(), W, Directed>,
    weight_fn: &mut F,
) -> Result<StableGraph<(), W, Directed>, E>
where
    W: Clone,
{
    let mut out_graph: StableGraph<(), W, Directed> = graph.clone();
    out_graph.clear_edges();
    let edges = _metric_closure_edges(graph, weight_fn)?;
    for edge in edges {
        out_graph.add_edge(
            NodeIndex::new(edge.source),
            NodeIndex::new(edge.target),
            edge.distance,
        );
    }
    Ok(out_graph)
}

fn _metric_closure_edges<F, E, W>(
    graph: &StableGraph<(), W, Directed>,
    weight_fn: &mut F,
) -> Result<Vec<MetricClosureEdge<W>>, E> {
    let node_count = graph.node_count();
    if node_count == 0 {
        return Ok(Vec::new());
    }

    // TODO implemented
    panic!("not implemented");
}

/// Computes the shortest path between all pairs `(s, t)` of the given `terminal_nodes`
/// *provided* that:
///   - there is an edge `(u, v)` in the graph and path pass through this edge.
///   - node `s` is the closest node to  `u` among all `terminal_nodes`
///   - node `t` is the closest node to `v` among all `terminal_nodes`
/// and wraps the result inside a `MetricClosureEdge`
///
/// For example, if all vertices are terminals, it returns the original edges of the graph.
fn fast_metric_edges<F, E, W>(
    graph: &mut StableGraph<(), W, Directed>,
    terminal_nodes: Vec<usize>,
    weight_fn: &mut F,
) -> Result<Vec<MetricClosureEdge<W>>, E>
where
    W: Clone
        + std::ops::Add<Output = W>
        + std::default::Default
        + std::marker::Copy
        + std::cmp::PartialOrd
        + std::fmt::Debug,
    F: FnMut(&W) -> Result<f64, E>,
{
    // temporarily add a ``dummy`` node, connect it with
    // all the terminal nodes and find all the shortest paths
    // starting from ``dummy`` node.
    let dummy = graph.add_node(());
    for node in terminal_nodes {
        graph.add_edge(dummy, NodeIndex::new(node), None);
    }
    let cost_fn = |edge: EdgeReference<'_, W>| -> Result<W, E> {
        if edge.source() != dummy && edge.target() != dummy {
            let weight: f64 = weight_fn(edge.weight())?;
            is_valid_weight(weight)
        } else {
            Ok(W::zero())
        }
    };
    let mut paths = DictMap::with_capacity(graph.node_count());
    let mut distance: DictMap<NodeIndex, W> =
        dijkstra(&*graph, dummy, None, cost_fn, Some(&mut paths))?;
    paths.remove(&dummy);
    distance.remove(&dummy);
    graph.remove_node(dummy);

    // ``partition[u]`` holds the terminal node closest to node ``u``.
    let mut partition: Vec<usize> = vec![std::usize::MAX; graph.node_bound()];
    for (u, path) in paths.iter() {
        let u = u.index();
        partition[u] = path[1].index();
    }

    let mut out_edges: Vec<MetricClosureEdge<W>> = Vec::with_capacity(graph.edge_count());
    for edge in graph.edge_references() {
        let source = edge.source();
        let target = edge.target();
        // assert that ``source`` is reachable from a terminal node.
        if distance.contains_key(&source) {
            let weight: W = distance[&source] + cost_fn(edge)? + distance[&target];
            let mut path: Vec<usize> = paths[&source].iter().skip(1).map(|x| x.index()).collect();
            path.append(
                &mut paths[&target]
                    .iter()
                    .skip(1)
                    .rev()
                    .map(|x| x.index())
                    .collect(),
            );

            let source = source.index();
            let target = target.index();

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

    //TODO
    Ok(Vec::new())
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
/// :param PyGraph graph: The graph to compute the minimum Steiner tree for
/// :param list terminal_nodes: The list of node indices for which the Steiner
///     tree is to be computed for.
/// :param weight_fn: A callable object that will be passed an edge's
///     weight/data payload and expected to return a ``float``. For example,
///     you can use ``weight_fn=float`` to cast every weight as a float.
///
/// :returns: An approximation to the minimal steiner tree of ``graph`` induced
///     by ``terminal_nodes``.
/// :rtype: PyGraph
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
///
/// .. [1] Kou, Markowsky & Berman,
///    "A fast algorithm for Steiner trees"
///    Acta Informatica 15, 141–145 (1981).
///    https://link.springer.com/article/10.1007/BF00288961
/// .. [2] Kurt Mehlhorn,
///    "A faster approximation algorithm for the Steiner problem in graphs"
///    https://doi.org/10.1016/0020-0190(88)90066-X
pub fn steiner_tree<F, E, W>(
    graph: &mut StableGraph<(), W, Directed>,
    terminal_nodes: Vec<usize>,
    weight_fn: &mut F,
) -> Result<StableGraph<(), W, Directed>, E>
where
    W: Copy
        + Clone
        + PartialOrd
        + std::fmt::Debug
        + std::default::Default
        + std::ops::Add<Output = W>,
    F: FnMut(&W) -> Result<f64, E>,
    MetricClosureEdge<W>: Send,
{
    let mut edge_list = fast_metric_edges(graph, terminal_nodes, weight_fn)?;
    let mut subgraphs = UnionFind::<usize>::new(graph.node_bound());
    edge_list.par_sort_unstable_by(|a, b| {
        let weight_a = (a.distance, a.source, a.target);
        let weight_b = (b.distance, b.source, b.target);
        weight_a.partial_cmp(&weight_b).unwrap_or(Ordering::Less)
    });
    let mut mst_edges: Vec<MetricClosureEdge<W>> = Vec::new();
    for float_edge_pair in edge_list {
        let u = float_edge_pair.source;
        let v = float_edge_pair.target;
        if subgraphs.union(u, v) {
            mst_edges.push(float_edge_pair);
        }
    }
    //TODO implement error
    // assert that the terminal nodes are connected.
    //if !terminal_nodes.is_empty() && mst_edges.len() != terminal_nodes.len() - 1 {
    //return Err(PyValueError::new_err( "The terminal nodes in the input graph must belong to the same connected component. The steiner tree is not defined for a graph with unconnected terminal nodes",));
    //}
    // Generate the output graph from the MST
    let out_edge_list: Vec<[usize; 2]> = mst_edges
        .into_iter()
        .flat_map(|edge| pairwise(edge.path))
        .filter_map(|x| x.0.map(|a| [a, x.1]))
        .collect();
    let out_edges: HashSet<(usize, usize)> = out_edge_list.iter().map(|x| (x[0], x[1])).collect();
    let mut out_graph = graph.clone();
    let out_nodes: HashSet<NodeIndex> = out_edge_list
        .iter()
        .flat_map(|x| x.iter())
        .copied()
        .map(NodeIndex::new)
        .collect();
    for node in graph
        .node_indices()
        .filter(|node| !out_nodes.contains(node))
    {
        out_graph.remove_node(node);
        //    out_graph.node_removed = true;
    }
    for edge in graph.edge_references().filter(|edge| {
        let source = edge.source().index();
        let target = edge.target().index();
        !out_edges.contains(&(source, target)) && !out_edges.contains(&(target, source))
    }) {
        out_graph.remove_edge(edge.id());
    }
    // Deduplicate potential duplicate edges
    deduplicate_edges(&mut out_graph, weight_fn)?;

    Ok(out_graph)
}

fn deduplicate_edges<F, E, W>(
    out_graph: &mut StableGraph<(), W, Directed>,
    weight_fn: &mut F,
) -> Result<(), E>
where
    W: Clone,
    F: FnMut(&W) -> Result<f64, E>,
{
    //if out_graph.multigraph {
    if true {
        // Find all edges between nodes
        let mut duplicate_map: HashMap<
            [NodeIndex; 2],
            Vec<(<StableGraph<(), W, Directed> as GraphBase>::EdgeId, W)>,
        > = HashMap::new();
        for edge in out_graph.edge_references() {
            if duplicate_map.contains_key(&[edge.source(), edge.target()]) {
                duplicate_map
                    .get_mut(&[edge.source(), edge.target()])
                    .unwrap()
                    .push((edge.id(), edge.weight().clone()));
            } else if duplicate_map.contains_key(&[edge.target(), edge.source()]) {
                duplicate_map
                    .get_mut(&[edge.target(), edge.source()])
                    .unwrap()
                    .push((edge.id(), edge.weight().clone()));
            } else {
                duplicate_map.insert(
                    [edge.source(), edge.target()],
                    vec![(edge.id(), edge.weight().clone())],
                );
            }
        }
        // For a node pair with > 1 edge find minimum edge and remove others
        for edges_raw in duplicate_map.values().filter(|x| x.len() > 1) {
            let mut edges: Vec<(<StableGraph<(), W, Directed> as GraphBase>::EdgeId, f64)> =
                Vec::with_capacity(edges_raw.len());
            for edge in edges_raw {
                let w = weight_fn(&edge.1)?;
                edges.push((edge.0, w));
            }
            edges.sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(Ordering::Less));
            edges[1..].iter().for_each(|x| {
                out_graph.remove_edge(x.0);
            });
        }
    }
    Ok(())
}

#[inline]
fn is_valid_weight<W: Float, E>(val: W) -> Result<W, E> {
    if val.is_sign_negative() {
        return Err(E);
        //return Err(E "Negative weights not supported.");
    }

    if val.is_nan() {
        return Err(E);
        //return Err(E "NaN weights not supported.");
    }

    Ok(val)
}
