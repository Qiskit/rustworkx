use hashbrown::HashMap;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::NodeIndexable;
use petgraph::visit::{EdgeRef, GraphBase, IntoEdgeReferences};
use petgraph::Directed;
use rayon::prelude::ParallelSliceMut;
use std::cmp::Ordering;

use crate::petgraph::unionfind::UnionFind;

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
    // let edges = _metric_closure_edges(graph, weight_fn)?;
    //for edge in edges {
    //   out_graph.add_edge(
    //      NodeIndex::new(edge.source),
    //     NodeIndex::new(edge.target),
    //    edge.distance,
    //);
    //}
    Ok(out_graph)
}

fn _metric_closure_edges<F, E, W>(
    graph: &StableGraph<(), W, Directed>,
    weight_fn: &mut F,
) -> Result<Vec<MetricClosureEdge>, E> {
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
) -> Result<Vec<MetricClosureEdge>, E>
where
    W: Clone,
    F: FnMut(&W) -> Result<f64, E>,
{
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
    //) -> Result<StableGraph<(), W, Directed>, E>
) -> Result<(), E>
where
    W: Clone,
    F: FnMut(&W) -> Result<f64, E>,
{
    let mut edge_list = fast_metric_edges(graph, terminal_nodes, &mut weight_fn)?;
    let mut subgraphs = UnionFind::<usize>::new(graph.node_bound());
    edge_list.par_sort_unstable_by(|a, b| {
        let weight_a = (a.distance, a.source, a.target);
        let weight_b = (b.distance, b.source, b.target);
        weight_a.partial_cmp(&weight_b).unwrap_or(Ordering::Less)
    });
    Ok(())
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
