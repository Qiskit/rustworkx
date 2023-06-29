use hashbrown::HashMap;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::{EdgeRef, GraphBase, IntoEdgeReferences};
use petgraph::Directed;
use std::cmp::Ordering;

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
