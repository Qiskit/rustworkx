use super::Modularity;
use petgraph::visit::EdgeRef;

pub fn total_edge_weight<G>(graph: &G) -> f64
where
    G: Modularity,
{
    graph
        .edge_references()
        .map(|edge| *edge.weight())
        .fold(0.0, |s, e| s + e.into())
}
