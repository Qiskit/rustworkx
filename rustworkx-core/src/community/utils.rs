use super::metrics::{Modularity, ModularityEdgeWeight};
use petgraph::visit::{Data, EdgeRef};

pub fn total_edge_weight<G>(graph: &G) -> f64
where
    G: Modularity,
    <G as Data>::EdgeWeight: ModularityEdgeWeight,
{
    graph
        .edge_references()
        .map(|edge| *edge.weight())
        .fold(0.0, |s, e| s + e.into())
}
