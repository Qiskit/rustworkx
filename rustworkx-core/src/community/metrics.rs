use num_traits::Num;
use std::collections::HashSet;
use std::vec::Vec;

use petgraph::graph::NodeIndex;
use petgraph::visit::{Data, EdgeRef, IntoEdgeReferences, NodeIndexable};

fn _number_internal_edges(graph: &G, community: &HashSet<NodeIndex>) -> u64 {}

fn _total_edge_weight<G, W>(graph: &G) -> W
where
    G: Data<EdgeWeight = W> + IntoEdgeReferences,
    W: Num + Copy,
{
    graph
        .edge_references()
        .map(|edge| *edge.weight())
        .fold(W::zero(), |s, e| s + e)
}

pub fn modularity<G, W>(graph: &G, communities: &Vec<Vec<NodeIndex>>, resolution: f64) -> f64
where
    G: Data<EdgeWeight = W> + NodeIndexable + IntoEdgeReferences,
    W: Num,
{
}
