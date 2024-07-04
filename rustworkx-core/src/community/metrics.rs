use num_traits::Float;
use std::collections::HashSet;
use std::vec::Vec;

use petgraph::graph::NodeIndex;
use petgraph::visit::{Data, NodeIndexable};

fn _number_internal_edges(graph: &G, community: &HashSet<NodeIndex>) -> u64 {}

fn _total_degree(graph: &G, community: &HashSet<NodeIndex>) -> u64 {}

pub fn modularity<G, W>(graph: &G, communities: &Vec<Vec<NodeIndex>>, resolution: f64) -> f64
where
    G: Data<EdgeWeight = W> + NodeIndexable,
    W: Float,
{
}
