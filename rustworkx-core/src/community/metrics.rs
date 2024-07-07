use core::fmt;
use num_traits::Num;
use std::collections::HashMap;
use std::error::Error;
use std::hash::Hash;
use std::ops::AddAssign;
use std::vec::Vec;

use petgraph::visit::{Data, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences};

#[derive(Debug, PartialEq, Eq)]
pub struct NotAPartitionError;
impl Error for NotAPartitionError {}
impl fmt::Display for NotAPartitionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "The input communities do not form a partition of the input graph."
        )
    }
}

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

pub fn modularity<G, W>(
    graph: &G,
    communities: &Vec<Vec<G::NodeId>>,
    resolution: f64,
) -> Result<f64, NotAPartitionError>
where
    G: GraphProp + Data<EdgeWeight = W> + IntoEdgeReferences,
    <G as GraphBase>::NodeId: Hash + Eq + Copy,
    W: Num + Copy + Into<f64> + AddAssign,
{
    let mut node_to_community: HashMap<G::NodeId, usize> =
        HashMap::with_capacity(communities.iter().map(|v| v.len()).sum());
    for (ii, &ref v) in communities.iter().enumerate() {
        for &node in v {
            if let Some(_n) = node_to_community.insert(node, ii) {
                // argument `communities` contains a duplicate node
                return Err(NotAPartitionError {});
            }
        }
    }

    let mut internal_edge_weights = vec![W::zero(); communities.len()];
    let mut outgoing_edge_weights = vec![W::zero(); communities.len()];
    let mut incoming_edge_weights = if graph.is_directed() {
        Some(vec![W::zero(); communities.len()])
    } else {
        None
    };

    for edge in graph.edge_references() {
        let (a, b) = (edge.source(), edge.target());
        if let (Some(&c_a), Some(&c_b)) = (node_to_community.get(&a), node_to_community.get(&b)) {
            let &w = edge.weight();
            if c_a == c_b {
                internal_edge_weights[c_a] += w;
            }
            outgoing_edge_weights[c_a] += w;
            if let Some(ref mut incoming) = incoming_edge_weights {
                incoming[c_b] += w;
            }
        } else {
            // At least one node was not included in `communities`
            return Err(NotAPartitionError {});
        }
    }

    let two_m: f64 = 2.0 * _total_edge_weight(graph).into();
    let weight_sum = |v: Vec<W>| -> f64 { v.iter().fold(W::zero(), |s, &w| s + w).into() };

    let sigma_internal = weight_sum(internal_edge_weights);
    let sigma_outgoing = weight_sum(outgoing_edge_weights);

    let sigma_incoming = if let Some(incoming) = incoming_edge_weights {
        weight_sum(incoming)
    } else {
        sigma_outgoing
    };

    Ok(sigma_internal / two_m - resolution * sigma_incoming * sigma_outgoing / (two_m * two_m))
}
