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
    graph: G,
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
    let mut incoming_edge_weights_opt = if graph.is_directed() {
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
            if let Some(ref mut incoming_edge_weights) = incoming_edge_weights_opt {
                incoming_edge_weights[c_b] += w;
            } else {
                outgoing_edge_weights[c_b] += w;
            }
        } else {
            // At least one node was not included in `communities`
            return Err(NotAPartitionError {});
        }
    }

    let m: f64 = _total_edge_weight(&graph).into();

    let sigma_internal: f64 = internal_edge_weights
        .iter()
        .fold(W::zero(), |s, &w| s + w)
        .into();

    let sigma_total_squared: W = if let Some(incoming_edge_weights) = incoming_edge_weights_opt {
        incoming_edge_weights
            .iter()
            .zip(outgoing_edge_weights.iter())
            .fold(W::zero(), |s, (&x, &y)| s + x * y)
    } else {
        outgoing_edge_weights
            .iter()
            .fold(W::zero(), |s, &x| s + x * x)
    };

    Ok(sigma_internal / m - resolution * sigma_total_squared.into() / (4.0 * m * m))
}

#[cfg(test)]
mod tests {
    use crate::generators::barbell_graph;
    use petgraph::graph::UnGraph;
    use petgraph::visit::{GraphBase, IntoNodeIdentifiers};
    use std::vec::Vec;

    use super::modularity;

    #[test]
    fn test_modularity_barbell_graph() {
        type G = UnGraph<(), f64>;
        type N = <G as GraphBase>::NodeId;

        let g: G = barbell_graph(Some(3), Some(0), None, None, || (), || 1.0f64).unwrap();
        let nodes: Vec<N> = g.node_identifiers().collect();
        let communities: Vec<Vec<N>> = vec![
            vec![nodes[0], nodes[1], nodes[2]],
            vec![nodes[3], nodes[4], nodes[5]],
        ];
        let resolution = 1.0;
        let m = modularity(&g, &communities, resolution).unwrap();
        assert!((m - 0.35714285714285715).abs() < 1.0e-9);
    }
}
