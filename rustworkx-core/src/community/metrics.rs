use core::fmt;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::hash::Hash;

use petgraph::visit::{
    Data, EdgeRef, GraphProp, IntoEdgeReferences, IntoNodeReferences, NodeCount,
};

#[derive(Debug, PartialEq, Eq)]
pub struct NotAPartitionError;
impl Error for NotAPartitionError {}
impl fmt::Display for NotAPartitionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "The input subsets do not form a partition of the input graph."
        )
    }
}

pub trait ModularityComputable:
    Data<EdgeWeight: Into<f64> + Copy, NodeId: Hash + Eq + Copy>
    + GraphProp
    + IntoEdgeReferences
    + NodeCount
    + IntoNodeReferences
{
}
impl<
        Graph: Data<EdgeWeight: Into<f64> + Copy, NodeId: Hash + Eq + Copy>
            + GraphProp
            + IntoEdgeReferences
            + NodeCount
            + IntoNodeReferences,
    > ModularityComputable for Graph
{
}

pub struct Partition<'g, G>
where
    G: ModularityComputable,
{
    graph: &'g G,
    n_subsets: usize,
    node_to_subset: HashMap<G::NodeId, usize>,
}
pub struct PartitionEdgeWeights {
    pub internal: Vec<f64>,
    pub outgoing: Vec<f64>,
    pub incoming: Option<Vec<f64>>,
}

impl<'g, G: ModularityComputable> Partition<'g, G> {
    pub fn new(
        graph: &'g G,
        subsets: &[HashSet<G::NodeId>],
    ) -> Result<Partition<'g, G>, NotAPartitionError> {
        let mut node_to_subset: HashMap<G::NodeId, usize> =
            HashMap::with_capacity(subsets.iter().map(|v| v.len()).sum());
        for (ii, v) in subsets.iter().enumerate() {
            for &node in v {
                if let Some(_n) = node_to_subset.insert(node, ii) {
                    // argument `communities` contains a duplicate node
                    return Err(NotAPartitionError {});
                }
            }
        }

        if node_to_subset.len() != graph.node_count() {
            return Err(NotAPartitionError {});
        }

        Ok(Partition::<'g, G> {
            graph: graph,
            n_subsets: subsets.len(),
            node_to_subset: node_to_subset,
        })
    }
    pub fn total_edge_weight(&self) -> f64 {
        self.graph
            .edge_references()
            .map(|edge| *edge.weight())
            .fold(0.0, |s, e| s + e.into())
    }

    pub fn get_subset_id(&self, node: &G::NodeId) -> Option<&usize> {
        self.node_to_subset.get(node)
    }

    pub fn partition_edge_weights(&self) -> Result<PartitionEdgeWeights, NotAPartitionError> {
        let mut internal_edge_weights = vec![0.0; self.n_subsets];
        let mut outgoing_edge_weights = vec![0.0; self.n_subsets];

        let directed = self.graph.is_directed();
        let mut incoming_edge_weights = if directed {
            Some(vec![0.0; self.n_subsets])
        } else {
            None
        };

        for edge in self.graph.edge_references() {
            let (a, b) = (edge.source(), edge.target());
            if let (Some(&c_a), Some(&c_b)) = (self.get_subset_id(&a), self.get_subset_id(&b)) {
                let w: f64 = (*edge.weight()).into();
                if c_a == c_b {
                    internal_edge_weights[c_a] += w;
                }
                outgoing_edge_weights[c_a] += w;
                if let Some(ref mut incoming) = incoming_edge_weights {
                    incoming[c_b] += w;
                } else {
                    outgoing_edge_weights[c_b] += w;
                }
            } else {
                return Err(NotAPartitionError {});
            }
        }

        Ok(PartitionEdgeWeights {
            internal: internal_edge_weights,
            outgoing: outgoing_edge_weights,
            incoming: incoming_edge_weights,
        })
    }

    pub fn modularity(&self, resolution: f64) -> Result<f64, NotAPartitionError> {
        let weights = self.partition_edge_weights()?;

        let sigma_internal: f64 = weights.internal.iter().sum();

        let sigma_total_squared: f64 = if let Some(incoming) = weights.incoming {
            incoming
                .iter()
                .zip(weights.outgoing.iter())
                .map(|(&x, &y)| x * y)
                .sum()
        } else {
            weights.outgoing.iter().map(|&x| x * x).sum::<f64>() / 4.0
        };

        let m: f64 = self.total_edge_weight();
        Ok(sigma_internal / m - resolution * sigma_total_squared / (m * m))
    }
}

pub fn modularity<G>(
    graph: G,
    communities: &[HashSet<G::NodeId>],
    resolution: f64,
) -> Result<f64, NotAPartitionError>
where
    G: ModularityComputable,
{
    let partition = Partition::new(&graph, &communities)?;

    partition.modularity(resolution)
}

#[cfg(test)]
mod tests {
    use crate::generators::barbell_graph;
    use petgraph::graph::{DiGraph, UnGraph};
    use petgraph::visit::{GraphBase, IntoNodeIdentifiers};
    use std::collections::HashSet;

    use super::modularity;

    #[test]
    fn test_modularity_barbell_graph() {
        type G = UnGraph<(), f64>;
        type N = <G as GraphBase>::NodeId;

        for n in 3..10 {
            let g: G = barbell_graph(Some(n), Some(0), None, None, || (), || 1.0f64).unwrap();
            let nodes: Vec<N> = g.node_identifiers().collect();
            let communities: Vec<HashSet<N>> = vec![
                (0..n).map(|ii| nodes[ii]).collect(),
                (n..(2 * n)).map(|ii| nodes[ii]).collect(),
            ];
            let resolution = 1.0;
            let m = modularity(&g, &communities, resolution).unwrap();
            // There are two complete subgraphs, each with:
            //     * e = n*(n-1)/2 internal edges
            //     * total node degree 2*e + 1
            // The edge weight for the whole graph is 2*e + 1. So the expected
            // modularity is 2 * [ e/(2*e + 1) - 1/4 ].
            let e = (n * (n - 1) / 2) as f64;
            let m_expected = 2.0 * (e / (2.0 * e + 1.0) - 0.25);
            assert!((m - m_expected).abs() < 1.0e-9);
        }
    }

    #[test]
    fn test_modularity_directed() {
        type G = DiGraph<(), f64>;
        type N = <G as GraphBase>::NodeId;

        for n in 3..10 {
            let mut g = G::with_capacity(2 * n, 2 * n + 2);
            for _ii in 0..2 * n {
                g.add_node(());
            }
            let nodes: Vec<N> = g.node_identifiers().collect();
            // Create two cycles
            for ii in 0..n {
                let jj = (ii + 1) % n;
                g.add_edge(nodes[ii], nodes[jj], 1.0);
                g.add_edge(nodes[n + ii], nodes[n + jj], 1.0);
            }
            // Add two edges connecting the cycles
            g.add_edge(nodes[0], nodes[n], 1.0);
            g.add_edge(nodes[n + 1], nodes[1], 1.0);

            let communities: Vec<HashSet<N>> = vec![
                (0..n).map(|ii| nodes[ii]).collect(),
                (n..2 * n).map(|ii| nodes[ii]).collect(),
            ];

            let resolution = 1.0;
            let m = modularity(&g, &communities, resolution).unwrap();

            // Each cycle subgraph has:
            //     * n internal edges
            //     * total node degree n + 1 (outgoing) and n + 1 (incoming)
            // The edge weight for the whole graph is 2*n + 2. So the expected
            // modularity is 2 * [ n/(2*n + 2) - (n+1)^2 / (2*n + 2)^2 ]
            //               = n/(n + 1) - 1/2
            let m_expected = n as f64 / (n as f64 + 1.0) - 0.5;
            assert!((m - m_expected).abs() < 1.0e-9);
        }
    }
}
