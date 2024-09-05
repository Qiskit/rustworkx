use super::utils::total_edge_weight;
use super::NotAPartitionError;

use petgraph::visit::{
    Data, EdgeRef, GraphProp, IntoEdgeReferences, IntoEdgesDirected, IntoNodeReferences, NodeCount,
    NodeIndexable,
};
use std::collections::HashSet;
use std::hash::Hash;

/// Trait for graphs for which it is possible to compute modularity
/// and apply the Louvain community detection method.
pub trait Modularity:
    Data<EdgeWeight: Into<f64> + Copy, NodeId: Hash + Eq + Copy>
    + GraphProp
    + IntoEdgeReferences
    + NodeCount
    + IntoNodeReferences
    + NodeIndexable
    + IntoEdgesDirected
{
}
impl<
        G: Data<EdgeWeight: Into<f64> + Copy, NodeId: Hash + Eq + Copy>
            + GraphProp
            + IntoEdgeReferences
            + NodeCount
            + IntoNodeReferences
            + NodeIndexable
            + IntoEdgesDirected,
    > Modularity for G
{
}

/// Struct representing a partition of a graph as a vector
/// `[s_0, ... s_n]`, where `n` is the number of nodes in
/// the graph and node `i` belongs to subset `s_i`.
pub struct Partition<'g, G>
where
    G: Modularity,
{
    pub graph: &'g G,
    pub n_subsets: usize,
    pub node_to_subset: Vec<usize>,
}

impl<'g, G: Modularity> Partition<'g, G> {
    /// Creates a partition where each node of the input graph is placed
    /// into its own subset, e.g. for the first step of the Louvain algorithm.
    pub fn new(graph: &'g G) -> Partition<'g, G> {
        Partition {
            graph,
            n_subsets: graph.node_count(),
            node_to_subset: (0..graph.node_count()).collect(),
        }
    }

    /// Creates a `Partition` from sets of graph nodes. Checks whether the
    /// sets actually form a partition of the input graph.
    pub fn from_subsets(
        graph: &'g G,
        subsets: &[HashSet<G::NodeId>],
    ) -> Result<Partition<'g, G>, NotAPartitionError> {
        let mut seen = vec![false; graph.node_count()];

        let mut node_to_subset = vec![0; graph.node_count()];

        for (ii, v) in subsets.iter().enumerate() {
            for &node in v {
                let idx = graph.to_index(node);
                if seen[idx] {
                    // argument `communities` contains a duplicate node
                    return Err(NotAPartitionError {});
                }
                node_to_subset[idx] = ii;
                seen[idx] = true;
            }
        }

        if !seen.iter().all(|&t| t) {
            return Err(NotAPartitionError {});
        }

        Ok(Partition::<'g, G> {
            graph,
            n_subsets: subsets.len(),
            node_to_subset,
        })
    }

    /// Returns the index of the subset that contains `node`.
    pub fn subset_idx(&self, node: G::NodeId) -> usize {
        let idx = self.graph.to_index(node);
        self.node_to_subset[idx]
    }

    /// Returns the modularity of the graph with the current partition.
    pub fn modularity(&self, resolution: f64) -> f64 {
        let mut internal_weights = vec![0.0; self.n_subsets];
        let mut outgoing_weights = vec![0.0; self.n_subsets];

        let directed = self.graph.is_directed();
        let mut incoming_weights = if directed {
            Some(vec![0.0; self.n_subsets])
        } else {
            None
        };

        for edge in self.graph.edge_references() {
            let (a, b) = (edge.source(), edge.target());
            let (c_a, c_b) = (self.subset_idx(a), self.subset_idx(b));
            let w: f64 = (*edge.weight()).into();
            if c_a == c_b {
                internal_weights[c_a] += w;
            }
            outgoing_weights[c_a] += w;
            if let Some(ref mut incoming) = incoming_weights {
                incoming[c_b] += w;
            } else {
                outgoing_weights[c_b] += w;
            }
        }

        let sigma_internal: f64 = internal_weights.iter().sum();

        let sigma_total_squared: f64 = if let Some(incoming) = incoming_weights {
            incoming
                .iter()
                .zip(outgoing_weights.iter())
                .map(|(&x, &y)| x * y)
                .sum()
        } else {
            outgoing_weights.iter().map(|&x| x * x).sum::<f64>() / 4.0
        };

        let m: f64 = total_edge_weight(self.graph);
        sigma_internal / m - resolution * sigma_total_squared / (m * m)
    }
}

/// Computes the modularity of a graph, given a partition of its nodes.
///
/// Arguments:
///
/// * `graph` - The input graph
/// * `communities` - Sets of nodes that form a partition of `graph`
/// * `resolution` - Controls the relative weight of intra-community and inter-community edges
pub fn modularity<G>(
    graph: G,
    communities: &[HashSet<G::NodeId>],
    resolution: f64,
) -> Result<f64, NotAPartitionError>
where
    G: Modularity,
{
    let partition = Partition::from_subsets(&graph, communities)?;
    Ok(partition.modularity(resolution))
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
