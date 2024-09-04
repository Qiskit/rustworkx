use super::metrics::{ModularityComputable, Partition};
use super::utils::total_edge_weight;

use petgraph::EdgeDirection;
use petgraph::{graph::UnGraph, visit::EdgeRef};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::{HashMap, HashSet};

/// Struct that holds an "inner graph" for one level of the Louvain algorithm,
/// i.e. a graph in which each community from the previous level is treated
/// as a single node.
struct InnerGraph<'g, G>
where
    G: ModularityComputable,
{
    input_graph: &'g G,
    inner_graph: Option<UnGraph<(), f64, usize>>,
}

impl<'g, G: ModularityComputable> InnerGraph<'g, G> {
    /// Compute the inner graph for a given partition.
    /// ToDo: fix redundant arguments
    pub fn new(graph: &'g G, partition: &Partition<G>) -> InnerGraph<'g, G> {
        if partition.n_subsets() == graph.node_count() {
            // At the start of the Louvain algorithm we put each node from the
            // input graph into its own commnuity, so the inner graph is the
            // same as the input graph. We should avoid copying the input.
            return InnerGraph {
                input_graph: graph,
                inner_graph: None,
            };
        }

        // Construct a new graph where:
        //   - Node `n_i` corresponds to the `i`th community in the partition
        //   - Nodes `n_i` and `n_j` have an edge with weight `w`, where `w` is
        //     the sum of all edge weights connecting nodes in `n_i` and `n_j`.
        //     (including self-loops)
        let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
        for e in graph.edge_references() {
            let (a, b) = (
                partition.subset_idx(e.source()),
                partition.subset_idx(e.target()),
            );
            let inner_edge = if graph.is_directed() {
                (std::cmp::min(a, b), std::cmp::max(a, b))
            } else {
                (a, b)
            };
            let w: f64 = (*e.weight()).into();
            edges.entry(inner_edge).and_modify(|x| *x += w).or_insert(w);
        }

        InnerGraph {
            input_graph: graph,
            inner_graph: Some(UnGraph::from_edges(
                edges.iter().map(|(k, &v)| (k.0, k.1, v)),
            )),
        }
    }

    /// Returns the number of nodes in the inner graph
    pub fn node_count(&self) -> usize {
        if let Some(g) = &self.inner_graph {
            g.node_count()
        } else {
            self.input_graph.node_count()
        }
    }

    /// Returns a vector `w` where `w[i]` is the total weight of the
    /// edges incident on the `i`th node.
    pub fn degrees(&self) -> Vec<f64> {
        let mut degrees = vec![0.0; self.node_count()];
        if let Some(g) = &self.inner_graph {
            for e in g.edge_references() {
                let w = e.weight();
                degrees[e.source().index()] += w;
                degrees[e.target().index()] += w;
            }
        } else {
            for e in self.input_graph.edge_references() {
                let w = (*e.weight()).into();
                let (a, b) = (
                    self.input_graph.to_index(e.source()),
                    self.input_graph.to_index(e.target()),
                );
                degrees[a] += w;
                degrees[b] += w;
            }
        }
        degrees
    }

    /// Given a node index `idx`, returns a map `w`. For each neighbor
    /// `nbr` of `idx`, `w[nbr]` is the total weight of all the edges
    /// connecting `idx` and `nbr`.
    pub fn neighbor_community_weights(
        &self,
        idx: usize,
        node_to_community: &[usize],
    ) -> HashMap<usize, f64> {
        let mut weights = HashMap::new();

        let mut add_weight = |n: usize, w: f64| {
            let com = node_to_community[n];
            weights.entry(com).and_modify(|x| *x += w).or_insert(w);
        };

        if let Some(g) = &self.inner_graph {
            for edge in g.edges_directed(idx.into(), EdgeDirection::Outgoing) {
                let n = edge.target().index();
                add_weight(n, *edge.weight());
            }
        } else {
            let node = self.input_graph.from_index(idx);
            for edge in self
                .input_graph
                .edges_directed(node, EdgeDirection::Outgoing)
            {
                let n = self.input_graph.to_index(edge.target());
                add_weight(n, (*edge.weight()).into());
            }
        }
        weights
    }
}

/// Performs one level of the Louvain algorithm.
///
/// Arguments:
///
/// * `graph`: The input graph
/// * `partition`: The current partition of the input graph
/// * `m`: Total weight of the edges of `graph`
/// * `resolution` : controls whether the algorithm favors larger communities (`resolution < 1`) or smaller communities (`resolution < 1`)
/// * `gain_threshold` : minimum acceptable increase in modularity
/// * `seed` : optional seed to determine the order in which we consider moving each node into a neighboring community
///
/// Returns true if it was possible to meet the specified `gain_threshold` by combining nodes into communities.
fn one_level_undirected<G>(
    graph: &G,
    partition: &mut Partition<G>,
    m: f64,
    resolution: f64,
    gain_threshold: f64,
    seed: Option<u64>,
) -> bool
where
    G: ModularityComputable,
{
    let inner_graph = InnerGraph::new(graph, partition);

    let node_count = inner_graph.node_count();

    let mut degrees = inner_graph.degrees();
    let mut s_tot = degrees.clone();

    // Start by placing each node into its own community
    let mut node_to_community: Vec<usize> = (0..node_count).collect();
    let mut total_gain = 0.0;
    loop {
        let mut performed_move = false;

        let mut node_shuffle: Pcg64 = match seed {
            Some(rng_seed) => Pcg64::seed_from_u64(rng_seed),
            None => Pcg64::from_entropy(),
        };

        // Try moving each node into a neighboring community, in the order
        // determined by `seed`. For each node, select the neighboring community
        // that gives the largest increase in modularity (if any).
        for node in rand::seq::index::sample(&mut node_shuffle, node_count, node_count) {
            let neighbor_weights = inner_graph.neighbor_community_weights(node, &node_to_community);

            let mut best_gain = 0.0;
            let init_com = node_to_community[node];
            let deg = degrees[init_com];
            let mut best_com = init_com;
            let two_m_sq = 2.0 * m * m;

            degrees[best_com] -= deg;

            let delta = if let Some(&w) = neighbor_weights.get(&best_com) {
                w
            } else {
                0.0
            };
            let remove_cost = -delta / m + resolution * (s_tot[best_com] * deg) / two_m_sq;

            for (&nbr_com, &wt) in neighbor_weights.iter() {
                let gain = remove_cost + wt / m - resolution * s_tot[nbr_com] * deg / two_m_sq;
                if gain > best_gain {
                    best_gain = gain;
                    best_com = nbr_com;
                }
            }

            s_tot[best_com] += deg;

            if best_com != init_com {
                performed_move = true;
                total_gain += best_gain;
                node_to_community[node] = best_com;
            }
        }
        if !performed_move {
            break;
        }
    }

    if total_gain < gain_threshold {
        return false;
    }

    // Compute the resulting new partition of the input graph
    let mut final_index = HashMap::new();
    let mut next_com = 0;
    let mut updated_partition: Vec<usize> = vec![0; node_count];

    for n in graph.node_identifiers() {
        let prev_com = partition.subset_idx(n);
        let inner_com = node_to_community[prev_com];
        let new_com = if let Some(&c) = final_index.get(&inner_com) {
            c
        } else {
            let c = next_com;
            final_index.insert(inner_com, c);
            next_com += 1;
            c
        };
        updated_partition[graph.to_index(n)] = new_com;
    }
    partition.update(updated_partition);

    true
}

/// Runs the Louvain community detection algorithm.
///
/// Arguments:
///
/// * `graph`: The input graph
/// * `resolution` : controls whether the algorithm favors larger communities (`resolution < 1`) or smaller communities (`resolution < 1`)
/// * `gain_threshold` : minimum acceptable increase in modularity at each level. The algorithm will
///    terminate if it is not possible to meet this threshold by performing another level of aggregation.
/// * `max_level` : Maximum number of levels (aggregation steps) to perform
/// * `seed` : seed for RNG that determines the order in which we consider moving each node into a neighboring community
pub fn louvain_communities<G>(
    graph: G,
    resolution: f64,
    gain_threshold: f64,
    max_level: Option<u32>,
    seed: Option<u64>,
) -> Vec<HashSet<G::NodeId>>
where
    G: ModularityComputable,
{
    let mut partition = Partition::new_isolated_nodes(&graph);

    let m = total_edge_weight(&graph);

    let mut n_levels = 0;
    while one_level_undirected(&graph, &mut partition, m, resolution, gain_threshold, seed) {
        if let Some(limit) = max_level {
            n_levels += 1;
            if n_levels >= limit {
                break;
            }
        }
    }

    partition.to_vec_of_hashsets()
}

#[cfg(test)]
mod tests {
    use crate::community::NotAPartitionError;
    use crate::generators::barbell_graph;
    use petgraph::graph::UnGraph;

    use super::louvain_communities;

    #[test]
    fn test_louvain_barbell_graph() -> Result<(), NotAPartitionError> {
        type G = UnGraph<(), f64>;

        for n in 3..10 {
            let g: G = barbell_graph(Some(n), Some(0), None, None, || (), || 1.0f64).unwrap();
            let resolution = 1.0;
            let gain_threshold = 0.01;
            let result = louvain_communities(&g, resolution, gain_threshold, None, None);
            // For a barbell graph, we expect the Louvain algorithm to identify
            // the two complete subgraphs as the final communities
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].len(), n);
            assert_eq!(result[1].len(), n);
        }
        Ok(())
    }
}
