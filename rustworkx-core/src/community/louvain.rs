use super::metrics::{ModularityComputable, Partition};
use super::utils::total_edge_weight;

use petgraph::{graph::UnGraph, visit::EdgeRef};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::{HashMap, HashSet};

fn one_level_undirected<'g, G>(
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
    let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
    for e in graph.edge_references() {
        let (a, b) = (
            partition.subset_idx(e.source()),
            partition.subset_idx(e.target()),
        );
        let w: f64 = (*e.weight()).into();
        edges.entry((a, b)).and_modify(|x| *x += w).or_insert(w);
    }

    let aggregated_graph: UnGraph<(), f64, usize> =
        UnGraph::from_edges(edges.iter().map(|(k, &v)| (k.0, k.1, v)));
    let node_count = aggregated_graph.node_count();

    let mut node_to_community: Vec<usize> = (0..node_count).collect();

    let mut degrees = vec![0.0; node_count];
    for e in aggregated_graph.edge_references() {
        let w = e.weight();
        degrees[e.source().index()] += w;
        degrees[e.target().index()] += w;
    }
    let mut s_tot = degrees.clone();

    let mut total_gain = 0.0;
    loop {
        let mut performed_move = false;

        let mut node_shuffle: Pcg64 = match seed {
            Some(rng_seed) => Pcg64::seed_from_u64(rng_seed),
            None => Pcg64::from_entropy(),
        };

        for node in rand::seq::index::sample(&mut node_shuffle, node_count, node_count) {
            let mut neighbor_weights: HashMap<usize, f64> = HashMap::new();
            for nbr in aggregated_graph.neighbors_undirected(node.into()) {
                for e in aggregated_graph.edges_connecting(node.into(), nbr) {
                    let w = e.weight();
                    let com = node_to_community[nbr.index()];
                    neighbor_weights
                        .entry(com)
                        .and_modify(|x| *x += w)
                        .or_insert(*w);
                }
            }

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
