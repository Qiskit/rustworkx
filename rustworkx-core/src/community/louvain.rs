use super::metrics::{ModularityComputable, Partition};
use super::utils::total_edge_weight;

use super::NotAPartitionError;
use petgraph::{
    graph::UnGraph,
    visit::{EdgeRef, NodeRef},
};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::{HashMap, HashSet};

fn _one_level_undirected<'g, G>(
    graph: &G,
    current_partition: &Partition<G>,
    m: f64,
    resolution: f64,
    gain_threshold: f64,
    seed: Option<u64>,
) -> Option<Vec<HashSet<G::NodeId>>>
where
    G: ModularityComputable,
{
    let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
    for e in graph.edge_references() {
        let (a, b) = (
            current_partition.get_subset_id(e.source()),
            current_partition.get_subset_id(e.target()),
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
        return None;
    }

    let mut com_to_final_index = HashMap::new();
    let mut final_partition: Vec<HashSet<G::NodeId>> = Vec::new();

    for n in graph.node_identifiers() {
        let prev_com = current_partition.get_subset_id(n);
        let inner_com = node_to_community[prev_com];
        let new_com = if let Some(&c) = com_to_final_index.get(&inner_com) {
            c
        } else {
            let n_com = final_partition.len();
            com_to_final_index.insert(inner_com, n_com);
            final_partition.push(HashSet::new());
            n_com
        };
        final_partition[new_com].insert(n);
    }
    Some(final_partition)
}

pub fn louvain_communities<G>(
    graph: G,
    resolution: f64,
    gain_threshold: f64,
    max_level: Option<u32>,
    seed: Option<u64>,
) -> Result<Vec<HashSet<G::NodeId>>, NotAPartitionError>
where
    G: ModularityComputable,
{
    let mut result: Vec<HashSet<G::NodeId>> = graph
        .node_references()
        .map(|n| HashSet::from([n.id()]))
        .collect();
    let mut current_partition = Partition::new(&graph, &result)?;

    let m = total_edge_weight(&graph);

    let mut n_levels = 0;
    while let Some(improved_partition) = _one_level_undirected(
        &graph,
        &current_partition,
        m,
        resolution,
        gain_threshold,
        seed,
    ) {
        let current_modularity = current_partition.modularity(resolution);

        result = improved_partition;
        current_partition = Partition::new(&graph, &result)?;

        let improved_modularity = current_partition.modularity(resolution);
        if improved_modularity - current_modularity < gain_threshold {
            break;
        }

        match max_level {
            Some(t) => {
                n_levels += 1;
                if n_levels >= t {
                    break;
                }
            }
            None => (),
        };
    }

    Ok(result)
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
            let result = louvain_communities(&g, resolution, gain_threshold, None, None)?;
            // For a barbell graph, we expect the Louvain algorithm to identify
            // the two complete subgraphs as the final communities
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].len(), n);
            assert_eq!(result[1].len(), n);
        }
        Ok(())
    }
}
