use super::metrics::{Modularity, Partition};
use super::utils::total_edge_weight;

use petgraph::EdgeDirection;
use petgraph::{graph::UnGraph, visit::EdgeRef};
use rand::SeedableRng;
use rand_pcg::Pcg32;
use std::collections::{HashMap, HashSet};

/// Enum that holds an "inner graph" for one level of the Louvain algorithm,
/// i.e. a graph in which each community from the previous level is treated
/// as a single node.
///
/// For the first stage of the algorithm, each node from the input graph
/// start out in its own community, so the inner graph is the same as the
/// input graph. In this case we avoid copying the input.
enum InnerGraph<'g, G>
where
    G: Modularity,
{
    Init(&'g G),
    Undirected(UnGraph<(), f64, usize>),
    // Directed case is not implemented yet
    // Directed(DiGraph<(), f64, usize>)
}

impl<'g, G: Modularity> InnerGraph<'g, G> {
    /// Returns the number of nodes in the inner graph
    pub fn node_count(&self) -> usize {
        match self {
            InnerGraph::Init(&g) => g.node_count(),
            InnerGraph::Undirected(g) => g.node_count(),
        }
    }

    /// Returns a vector `w` where `w[i]` is the total weight of the
    /// edges incident on the `i`th node.
    pub fn degrees(&self) -> Vec<f64> {
        let mut degrees = vec![0.0; self.node_count()];
        match self {
            InnerGraph::Init(&g) => {
                for e in g.edge_references() {
                    let w = (*e.weight()).into();
                    let (a, b) = (g.to_index(e.source()), g.to_index(e.target()));
                    degrees[a] += w;
                    degrees[b] += w;
                }
            }
            InnerGraph::Undirected(g) => {
                for e in g.edge_references() {
                    let w = e.weight();
                    degrees[e.source().index()] += w;
                    degrees[e.target().index()] += w;
                }
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

        match self {
            InnerGraph::Init(&g) => {
                let node = g.from_index(idx);
                for edge in g.edges_directed(node, EdgeDirection::Outgoing) {
                    let n = g.to_index(edge.target());
                    add_weight(n, (*edge.weight()).into());
                }
            }
            InnerGraph::Undirected(g) => {
                for edge in g.edges_directed(idx.into(), EdgeDirection::Outgoing) {
                    let n = edge.target().index();
                    add_weight(n, *edge.weight());
                }
            }
        }

        weights
    }
}

/// Trait for additional functions used int the Louvain algorithm. Since the idea
/// is to compute increasingly coarse partitions of the input graph, we implement
/// these for `Partition`.
trait LouvainAlgo<'g, G>
where
    G: Modularity,
{
    /// Compute the inner graph for a given partition.
    fn to_inner_graph(&self) -> InnerGraph<'g, G>;

    /// Replaces the current partition. The argument `new_partition` should be
    /// a vector of size `n` (where `n` is the number of nodes in `self.graph`).
    fn update(&mut self, new_partition: Vec<usize>);

    /// Returns the current graph partition as a vector of sets of `NodeId`, for
    /// example to return to the Python layer.
    fn to_vec_of_hashsets(&self) -> Vec<HashSet<G::NodeId>>;
}

impl<'g, G: Modularity> LouvainAlgo<'g, G> for Partition<'g, G> {
    fn to_inner_graph(&self) -> InnerGraph<'g, G> {
        if self.n_subsets == self.graph.node_count() {
            return InnerGraph::Init(self.graph);
        }

        // Construct a new graph where:
        //   - Node `n_i` corresponds to the `i`th community in the partition
        //   - Nodes `n_i` and `n_j` have an edge with weight `w`, where `w` is
        //     the sum of all edge weights connecting nodes in `n_i` and `n_j`.
        //     (including self-loops)
        let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
        for e in self.graph.edge_references() {
            let (a, b) = (self.subset_idx(e.source()), self.subset_idx(e.target()));
            let inner_edge = if self.graph.is_directed() {
                (std::cmp::min(a, b), std::cmp::max(a, b))
            } else {
                (a, b)
            };
            let w: f64 = (*e.weight()).into();
            edges.entry(inner_edge).and_modify(|x| *x += w).or_insert(w);
        }

        InnerGraph::Undirected(UnGraph::from_edges(
            edges.iter().map(|(k, &v)| (k.0, k.1, v)),
        ))
    }

    fn update(&mut self, new_partition: Vec<usize>) {
        self.node_to_subset = new_partition;
        self.n_subsets = *self.node_to_subset.iter().max().unwrap_or(&0) + 1;
    }

    fn to_vec_of_hashsets(&self) -> Vec<HashSet<G::NodeId>> {
        let mut v = vec![HashSet::new(); self.n_subsets];
        for (idx, &s) in self.node_to_subset.iter().enumerate() {
            let node = self.graph.from_index(idx);
            v[s].insert(node);
        }
        v
    }
}

/// Performs one level of the Louvain algorithm.
///
/// Arguments:
///
/// * `partition`: The current partition of the input graph
/// * `m`: Total weight of the edges of `graph`
/// * `resolution` : controls whether the algorithm favors larger communities (`resolution < 1`) or smaller communities (`resolution < 1`)
/// * `gain_threshold` : minimum acceptable increase in modularity
/// * `seed` : optional seed to determine the order in which we consider moving each node into a neighboring community
///
/// Returns true if it was possible to meet the specified `gain_threshold` by combining nodes into communities.
fn one_level_undirected<G>(
    partition: &mut Partition<G>,
    m: f64,
    resolution: f64,
    gain_threshold: f64,
    seed: Option<u64>,
) -> bool
where
    G: Modularity,
{
    let inner_graph = partition.to_inner_graph();

    let node_count = inner_graph.node_count();

    let degrees = inner_graph.degrees();
    let mut s_tot = degrees.clone();

    // Start by placing each node into its own community
    let mut node_to_community: Vec<usize> = (0..node_count).collect();
    let mut total_gain = 0.0;
    loop {
        let mut performed_move = false;

        let mut node_shuffle: Pcg32 = match seed {
            Some(rng_seed) => Pcg32::seed_from_u64(rng_seed),
            None => Pcg32::from_entropy(),
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

            s_tot[best_com] -= deg;

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
    let input_graph = &partition.graph;
    let mut final_index = HashMap::new();
    let mut next_com = 0;
    let mut updated_partition: Vec<usize> = vec![0; input_graph.node_count()];

    for n in input_graph.node_identifiers() {
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
        updated_partition[input_graph.to_index(n)] = new_com;
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
    G: Modularity,
{
    let mut partition = Partition::new(&graph);

    let m = total_edge_weight(&graph);

    let mut n_levels = 0;
    while one_level_undirected(&mut partition, m, resolution, gain_threshold, seed) {
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
    use crate::generators::barbell_graph;
    use petgraph::graph::UnGraph;

    use super::louvain_communities;

    #[test]
    fn test_louvain_barbell_graph() {
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
    }

    #[test]
    fn test_louvain_karate_club_graph() {
        let edges = [
            (0, 1, 4.0),
            (0, 2, 5.0),
            (0, 3, 3.0),
            (0, 4, 3.0),
            (0, 5, 3.0),
            (0, 6, 3.0),
            (0, 7, 2.0),
            (0, 8, 2.0),
            (0, 10, 2.0),
            (0, 11, 3.0),
            (0, 12, 1.0),
            (0, 13, 3.0),
            (0, 17, 2.0),
            (0, 19, 2.0),
            (0, 21, 2.0),
            (0, 31, 2.0),
            (1, 2, 6.0),
            (1, 3, 3.0),
            (1, 7, 4.0),
            (1, 13, 5.0),
            (1, 17, 1.0),
            (1, 19, 2.0),
            (1, 21, 2.0),
            (1, 30, 2.0),
            (2, 3, 3.0),
            (2, 7, 4.0),
            (2, 8, 5.0),
            (2, 9, 1.0),
            (2, 13, 3.0),
            (2, 27, 2.0),
            (2, 28, 2.0),
            (2, 32, 2.0),
            (3, 7, 3.0),
            (3, 12, 3.0),
            (3, 13, 3.0),
            (4, 6, 2.0),
            (4, 10, 3.0),
            (5, 6, 5.0),
            (5, 10, 3.0),
            (5, 16, 3.0),
            (6, 16, 3.0),
            (8, 30, 3.0),
            (8, 32, 3.0),
            (8, 33, 4.0),
            (9, 33, 2.0),
            (13, 33, 3.0),
            (14, 32, 3.0),
            (14, 33, 2.0),
            (15, 32, 3.0),
            (15, 33, 4.0),
            (18, 32, 1.0),
            (18, 33, 2.0),
            (19, 33, 1.0),
            (20, 32, 3.0),
            (20, 33, 1.0),
            (22, 32, 2.0),
            (22, 33, 3.0),
            (23, 25, 5.0),
            (23, 27, 4.0),
            (23, 29, 3.0),
            (23, 32, 5.0),
            (23, 33, 4.0),
            (24, 25, 2.0),
            (24, 27, 3.0),
            (24, 31, 2.0),
            (25, 31, 7.0),
            (26, 29, 4.0),
            (26, 33, 2.0),
            (27, 33, 4.0),
            (28, 31, 2.0),
            (28, 33, 2.0),
            (29, 32, 4.0),
            (29, 33, 2.0),
            (30, 32, 3.0),
            (30, 33, 3.0),
            (31, 32, 4.0),
            (31, 33, 4.0),
            (32, 33, 5.0),
        ];
        let graph: UnGraph<(), f64> = UnGraph::from_edges(edges.iter());
        let communities = louvain_communities(&graph, 1.0, 0.01, None, Some(7));

        // The result is very sensitive to the random seed. For this seed we
        // happen to get the same result as:
        //      import networkx as nx
        //      g = nx.karate_club_graph()
        //      communities = nx.community.louvain_communities(g, weight='weight', seed=12)
        let mut vecs: Vec<Vec<usize>> = communities
            .iter()
            .map(|h| h.iter().map(|n| n.index()).collect::<Vec<usize>>())
            .collect::<Vec<Vec<usize>>>();
        for v in vecs.iter_mut() {
            v.sort();
        }
        assert_eq!(vecs[0], vec![0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21]);
        assert_eq!(vecs[1], vec![4, 5, 6, 10, 16]);
        assert_eq!(
            vecs[2],
            vec![8, 9, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33]
        );
        assert_eq!(vecs[3], vec![24, 25, 28, 31]);
    }
}
