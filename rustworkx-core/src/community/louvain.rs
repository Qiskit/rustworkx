use super::metrics::{ModularityComputable, Partition};
use std::collections::{HashMap, HashSet};

struct LouvainLevel<'g, G>
where
    G: ModularityComputable,
{
    partition: Partition<'g, G>,
}

impl<'g, G: ModularityComputable> LouvainLevel<'g, G> {
    pub fn new(input_graph: &'g G) -> LouvainLevel<'g, G> {
        LouvainLevel {
            partition: Partition::<'g, G>::isolated_nodes_partition(input_graph),
        }
    }
}

pub fn louvain_communities<G>(
    graph: &G,
    resolution: f64,
    gain_threshold: f64,
    max_level: Option<u32>,
    seed: Option<u64>,
) -> Vec<HashSet<G::NodeId>>
where
    G: ModularityComputable,
{
    let current_partition = LouvainLevel::new(graph);
    

    vec![]
}
