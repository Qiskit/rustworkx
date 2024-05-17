// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

use hashbrown::{HashMap, HashSet};
use petgraph::visit::{
    EdgeRef, GraphBase, IntoEdgesDirected, IntoNeighborsDirected, IntoNodeIdentifiers,
    NodeIndexable,
};

/// Error returned by generator functions when the input arguments are an
/// invalid combination (such as missing required options).
#[derive(Debug, PartialEq, Eq)]
pub struct LayersInvalidIndex(pub Option<String>);

impl Error for LayersInvalidIndex {}

impl fmt::Display for LayersInvalidIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0 {
            Some(message) => write!(f, "{message}"),
            None => write!(
                f,
                "The provided layer contains an index that is not present in the graph"
            ),
        }
    }
}

use std::{error::Error, fmt, hash::Hash};

/// Return a list of layers
///
/// A layer is a subgraph whose nodes are disjoint, i.e.,
/// a layer has depth 1. The layers are constructed using a greedy algorithm.
///
/// :param graph: The DAG to get the layers from
/// :param list first_layer: A list of node ids for the first layer. This
///     will be the first layer in the output
///
/// :returns: A list of layers, each layer is a list of node data, or if
///     ``index_output`` is ``True`` each layer is a list of node indices.
/// :rtype: list
///
/// :raises InvalidNode: If a node index in ``first_layer`` is not in the graph
pub fn layers<G>(graph: G, first_layer: Vec<usize>) -> Result<Vec<Vec<usize>>, LayersInvalidIndex>
where
    G: NodeIndexable // Used in from_index and to_index.
        + IntoNodeIdentifiers // Used for .node_identifiers
        + IntoNeighborsDirected // Used for .neighbors_directed
        + IntoEdgesDirected, // Used for .edged_directed
    <G as GraphBase>::NodeId: Eq + Hash,
{
    let mut output_indices: Vec<Vec<usize>> = Vec::new();

    let first_layer_index: Vec<G::NodeId> =
        first_layer.iter().map(|x| graph.from_index(*x)).collect();
    let mut cur_layer = first_layer_index;
    let mut next_layer: Vec<G::NodeId> = Vec::new();
    let mut predecessor_count: HashMap<G::NodeId, usize> = HashMap::new();

    let node_set = graph.node_identifiers().collect::<HashSet<G::NodeId>>();
    for layer_node in &cur_layer {
        if !node_set.contains(layer_node) {
            return Err(LayersInvalidIndex(Some(format!(
                "An index input in 'first_layer' {} is not a valid node index in the graph",
                graph.to_index(*layer_node)
            ))));
        }
    }
    output_indices.push(cur_layer.iter().map(|x| graph.to_index(*x)).collect());

    // Iterate until there are no more
    while !cur_layer.is_empty() {
        for node in &cur_layer {
            let children = graph.neighbors_directed(*node, petgraph::Direction::Outgoing);
            let mut used_indices: HashSet<G::NodeId> = HashSet::new();
            for succ in children {
                // Skip duplicate successors
                if used_indices.contains(&succ) {
                    continue;
                }
                used_indices.insert(succ);
                let mut multiplicity: usize = 0;
                let raw_edges: G::EdgesDirected =
                    graph.edges_directed(*node, petgraph::Direction::Outgoing);
                for edge in raw_edges {
                    if edge.target() == succ {
                        multiplicity += 1;
                    }
                }
                predecessor_count
                    .entry(succ)
                    .and_modify(|e| *e -= multiplicity)
                    .or_insert(
                        // Get the number of incoming edges to the successor
                        graph
                            .edges_directed(succ, petgraph::Direction::Incoming)
                            .count()
                            - multiplicity,
                    );
                if *predecessor_count.get(&succ).unwrap() == 0 {
                    next_layer.push(succ);
                    predecessor_count.remove(&succ);
                }
            }
        }
        if !next_layer.is_empty() {
            output_indices.push(next_layer.iter().map(|x| graph.to_index(*x)).collect());
        }
        cur_layer = next_layer;
        next_layer = Vec::new();
    }
    Ok(output_indices)
}
