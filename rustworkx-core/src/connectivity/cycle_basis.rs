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
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, NodeCount};
use std::hash::Hash;

/// Return a list of cycles which form a basis for cycles of a given graph.
///
/// A basis for cycles of a graph is a minimal collection of
/// cycles such that any cycle in the graph can be written
/// as a sum of cycles in the basis.  Here summation of cycles
/// is defined as the exclusive-or of the edges.
///
/// This is adapted from
///    Paton, K. An algorithm for finding a fundamental set of
///    cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.
///
/// The function implicitly assumes that there are no parallel edges.
/// It may produce incorrect/unexpected results if the input graph has
/// parallel edges.
///
///
/// Arguments:
///
/// * `graph` - The graph in which to find the basis.
/// * `root` - Optional node index for starting the basis search. If not
///     specified, an arbitrary node is chosen.
///
/// # Example
/// ```rust
/// use petgraph::prelude::*;
/// use rustworkx_core::connectivity::cycle_basis;
///
/// let edge_list = [(0, 1), (0, 3), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)];
/// let graph = UnGraph::<i32, i32>::from_edges(&edge_list);
/// let mut res: Vec<Vec<NodeIndex>> = cycle_basis(&graph, Some(NodeIndex::new(0)));
/// ```
pub fn cycle_basis<G>(graph: G, root: Option<G::NodeId>) -> Vec<Vec<G::NodeId>>
where
    G: NodeCount,
    G: IntoNeighbors,
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
{
    let mut root_node = root;
    let mut graph_nodes: HashSet<G::NodeId> = graph.node_identifiers().collect();
    let mut cycles: Vec<Vec<G::NodeId>> = Vec::new();
    while !graph_nodes.is_empty() {
        let temp_value: G::NodeId;
        // If root_node is not set get an arbitrary node from the set of graph
        // nodes we've not "examined"
        let root_index = match root_node {
            Some(root_node) => root_node,
            None => {
                temp_value = *graph_nodes.iter().next().unwrap();
                graph_nodes.remove(&temp_value);
                temp_value
            }
        };
        // Stack (ie "pushdown list") of vertices already in the spanning tree
        let mut stack: Vec<G::NodeId> = vec![root_index];
        // Map of node index to predecessor node index
        let mut pred: HashMap<G::NodeId, G::NodeId> = HashMap::new();
        pred.insert(root_index, root_index);
        // Set of examined nodes during this iteration
        let mut used: HashMap<G::NodeId, HashSet<G::NodeId>> = HashMap::new();
        used.insert(root_index, HashSet::new());
        // Walk the spanning tree
        // Use the last element added so that cycles are easier to find
        while let Some(z) = stack.pop() {
            for neighbor in graph.neighbors(z) {
                // A new node was encountered:
                if !used.contains_key(&neighbor) {
                    pred.insert(neighbor, z);
                    stack.push(neighbor);
                    let mut temp_set: HashSet<G::NodeId> = HashSet::new();
                    temp_set.insert(z);
                    used.insert(neighbor, temp_set);
                // A self loop:
                } else if z == neighbor {
                    let cycle: Vec<G::NodeId> = vec![z];
                    cycles.push(cycle);
                // A cycle was found:
                } else if !used.get(&z).unwrap().contains(&neighbor) {
                    let pn = used.get(&neighbor).unwrap();
                    let mut cycle: Vec<G::NodeId> = vec![neighbor, z];
                    let mut p = pred.get(&z).unwrap();
                    while !pn.contains(p) {
                        cycle.push(*p);
                        p = pred.get(p).unwrap();
                    }
                    cycle.push(*p);
                    cycles.push(cycle);
                    let neighbor_set = used.get_mut(&neighbor).unwrap();
                    neighbor_set.insert(z);
                }
            }
        }
        let mut temp_hashset: HashSet<G::NodeId> = HashSet::new();
        for key in pred.keys() {
            temp_hashset.insert(*key);
        }
        graph_nodes = graph_nodes.difference(&temp_hashset).copied().collect();
        root_node = None;
    }
    cycles
}

#[cfg(test)]
mod tests {
    use crate::connectivity::cycle_basis;
    use petgraph::prelude::*;

    fn sorted_cycle(cycles: Vec<Vec<NodeIndex>>) -> Vec<Vec<usize>> {
        let mut sorted_cycles: Vec<Vec<usize>> = vec![];
        for cycle in cycles {
            let mut cycle: Vec<usize> = cycle.iter().map(|x| x.index()).collect();
            cycle.sort();
            sorted_cycles.push(cycle);
        }
        sorted_cycles.sort();
        sorted_cycles
    }

    #[test]
    fn test_cycle_basis_source() {
        let edge_list = vec![
            (0, 1),
            (0, 3),
            (0, 5),
            (0, 8),
            (1, 2),
            (1, 6),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
        ];
        let graph = UnGraph::<i32, i32>::from_edges(edge_list);
        let expected = vec![vec![0, 1, 2, 3], vec![0, 1, 6, 7, 8], vec![0, 3, 4, 5]];
        let res_0 = cycle_basis(&graph, Some(NodeIndex::new(0)));
        assert_eq!(sorted_cycle(res_0), expected);
        let res_1 = cycle_basis(&graph, Some(NodeIndex::new(1)));
        assert_eq!(sorted_cycle(res_1), expected);
        let res_9 = cycle_basis(&graph, Some(NodeIndex::new(9)));
        assert_eq!(sorted_cycle(res_9), expected);
    }

    #[test]
    fn test_self_loop() {
        let edge_list = vec![
            (0, 1),
            (0, 3),
            (0, 5),
            (0, 8),
            (1, 2),
            (1, 6),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
        ];
        let mut graph = UnGraph::<i32, i32>::from_edges(edge_list);
        graph.add_edge(NodeIndex::new(1), NodeIndex::new(1), 0);
        let res_0 = cycle_basis(&graph, Some(NodeIndex::new(0)));
        assert_eq!(
            sorted_cycle(res_0),
            vec![
                vec![0, 1, 2, 3],
                vec![0, 1, 6, 7, 8],
                vec![0, 3, 4, 5],
                vec![1]
            ]
        );
    }
}
