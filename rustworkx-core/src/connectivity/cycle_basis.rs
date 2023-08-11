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
use petgraph::visit::{EdgeRef, IntoEdges, IntoNeighbors, IntoNodeIdentifiers, NodeCount};
use std::hash::Hash;

/// Inner private function for `cycle_basis` and `cycle_basis_edges`.
/// Returns a list of cycles which forms a basis of cycles of a given
/// graph.
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
/// Arguments:
///
/// * `graph` - The graph in which to find the basis.
/// * `root` - Optional node index for starting the basis search. If not
///     specified, an arbitrary node is chosen.
/// * `edges` - bool for when the user requests the edges instead
///     of the nodes of the cycles.

fn inner_cycle_basis<G>(
    graph: G,
    root: Option<G::NodeId>,
    edges: bool,
) -> EdgesOrNodes<G::NodeId, G::EdgeId>
where
    G: NodeCount,
    G: IntoNeighbors,
    G: IntoEdges,
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    G::EdgeId: Eq + Hash,
{
    let mut root_node: Option<G::NodeId> = root;
    let mut graph_nodes: HashSet<G::NodeId> = graph.node_identifiers().collect();
    let mut cycles_edges: Vec<Vec<G::EdgeId>> = Vec::new();
    let mut cycles_nodes: Vec<Vec<G::NodeId>> = Vec::new();

    /// Method used to retrieve all the edges between an origin node and a target node.
    fn get_edge_between<G>(orig_graph: G, origin: G::NodeId, target: G::NodeId) -> G::EdgeId
    where
        G: IntoEdges,
    {
        orig_graph
            .edges(origin)
            .filter(|edge: &G::EdgeRef| edge.target() == target)
            .map(|edge: G::EdgeRef| edge.id())
            .next()
            .unwrap()
    }

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
        while let Some(z) = stack.pop() {
            // Use the last element added so that cycles are easier to find
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
                    if edges {
                        let cycle_edge: Vec<G::EdgeId> = vec![get_edge_between(graph, z, z)];
                        cycles_edges.push(cycle_edge);
                    } else {
                        let cycle: Vec<G::NodeId> = vec![z];
                        cycles_nodes.push(cycle);
                    }
                // A cycle was found:
                } else if !used.get(&z).unwrap().contains(&neighbor) {
                    let pn = used.get(&neighbor).unwrap();
                    let mut p = pred.get(&z).unwrap();
                    if edges {
                        let mut cycle: Vec<G::EdgeId> = Vec::new();
                        // Retreive all edges from z to neighbor and push to cycle
                        cycle.push(get_edge_between(graph, z, neighbor));

                        // Make last p_node == z
                        let mut prev_p: &G::NodeId = &z;
                        // While p is in the neighborhood of neighbor
                        while !pn.contains(p) {
                            // Retrieve all edges from prev_p to p and vice versa append to cycle
                            cycle.push(get_edge_between(graph, *prev_p, *p));
                            // Update prev_p to p
                            prev_p = p;
                            // Retreive a new predecessor node from p and replace p
                            p = pred.get(p).unwrap();
                        }
                        // When loop ends add remaining edges from prev_p to p.
                        cycle.push(get_edge_between(graph, *prev_p, *p));
                        // Also retreive all edges between the last p and neighbor
                        cycle.push(get_edge_between(graph, *p, neighbor));
                        // Once all edges within cycle have been found, push to cycle list.
                        cycles_edges.push(cycle);
                    } else {
                        // Append neighbor and z to cycle.
                        let mut cycle: Vec<G::NodeId> = vec![neighbor, z];
                        while !pn.contains(p) {
                            cycle.push(*p);
                            p = pred.get(p).unwrap();
                        }
                        cycle.push(*p);
                        cycles_nodes.push(cycle);
                    }
                    let neighbor_set: &mut HashSet<G::NodeId> = used.get_mut(&neighbor).unwrap();
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
    if edges {
        EdgesOrNodes::Edges(cycles_edges)
    } else {
        EdgesOrNodes::Nodes(cycles_nodes)
    }
}

/// Enum for custom return types of `cycle_basis()`.
enum EdgesOrNodes<N, E> {
    Nodes(Vec<Vec<N>>),
    Edges(Vec<Vec<E>>),
}
/// Functions used to unwrap the desired datatype of `EdgesOrNodes`.
impl<N, E> EdgesOrNodes<N, E> {
    fn unwrap_nodes(self) -> Vec<Vec<N>> {
        match self {
            Self::Nodes(x) => x,
            Self::Edges(_) => unreachable!(
                "Function should only return instances of {}.",
                std::any::type_name::<N>()
            ),
        }
    }
    fn unwrap_edges(self) -> Vec<Vec<E>> {
        match self {
            Self::Edges(x) => x,
            Self::Nodes(_) => unreachable!(
                "Function should only return instances of {}.",
                std::any::type_name::<E>()
            ),
        }
    }
}

/// Returns lists of `NodeIndex` representing cycles which form
/// a basis for cycles of a given graph.
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
    G: IntoEdges,
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    G::EdgeId: Eq + Hash,
{
    inner_cycle_basis(graph, root, false).unwrap_nodes()
}

/// Returns lists of `EdgeIndex` representing cycles which form
/// a basis for cycles of a given graph.
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
/// use rustworkx_core::connectivity::cycle_basis_edges;
///
/// let edge_list = [(0, 1), (0, 3), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)];
/// let graph = UnGraph::<i32, i32>::from_edges(&edge_list);
/// let mut res: Vec<Vec<EdgeIndex>> = cycle_basis_edges(&graph, Some(NodeIndex::new(0)));
/// ```
pub fn cycle_basis_edges<G>(graph: G, root: Option<G::NodeId>) -> Vec<Vec<G::EdgeId>>
where
    G: NodeCount,
    G: IntoEdges,
    G: IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    G::EdgeId: Eq + Hash,
{
    inner_cycle_basis(graph, root, true).unwrap_edges()
}

#[cfg(test)]
mod tests {
    use crate::connectivity::cycle_basis;
    use crate::connectivity::cycle_basis_edges;
    use petgraph::prelude::*;
    use petgraph::stable_graph::GraphIndex;

    fn sorted_cycle<T>(cycles: Vec<Vec<T>>) -> Vec<Vec<usize>>
    where
        T: GraphIndex,
    {
        let mut sorted_cycles: Vec<Vec<usize>> = vec![];
        for cycle in cycles {
            let mut cycle: Vec<usize> = cycle.iter().map(|x: &T| x.index()).collect();
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
        let graph = UnGraph::<i32, i32>::from_edges(&edge_list);
        let expected = vec![vec![0, 1, 2, 3], vec![0, 1, 6, 7, 8], vec![0, 3, 4, 5]];
        let res_0 = cycle_basis(&graph, Some(NodeIndex::new(0)));
        assert_eq!(sorted_cycle(res_0), expected);
        let res_1 = cycle_basis(&graph, Some(NodeIndex::new(1)));
        assert_eq!(sorted_cycle(res_1), expected);
        let res_9 = cycle_basis(&graph, Some(NodeIndex::new(9)));
        assert_eq!(sorted_cycle(res_9), expected);
    }

    #[test]
    fn test_cycle_edge_basis_source() {
        let edge_list = vec![
            (0, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 5),
            (5, 6),
            (3, 6),
            (3, 4),
        ];
        let graph = UnGraph::<i32, i32>::from_edges(&edge_list);
        let expected = vec![vec![0], vec![3, 4, 5, 6]];
        let res_0 = cycle_basis_edges(&graph, Some(NodeIndex::new(0)));
        assert_eq!(sorted_cycle(res_0), expected);
        let res_1 = cycle_basis_edges(&graph, Some(NodeIndex::new(2)));
        assert_eq!(sorted_cycle(res_1), expected);
        let res_9 = cycle_basis_edges(&graph, Some(NodeIndex::new(6)));
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
        let mut graph = UnGraph::<i32, i32>::from_edges(&edge_list);
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

    #[test]
    fn test_self_loop_edges() {
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
        let mut graph = UnGraph::<i32, i32>::from_edges(&edge_list);
        graph.add_edge(NodeIndex::new(1), NodeIndex::new(1), 0);
        let res_0 = cycle_basis_edges(&graph, Some(NodeIndex::new(0)));
        assert_eq!(
            sorted_cycle(res_0),
            vec![
                vec![0, 1, 4, 6],
                vec![0, 3, 5, 9, 10],
                vec![1, 2, 7, 8],
                vec![12],
            ]
        );
    }
}
