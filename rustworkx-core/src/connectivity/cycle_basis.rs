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

// This module was forked from petgraph:
//
// https://github.com/petgraph/petgraph/blob/9ff688872b467d3e1b5adef19f5c52f519d3279c/src/algo/simple_paths.rs
//
// to add support for returning all simple paths to a list of targets instead
// of just between a single node pair.

use hashbrown::HashSet;
use indexmap::map::Entry;
use indexmap::IndexSet;
use petgraph::visit::{IntoNeighborsDirected, NodeCount};
use petgraph::Direction::Outgoing;
use std::iter;
use std::{hash::Hash, iter::FromIterator};

use crate::dictmap::*;

/// Return a list of cycles which form a basis for cycles of a given PyGraph
///
/// A basis for cycles of a graph is a minimal collection of
/// cycles such that any cycle in the graph can be written
/// as a sum of cycles in the basis.  Here summation of cycles
/// is defined as the exclusive or of the edges.
///
/// This is adapted from algorithm CACM 491 [1]_.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges.
///     It may produce incorrect/unexpected results if the input graph has
///     parallel edges.
///
/// :param PyGraph graph: The graph to find the cycle basis in
/// :param int root: Optional index for starting node for basis
///
/// :returns: A list of cycle lists. Each list is a list of node ids which
///     forms a cycle (loop) in the input graph
/// :rtype: list
///
/// .. [1] Paton, K. An algorithm for finding a fundamental set of
///    cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.

/// Returns a dictionary with all simple paths from `from` node to all nodes in `to`, which contains at least `min_intermediate_nodes` nodes
/// and at most `max_intermediate_nodes`, if given, or limited by the graph's order otherwise. The simple path is a path without repetitions.
///
/// This algorithm is adapted from <https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.all_simple_paths.html>.
///
/// # Example
/// ```
/// use petgraph::prelude::*;
/// use hashbrown::HashSet;
/// use rustworkx_core::connectivity::all_simple_paths_multiple_targets;
///
/// let mut graph = DiGraph::<&str, i32>::new();
///
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let c = graph.add_node("c");
/// let d = graph.add_node("d");
///
/// graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (a, b, 1), (b, d, 1)]);
///
/// let mut to_set = HashSet::new();
/// to_set.insert(d);
///
/// let ways = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, None);
///
/// let d_path = ways.get(&d).unwrap();
/// assert_eq!(4, d_path.len());
/// ```
pub fn cycle_basis<G>(
    graph: G,
    root: Option(G::NodeId),
) -> Vec<Vec<G::NodeId>>
where
    G: NodeCount,
    G: IntoNeighbors,
    G::NodeId: Eq + Hash,
{
    let mut root_node = root;
    let mut graph_nodes: HashSet<NodeIndex> = graph.graph.node_indices().collect();
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    while !graph_nodes.is_empty() {
        let temp_value: NodeIndex;
        // If root_node is not set get an arbitrary node from the set of graph
        // nodes we've not "examined"
        let root_index = match root_node {
            Some(root_value) => NodeIndex::new(root_value),
            None => {
                temp_value = *graph_nodes.iter().next().unwrap();
                graph_nodes.remove(&temp_value);
                temp_value
            }
        };
        // Stack (ie "pushdown list") of vertices already in the spanning tree
        let mut stack: Vec<NodeIndex> = vec![root_index];
        // Map of node index to predecessor node index
        let mut pred: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        pred.insert(root_index, root_index);
        // Set of examined nodes during this iteration
        let mut used: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();
        used.insert(root_index, HashSet::new());
        // Walk the spanning tree
        while !stack.is_empty() {
            // Use the last element added so that cycles are easier to find
            let z = stack.pop().unwrap();
            for neighbor in graph.graph.neighbors(z) {
                // A new node was encountered:
                if !used.contains_key(&neighbor) {
                    pred.insert(neighbor, z);
                    stack.push(neighbor);
                    let mut temp_set: HashSet<NodeIndex> = HashSet::new();
                    temp_set.insert(z);
                    used.insert(neighbor, temp_set);
                // A self loop:
                } else if z == neighbor {
                    let cycle: Vec<usize> = vec![z.index()];
                    cycles.push(cycle);
                // A cycle was found:
                } else if !used.get(&z).unwrap().contains(&neighbor) {
                    let pn = used.get(&neighbor).unwrap();
                    let mut cycle: Vec<NodeIndex> = vec![neighbor, z];
                    let mut p = pred.get(&z).unwrap();
                    while !pn.contains(p) {
                        cycle.push(*p);
                        p = pred.get(p).unwrap();
                    }
                    cycle.push(*p);
                    cycles.push(cycle.iter().map(|x| x.index()).collect());
                    let neighbor_set = used.get_mut(&neighbor).unwrap();
                    neighbor_set.insert(z);
                }
            }
        }
        let mut temp_hashset: HashSet<NodeIndex> = HashSet::new();
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
    use crate::connectivity::all_simple_paths_multiple_targets;
    use hashbrown::HashSet;
    use petgraph::prelude::*;

    #[test]
    fn test_all_simple_paths() {
        // create a path graph
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);

        graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (d, e, 1)]);

        let mut to_set = HashSet::new();
        to_set.insert(d);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, None);

        assert_eq!(paths.get(&d).unwrap(), &vec![vec![a, b, c, d]]);
    }

    #[test]
    fn test_all_simple_paths_with_two_targets_emits_two_paths() {
        // create a path graph
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);

        graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (d, e, 1), (c, e, 1)]);

        let mut to_set = HashSet::new();
        to_set.insert(d);
        to_set.insert(e);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, None);

        assert_eq!(
            paths.get(&d).unwrap(),
            &vec![vec![a, b, c, e, d], vec![a, b, c, d]]
        );
        assert_eq!(
            paths.get(&e).unwrap(),
            &vec![vec![a, b, c, e], vec![a, b, c, d, e]]
        );
    }

    #[test]
    fn test_digraph_all_simple_paths_with_two_targets_emits_two_paths() {
        // create a path graph
        let mut graph = Graph::new();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);

        graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (d, e, 1), (c, e, 1)]);

        let mut to_set = HashSet::new();
        to_set.insert(d);
        to_set.insert(e);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, None);

        assert_eq!(paths.get(&d).unwrap(), &vec![vec![a, b, c, d]]);
        assert_eq!(
            paths.get(&e).unwrap(),
            &vec![vec![a, b, c, e], vec![a, b, c, d, e]]
        );
    }

    #[test]
    fn test_all_simple_paths_max_nodes() {
        // create a complete graph
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);

        graph.extend_with_edges(&[
            (a, b, 1),
            (a, c, 1),
            (a, d, 1),
            (a, e, 1),
            (b, c, 1),
            (b, d, 1),
            (b, e, 1),
            (c, d, 1),
            (c, e, 1),
            (d, e, 1),
        ]);

        let mut to_set = HashSet::new();
        to_set.insert(b);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, Some(0));

        assert_eq!(paths.get(&b).unwrap(), &vec![vec![a, b]]);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, Some(1));

        assert_eq!(
            paths.get(&b).unwrap(),
            &vec![vec![a, e, b], vec![a, d, b], vec![a, c, b], vec![a, b],]
        );
    }

    #[test]
    fn test_all_simple_paths_with_two_targets_max_nodes() {
        // create a path graph
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);

        graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (d, e, 1), (c, e, 1)]);

        let mut to_set = HashSet::new();
        to_set.insert(d);
        to_set.insert(e);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, Some(2));

        assert_eq!(paths.get(&d).unwrap(), &vec![vec![a, b, c, d]]);
        assert_eq!(paths.get(&e).unwrap(), &vec![vec![a, b, c, e]]);
    }

    #[test]
    fn test_digraph_all_simple_paths_with_two_targets_max_nodes() {
        // create a path graph
        let mut graph = Graph::new();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);

        graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (d, e, 1), (c, e, 1)]);

        let mut to_set = HashSet::new();
        to_set.insert(d);
        to_set.insert(e);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, Some(2));

        assert_eq!(paths.get(&d).unwrap(), &vec![vec![a, b, c, d]]);
        assert_eq!(paths.get(&e).unwrap(), &vec![vec![a, b, c, e]]);
    }

    #[test]
    fn test_all_simple_paths_with_two_targets_in_line_emits_two_paths() {
        // create a path graph
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);

        graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (d, e, 1)]);

        let mut to_set = HashSet::new();
        to_set.insert(c);
        to_set.insert(d);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, None);

        assert_eq!(paths.get(&c).unwrap(), &vec![vec![a, b, c]]);
        assert_eq!(paths.get(&d).unwrap(), &vec![vec![a, b, c, d]]);
    }

    #[test]
    fn test_all_simple_paths_min_nodes() {
        // create a cycle graph
        let mut graph = Graph::new();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);

        graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (d, a, 1), (b, d, 1)]);

        let mut to_set = HashSet::new();
        to_set.insert(d);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 2, None);

        assert_eq!(paths.get(&d).unwrap(), &vec![vec![a, b, c, d]]);
    }

    #[test]
    fn test_all_simple_paths_with_two_targets_min_nodes() {
        // create a cycle graph
        let mut graph = Graph::new();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);

        graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (d, a, 1), (b, d, 1)]);

        let mut to_set = HashSet::new();
        to_set.insert(c);
        to_set.insert(d);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 2, None);

        assert_eq!(paths.get(&c), None);
        assert_eq!(paths.get(&d).unwrap(), &vec![vec![a, b, c, d]]);
    }

    #[test]
    fn test_all_simple_paths_source_target() {
        // create a path graph
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);

        graph.extend_with_edges(&[(a, b, 1), (b, c, 1), (c, d, 1), (d, e, 1)]);

        let mut to_set = HashSet::new();
        to_set.insert(a);

        let paths = all_simple_paths_multiple_targets(&graph, a, &to_set, 0, None);

        assert_eq!(paths.get(&a), None);
    }

    #[test]
    fn test_all_simple_paths_on_non_trivial_graph() {
        // create a path graph
        let mut graph = Graph::new();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);
        let f = graph.add_node(5);

        graph.extend_with_edges(&[
            (a, b, 1),
            (b, c, 1),
            (c, d, 1),
            (d, e, 1),
            (e, f, 1),
            (a, f, 1),
            (b, f, 1),
            (b, d, 1),
            (f, e, 1),
            (e, c, 1),
            (e, d, 1),
        ]);

        let mut to_set = HashSet::new();
        to_set.insert(c);
        to_set.insert(d);

        let paths = all_simple_paths_multiple_targets(&graph, b, &to_set, 0, None);

        assert_eq!(
            paths.get(&c).unwrap(),
            &vec![vec![b, d, e, c], vec![b, f, e, c], vec![b, c]]
        );
        assert_eq!(
            paths.get(&d).unwrap(),
            &vec![
                vec![b, d],
                vec![b, f, e, d],
                vec![b, f, e, c, d],
                vec![b, c, d]
            ]
        );

        let paths = all_simple_paths_multiple_targets(&graph, b, &to_set, 1, None);

        assert_eq!(
            paths.get(&c).unwrap(),
            &vec![vec![b, d, e, c], vec![b, f, e, c]]
        );
        assert_eq!(
            paths.get(&d).unwrap(),
            &vec![vec![b, f, e, d], vec![b, f, e, c, d], vec![b, c, d]]
        );

        let paths = all_simple_paths_multiple_targets(&graph, b, &to_set, 0, Some(2));

        assert_eq!(
            paths.get(&c).unwrap(),
            &vec![vec![b, d, e, c], vec![b, f, e, c], vec![b, c]]
        );
        assert_eq!(
            paths.get(&d).unwrap(),
            &vec![vec![b, d], vec![b, f, e, d], vec![b, c, d]]
        );
    }
}
