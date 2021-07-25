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

#![allow(clippy::float_cmp)]

use super::{digraph, graph, NullGraph};

use hashbrown::{HashMap, HashSet};
use std::collections::BTreeSet;

use pyo3::prelude::*;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::visit::NodeCount;

use crate::iterators::EdgeList;

/// Return a list of cycles which form a basis for cycles of a given PyGraph
///
/// A basis for cycles of a graph is a minimal collection of
/// cycles such that any cycle in the graph can be written
/// as a sum of cycles in the basis.  Here summation of cycles
/// is defined as the exclusive or of the edges.
///
/// This is adapted from algorithm CACM 491 [1]_.
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
#[pyfunction]
#[pyo3(text_signature = "(graph, /, root=None)")]
pub fn cycle_basis(
    graph: &graph::PyGraph,
    root: Option<usize>,
) -> Vec<Vec<usize>> {
    let mut root_node = root;
    let mut graph_nodes: HashSet<NodeIndex> =
        graph.graph.node_indices().collect();
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

/// Compute the strongly connected components for a directed graph
///
/// This function is implemented using Kosaraju's algorithm
///
/// :param PyDiGraph graph: The input graph to find the strongly connected
///     components for.
///
/// :return: A list of list of node ids for strongly connected components
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn strongly_connected_components(
    graph: &digraph::PyDiGraph,
) -> Vec<Vec<usize>> {
    algo::kosaraju_scc(graph)
        .iter()
        .map(|x| x.iter().map(|id| id.index()).collect())
        .collect()
}

/// Return the first cycle encountered during DFS of a given PyDiGraph,
/// empty list is returned if no cycle is found
///
/// :param PyDiGraph graph: The graph to find the cycle in
/// :param int source: Optional index to find a cycle for. If not specified an
///     arbitrary node will be selected from the graph.
///
/// :returns: A list describing the cycle. The index of node ids which
///     forms a cycle (loop) in the input graph
/// :rtype: EdgeList
#[pyfunction]
#[pyo3(text_signature = "(graph, /, source=None)")]
pub fn digraph_find_cycle(
    graph: &digraph::PyDiGraph,
    source: Option<usize>,
) -> EdgeList {
    let mut graph_nodes: HashSet<NodeIndex> =
        graph.graph.node_indices().collect();
    let mut cycle: Vec<(usize, usize)> =
        Vec::with_capacity(graph.graph.edge_count());
    let temp_value: NodeIndex;
    // If source is not set get an arbitrary node from the set of graph
    // nodes we've not "examined"
    let source_index = match source {
        Some(source_value) => NodeIndex::new(source_value),
        None => {
            temp_value = *graph_nodes.iter().next().unwrap();
            graph_nodes.remove(&temp_value);
            temp_value
        }
    };

    // Stack (ie "pushdown list") of vertices already in the spanning tree
    let mut stack: Vec<NodeIndex> = vec![source_index];
    // map to store parent of a node
    let mut pred: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    // a node is in the visiting set if at least one of its child is unexamined
    let mut visiting = HashSet::new();
    // a node is in visited set if all of its children have been examined
    let mut visited = HashSet::new();
    while !stack.is_empty() {
        let mut z = *stack.last().unwrap();
        visiting.insert(z);

        let children = graph
            .graph
            .neighbors_directed(z, petgraph::Direction::Outgoing);

        for child in children {
            //cycle is found
            if visiting.contains(&child) {
                cycle.push((z.index(), child.index()));
                //backtrack
                loop {
                    if z == child {
                        cycle.reverse();
                        break;
                    }
                    cycle.push((pred[&z].index(), z.index()));
                    z = pred[&z];
                }
                return EdgeList { edges: cycle };
            }
            //if an unexplored node is encountered
            if !visited.contains(&child) {
                stack.push(child);
                pred.insert(child, z);
            }
        }

        let top = *stack.last().unwrap();
        //if no further children and explored, move to visited
        if top.index() == z.index() {
            stack.pop();
            visiting.remove(&z);
            visited.insert(z);
        }
    }
    EdgeList { edges: cycle }
}

/// Find the number of weakly connected components in a DAG.
///
/// :param PyDiGraph graph: The graph to find the number of weakly connected
///     components on
///
/// :returns: The number of weakly connected components in the DAG
/// :rtype: int
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
fn number_weakly_connected_components(graph: &digraph::PyDiGraph) -> usize {
    algo::connected_components(graph)
}

/// Find the weakly connected components in a directed graph
///
/// :param PyDiGraph graph: The graph to find the weakly connected components
///     in
///
/// :returns: A list of sets where each set it a weakly connected component of
///     the graph
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn weakly_connected_components(
    graph: &digraph::PyDiGraph,
) -> Vec<BTreeSet<usize>> {
    let mut seen: HashSet<NodeIndex> =
        HashSet::with_capacity(graph.node_count());
    let mut out_vec: Vec<BTreeSet<usize>> = Vec::new();
    for node in graph.graph.node_indices() {
        if !seen.contains(&node) {
            // BFS node generator
            let mut component_set: BTreeSet<usize> = BTreeSet::new();
            let mut bfs_seen: HashSet<NodeIndex> = HashSet::new();
            let mut next_level: HashSet<NodeIndex> = HashSet::new();
            next_level.insert(node);
            while !next_level.is_empty() {
                let this_level = next_level;
                next_level = HashSet::new();
                for bfs_node in this_level {
                    if !bfs_seen.contains(&bfs_node) {
                        component_set.insert(bfs_node.index());
                        bfs_seen.insert(bfs_node);
                        for neighbor in
                            graph.graph.neighbors_undirected(bfs_node)
                        {
                            next_level.insert(neighbor);
                        }
                    }
                }
            }
            out_vec.push(component_set);
            seen.extend(bfs_seen);
        }
    }
    out_vec
}

/// Check if the graph is weakly connected
///
/// :param PyDiGraph graph: The graph to check if it is weakly connected
///
/// :returns: Whether the graph is weakly connected or not
/// :rtype: bool
///
/// :raises NullGraph: If an empty graph is passed in
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn is_weakly_connected(graph: &digraph::PyDiGraph) -> PyResult<bool> {
    if graph.graph.node_count() == 0 {
        return Err(NullGraph::new_err("Invalid operation on a NullGraph"));
    }
    Ok(weakly_connected_components(graph)[0].len() == graph.graph.node_count())
}
