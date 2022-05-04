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

use std::collections::VecDeque;
use std::hash::Hash;

use fixedbitset::FixedBitSet;
use hashbrown::HashMap;
use petgraph::algo::kosaraju_scc;
use petgraph::algo::{is_cyclic_directed, Measure};
use petgraph::graph::IndexType;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{
    EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable, Visitable,
};

use crate::dictmap::*;
use crate::distancemap::DistanceMap;

/// Bellman-Ford shortest path algorithm with the SPFA heuristic.
///
/// Compute the length of the shortest path from `start` to every reachable
/// node.
///
/// The graph should be [`Visitable`] and implement [`IntoEdges`]. The function
/// `edge_cost` should return the cost for a particular edge, which is used
/// to compute path costs. Edge costs can be negative, as long as there is no
/// negative cycle.
///
///
/// If `path` is not [`None`], then the algorithm will mutate the input
/// [`DictMap`] to insert an entry where the index is the dest node index
/// the value is a Vec of node indices of the path starting with `start` and
/// ending at the index.
///
/// Returns a [`DistanceMap`] that maps `NodeId` to path cost if there are no
/// negative cycles. If there are negative cycles, the Result is None.
/// # Example
/// ```rust
/// use retworkx_core::petgraph::Graph;
/// use retworkx_core::petgraph::prelude::*;
/// use retworkx_core::dictmap::DictMap;
/// use retworkx_core::shortest_path::bellman_ford;
/// use retworkx_core::Result;
///
/// let mut graph : Graph<(),(),Directed>= Graph::new();
/// let a = graph.add_node(()); // node with no weight
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// let d = graph.add_node(());
/// let e = graph.add_node(());
/// let f = graph.add_node(());
/// let g = graph.add_node(());
/// let h = graph.add_node(());
/// // z will be in another connected component
/// let z = graph.add_node(());
///
/// graph.extend_with_edges(&[
///     (a, b),
///     (b, c),
///     (c, d),
///     (d, a),
///     (e, f),
///     (b, e),
///     (f, g),
///     (g, h),
///     (h, e)
/// ]);
/// // a ----> b ----> e ----> f
/// // ^       |       ^       |
/// // |       v       |       v
/// // d <---- c       h <---- g
///
/// let expected_res: DictMap<NodeIndex, usize> = [
///      (a, 3),
///      (b, 0),
///      (c, 1),
///      (d, 2),
///      (e, 1),
///      (f, 2),
///      (g, 3),
///      (h, 4)
///     ].iter().cloned().collect();
/// let res: Result<Option<DictMap<NodeIndex, usize>>> = bellman_ford(
///     &graph, b, |_| Ok(1), None
/// );
/// assert_eq!(res.unwrap().unwrap(), expected_res);
/// // z is not inside res because there is not path from b to z.
/// ```
pub fn bellman_ford<G, F, K, E, S>(
    graph: G,
    start: G::NodeId,
    mut edge_cost: F,
    mut path: Option<&mut DictMap<G::NodeId, Vec<G::NodeId>>>,
) -> Result<Option<S>, E>
where
    G: IntoEdges + Visitable + NodeIndexable + NodeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + IndexType,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
    S: DistanceMap<G::NodeId, K>,
{
    let node_count = graph.node_count();
    let mut in_queue = FixedBitSet::with_capacity(graph.node_bound());
    let mut scores: S = S::build(graph.node_bound());
    let mut predecessor: Vec<Option<G::NodeId>> = vec![None; graph.node_bound()];
    let mut visit_next = VecDeque::with_capacity(graph.node_bound());
    let zero_score = K::default();
    let mut relaxation_count: usize = 0;

    scores.put_item(start, zero_score);
    visit_next.push_back(start);

    // SPFA heuristic: relax only nodes that need to be relaxed
    while let Some(node) = visit_next.pop_front() {
        in_queue.set(node.index(), false);
        let node_score = *scores.get_item(node).unwrap();

        for edge in graph.edges(node) {
            let next = edge.target();
            let current_score = scores.get_item(next);
            let cost = edge_cost(edge)?;
            let next_score = node_score + cost;

            if current_score.is_none() || next_score < *current_score.unwrap() {
                scores.put_item(next, next_score);
                predecessor[next.index()] = Some(node);
                relaxation_count += 1;

                // We do the negative cycle check every O(|V|)
                // iterations to amortize the cost, as it costs O(|V|) to run it
                if relaxation_count == node_count {
                    relaxation_count = 0;

                    if check_for_negative_cycle(
                        predecessor.iter().map(|x| x.map(|y| y.index())).collect(),
                    ) {
                        return Ok(None);
                    }
                }

                // Node needs to be relaxed on a future iteration
                if !in_queue.contains(next.index()) {
                    visit_next.push_back(next);
                    in_queue.set(next.index(), true);
                }
            }
        }
    }

    // Build path from predecessors
    if path.is_some() {
        for node in graph.node_identifiers() {
            if scores.get_item(node).is_some() {
                let mut node_path = Vec::<G::NodeId>::new();
                let mut current_node = node;
                node_path.push(current_node);
                while predecessor[current_node.index()].is_some() {
                    current_node = predecessor[current_node.index()].unwrap();
                    node_path.push(current_node);
                }
                node_path.reverse();
                path.as_mut().unwrap().insert(node, node_path);
            }
        }
    }

    Ok(Some(scores))
}

/// Finds an arbitrary negative cycle in a graph using the Bellman-Ford
/// algorithm with the SPFA heuristic.
///
/// Returns a vector of NodeIds if there are cycles, and None if there aren't.
/// The output is an arbitrary cycle with the property that the first node of the cycle is also the first
/// and last element of the vector.
/// # Example
/// ```rust
/// use retworkx_core::petgraph::Graph;
/// use retworkx_core::petgraph::prelude::*;
/// use retworkx_core::dictmap::DictMap;
/// use retworkx_core::shortest_path::negative_cycle_finder;
/// use retworkx_core::Result;
///
/// let mut graph : Graph<(),i32,Directed>= Graph::new();
/// let a = graph.add_node(()); // node with no weight
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// let d = graph.add_node(());
/// let e = graph.add_node(());
/// let f = graph.add_node(());
/// let g = graph.add_node(());
/// let h = graph.add_node(());
/// // z will be in another connected component
/// let z = graph.add_node(());
///
/// graph.extend_with_edges(&[
///     (a, b, 1),
///     (b, c, 1),
///     (c, d, 1),
///     (d, a, 1),
///     (e, f, 1),
///     (b, e, 1),
///     (f, g, -4),
///     (g, h, 1),
///     (h, e, 1)
/// ]);
/// // a ----> b ----> e ----> f
/// // ^       |       ^       |
/// // |       v       |       v
/// // d <---- c       h <---- g
///
/// let res: Result<Option<Vec<NodeIndex>>> = negative_cycle_finder(
///     &graph, |x| Ok(*x.weight())
/// );
/// assert_eq!(res.unwrap().unwrap().len() - 1, 4);
/// // the first/last node in the cycle appears twice
/// ```
pub fn negative_cycle_finder<G, F, K, E>(
    graph: G,
    mut edge_cost: F,
) -> Result<Option<Vec<G::NodeId>>, E>
where
    G: IntoEdges + Visitable + NodeIndexable + NodeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + IndexType,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
{
    let node_count = graph.node_count();
    let mut in_queue = FixedBitSet::with_capacity(graph.node_bound());
    let zero_score = K::default();
    let mut scores: Vec<K> = vec![zero_score; graph.node_bound()];
    let mut predecessor: Vec<Option<G::NodeId>> = vec![None; graph.node_bound()];
    let mut visit_next = VecDeque::with_capacity(graph.node_bound());
    let mut relaxation_count: usize = 0;

    // For detecting cycles, this is equivalent to connecting all nodes
    // to a source with weight equal to zero. This avoids having to loop
    // through components to find the cycle.
    for node in graph.node_identifiers() {
        visit_next.push_back(node);
        in_queue.set(node.index(), true);
    }

    // SPFA heuristic: relax only nodes that need to be relaxed
    while let Some(node) = visit_next.pop_front() {
        in_queue.set(node.index(), false);
        let node_score = scores[node.index()];

        for edge in graph.edges(node) {
            let next = edge.target();
            let current_score = scores[next.index()];
            let cost = edge_cost(edge)?;
            let next_score = node_score + cost;

            if next_score < current_score {
                scores[next.index()] = next_score;
                predecessor[next.index()] = Some(node);
                relaxation_count += 1;

                // We do the negative cycle check every O(|V|)
                // iterations to amortize the cost, as it costs O(|V|) to run it
                if relaxation_count == node_count {
                    relaxation_count = 0;

                    if check_for_negative_cycle(
                        predecessor.iter().map(|x| x.map(|y| y.index())).collect(),
                    ) {
                        return Ok(Some(recover_negative_cycle_from_predecessors(
                            graph,
                            predecessor,
                        )));
                    }
                }

                // Node needs to be relaxed on a future iteration
                if !in_queue.contains(next.index()) {
                    visit_next.push_back(next);
                    in_queue.set(next.index(), true);
                }
            }
        }
    }

    Ok(None)
}

/// Function that checks if there is a cycle in the shortest path graph
///
/// The shortest path graph has N nodes and up to N edges. For a connected
/// graph without negative cycles, the graph is a DAG with N - 1 edges.
///
/// For graphs with negative cycles, the graph is cyclic and the cycle
/// in the shortest path graph is equivalent to the negative cycle in
/// the original graph.
fn check_for_negative_cycle(predecessor: Vec<Option<usize>>) -> bool {
    let mut path_graph =
        StableDiGraph::<usize, ()>::with_capacity(predecessor.len(), predecessor.len());

    let node_indices: Vec<_> = (0..predecessor.len())
        .map(|x| path_graph.add_node(x))
        .collect();

    for (u, pred_u) in predecessor.into_iter().enumerate() {
        if let Some(v) = pred_u {
            path_graph.add_edge(node_indices[v], node_indices[u], ());
        }
    }

    is_cyclic_directed(&path_graph)
}

/// Returns the cycle found in the shortest path graph
/// using Strongly Connected Components to detect the cycle.
fn recover_negative_cycle_from_predecessors<G>(
    graph: G,
    predecessor: Vec<Option<G::NodeId>>,
) -> Vec<G::NodeId>
where
    G: IntoEdges + Visitable + NodeIndexable + NodeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + IndexType,
{
    // We build the graph with just the edges in the shortest path graph
    let mut path_graph =
        StableDiGraph::<usize, ()>::with_capacity(predecessor.len(), predecessor.len());

    let original_node_indices: HashMap<usize, G::NodeId> =
        graph.node_identifiers().map(|x| (x.index(), x)).collect();

    let node_indices: Vec<_> = (0..predecessor.len())
        .map(|x| path_graph.add_node(x))
        .collect();

    for (u, pred_u) in predecessor.into_iter().enumerate() {
        if let Some(v) = pred_u {
            path_graph.add_edge(node_indices[v.index()], node_indices[u], ());

            if v.index() == u {
                // Edge case: self-loop with negative edge
                return vec![v, v];
            }
        }
    }

    // Then we find the strongly connected components
    let sccs = kosaraju_scc(&path_graph);

    for component in sccs {
        // Because there are N nodes and N edges, the SCC
        // that consists of more than one node *is* the negative cycle
        if component.len() >= 2 {
            let mut cycle: Vec<G::NodeId> = Vec::with_capacity(component.len() + 1);

            for node in component {
                if let Some(original_node) = original_node_indices.get(&node.index()) {
                    cycle.push(*original_node);
                }
            }

            cycle.push(cycle[0]); // first node must be equal to last node

            return cycle;
        }
    }

    // If we reach this line, it means the graph does not have a negative cycle
    Vec::new()
}
