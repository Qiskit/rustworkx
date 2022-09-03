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
use petgraph::algo::kosaraju_scc;
use petgraph::algo::{is_cyclic_directed, Measure};
use petgraph::graph::IndexType;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::{
    EdgeRef, GraphBase, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable, Visitable,
};

use crate::dictmap::*;
use crate::distancemap::DistanceMap;

struct BellmanFordData<G, S>
where
    G: GraphBase,
{
    negative_cycle: bool,
    scores: S,
    predecessor: Vec<Option<G::NodeId>>,
}

/// Bellman-Ford shortest path algorithm with the SPFA heuristic.
///
/// Compute the length of the shortest path from `start` to every reachable
/// node. This implementation differs from petgraph's implementation because it
/// has an expected time complexity of O(kE) with 1 < k <= |V|. For random
/// graphs, we expect the SPFA heuristic to be very efficient.
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
/// negative cycles. If there are negative cycles, the result is [`None`].
/// # Example
/// ```rust
/// use rustworkx_core::petgraph::Graph;
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::dictmap::DictMap;
/// use rustworkx_core::shortest_path::bellman_ford;
/// use rustworkx_core::Result;
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
///     (e, f, -2),
///     (b, e, -1),
///     (f, g, 5),
///     (g, h, 2),
///     (h, e, -1)
/// ]);
/// // a ----> b ----> e ----> f
/// // ^       |       ^       |
/// // |       v       |       v
/// // d <---- c       h <---- g
///
/// let expected_res: DictMap<NodeIndex, i32> = [
///      (a, 3),
///      (b, 0),
///      (c, 1),
///      (d, 2),
///      (e, -1),
///      (f, -3),
///      (g, 2),
///      (h, 4)
///     ].iter().cloned().collect();
/// let res: Result<Option<DictMap<NodeIndex, i32>>> = bellman_ford(
///     &graph, b, |x| Ok(*x.weight()), None
/// );
/// assert_eq!(res.unwrap().unwrap(), expected_res);
/// // z is not inside res because there is not path from b to z.
/// ```
pub fn bellman_ford<G, F, K, E, S>(
    graph: G,
    start: G::NodeId,
    edge_cost: F,
    mut path: Option<&mut DictMap<G::NodeId, Vec<G::NodeId>>>,
) -> Result<Option<S>, E>
where
    G: IntoEdges + Visitable + NodeIndexable + NodeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + IndexType,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
    S: DistanceMap<G::NodeId, K>,
{
    let res: BellmanFordData<G, S> = inner_bellman_ford(graph, vec![start], edge_cost)?;

    let BellmanFordData {
        negative_cycle,
        scores,
        predecessor,
    } = res;

    // Shortest path is not defined
    if negative_cycle {
        return Ok(None);
    }

    // Build path from predecessors
    if path.is_some() {
        for node in graph.node_identifiers() {
            if scores.get_item(node).is_some() {
                let mut node_path = Vec::<G::NodeId>::new();
                let mut current_node = node;
                node_path.push(current_node);
                while let Some(pred_node) = predecessor[current_node.index()] {
                    current_node = pred_node;
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
/// Returns a vector of NodeIds if there are cycles, and [`None`] if there aren't.
/// The output is an arbitrary cycle with the property that the first node of the cycle is also the first
/// and last element of the vector.
/// # Example
/// ```rust
/// use rustworkx_core::petgraph::Graph;
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::dictmap::DictMap;
/// use rustworkx_core::shortest_path::negative_cycle_finder;
/// use rustworkx_core::Result;
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
    edge_cost: F,
) -> Result<Option<Vec<G::NodeId>>, E>
where
    G: IntoEdges + Visitable + NodeIndexable + NodeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + IndexType,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
{
    let res: BellmanFordData<G, Vec<Option<K>>> =
        inner_bellman_ford(graph, graph.node_identifiers(), edge_cost)?;

    let BellmanFordData {
        negative_cycle,
        scores: _,
        predecessor,
    } = res;

    // There is no cycle in the graph
    if !negative_cycle {
        return Ok(None);
    }

    Ok(Some(recover_negative_cycle_from_predecessors(
        graph,
        predecessor,
    )))
}

fn inner_bellman_ford<G, I, F, K, E, S>(
    graph: G,
    starts: I,
    mut edge_cost: F,
) -> Result<BellmanFordData<G, S>, E>
where
    G: IntoEdges + Visitable + NodeIndexable + NodeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + IndexType,
    F: FnMut(G::EdgeRef) -> Result<K, E>,
    K: Measure + Copy,
    S: DistanceMap<G::NodeId, K>,
    I: IntoIterator<Item = G::NodeId>,
{
    let node_count = graph.node_count();
    let mut in_queue = FixedBitSet::with_capacity(graph.node_bound());
    let mut scores: S = S::build(graph.node_bound());
    let mut predecessor: Vec<Option<G::NodeId>> = vec![None; graph.node_bound()];
    let mut visit_next = VecDeque::with_capacity(graph.node_bound());
    let zero_score = K::default();
    let mut relaxation_count: usize = 0;

    for start in starts {
        scores.put_item(start, zero_score);
        visit_next.push_back(start);
        in_queue.set(graph.to_index(start), true);
    }

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

                    if check_for_negative_cycle(graph, &predecessor) {
                        return Ok(BellmanFordData {
                            negative_cycle: true,
                            scores,
                            predecessor,
                        });
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

    Ok(BellmanFordData {
        negative_cycle: false,
        scores,
        predecessor,
    })
}

/// Function that checks if there is a cycle in the shortest path graph
///
/// The shortest path graph has N nodes and up to N edges. For a connected
/// graph without negative cycles, the graph is a DAG with N - 1 edges.
///
/// For graphs with negative cycles, the graph is cyclic and the cycle
/// in the shortest path graph is equivalent to the negative cycle in
/// the original graph.
fn check_for_negative_cycle<G>(graph: G, predecessor: &[Option<G::NodeId>]) -> bool
where
    G: GraphBase + NodeIndexable,
{
    let mut path_graph =
        StableDiGraph::<usize, ()>::with_capacity(predecessor.len(), predecessor.len());

    let node_indices: Vec<_> = (0..predecessor.len())
        .map(|x| path_graph.add_node(x))
        .collect();

    for (u, pred_u) in predecessor.iter().enumerate() {
        if let Some(v) = pred_u {
            path_graph.add_edge(node_indices[graph.to_index(*v)], node_indices[u], ());
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

    let node_indices: Vec<_> = (0..predecessor.len())
        .map(|x| path_graph.add_node(x))
        .collect();

    for (u, pred_u) in predecessor.iter().enumerate() {
        if let Some(v) = pred_u {
            path_graph.add_edge(node_indices[v.index()], node_indices[u], ());

            if v.index() == u {
                // Edge case: self-loop with negative edge
                return vec![*v, *v];
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

            let start_node = graph.from_index(component[0].index());
            let mut current_node = start_node;

            loop {
                cycle.push(current_node);
                current_node = predecessor[current_node.index()].unwrap();

                if current_node == start_node {
                    break;
                }
            }

            cycle.push(cycle[0]); // first node must be equal to last node
            cycle.reverse(); // the real path uses edges with (pred_u, u) not (u, pred_u)

            return cycle;
        }
    }

    // If we reach this line, it means the graph does not have a negative cycle
    Vec::new()
}

#[cfg(test)]
mod tests {
    use crate::dictmap::DictMap;
    use crate::shortest_path::negative_cycle_finder;
    use crate::shortest_path::{bellman_ford, dijkstra};
    use crate::Result;
    use petgraph::graph::NodeIndex;
    use petgraph::Graph;

    #[test]
    fn test_bell() {
        let mut graph = Graph::new_undirected();
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");
        let e = graph.add_node("E");
        let f = graph.add_node("F");
        graph.add_edge(a, b, 7);
        graph.add_edge(a, c, 9);
        graph.add_edge(a, d, 14);
        graph.add_edge(b, c, 10);
        graph.add_edge(d, c, 2);
        graph.add_edge(d, e, 9);
        graph.add_edge(b, f, 15);
        graph.add_edge(c, f, 11);
        graph.add_edge(e, f, 6);

        let res: Result<Option<DictMap<NodeIndex, i32>>> =
            bellman_ford(&graph, a, |e| Ok(*e.weight()), None);
        let res_dijk: Result<DictMap<NodeIndex, i32>> =
            dijkstra(&graph, a, None, |e| Ok(*e.weight()), None);

        assert_eq!(res.unwrap(), Some(res_dijk.unwrap()));
    }

    #[test]
    fn test_negative_cycle_finder_single_edge() {
        let mut g = Graph::new_undirected();
        let a = g.add_node(0);
        let b = g.add_node(1);
        g.add_edge(a, b, -1);

        let res: Result<Option<Vec<NodeIndex>>> = negative_cycle_finder(&g, |e| Ok(*e.weight()));

        assert_eq!(res.unwrap(), Some(vec![a, b, a]));
    }

    #[test]
    fn test_negative_cycle_finder_no_cycle() {
        let mut g = Graph::new_undirected();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);
        g.add_edge(a, b, 1);
        g.add_edge(b, c, 1);
        g.add_edge(a, c, 2);

        let res: Result<Option<Vec<NodeIndex>>> = negative_cycle_finder(&g, |e| Ok(*e.weight()));

        assert_eq!(res.unwrap(), None);
    }

    #[test]
    fn test_negative_cycle_finder_longer_cycle() {
        let mut g = Graph::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);
        let d = g.add_node(3);
        let e = g.add_node(4);
        g.add_edge(a, b, 1);
        g.add_edge(b, c, 1);
        g.add_edge(c, d, 1);
        g.add_edge(d, e, 1);
        g.add_edge(e, a, -5);

        let res: Result<Option<Vec<NodeIndex>>> = negative_cycle_finder(&g, |e| Ok(*e.weight()));

        assert_eq!(res.unwrap(), Some(vec![a, b, c, d, e, a]));
    }
}
