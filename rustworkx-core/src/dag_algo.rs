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

use std::cmp::{Eq, Ordering};
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::hash::Hash;

use hashbrown::{HashMap, HashSet};
use std::fmt::Debug;
use std::mem::swap;

use petgraph::algo;
use petgraph::data::DataMap;
use petgraph::visit::{
    EdgeRef, GraphBase, GraphProp, IntoEdgesDirected, IntoNeighborsDirected, IntoNodeIdentifiers,
    NodeCount, NodeIndexable, Visitable,
};
use petgraph::Directed;

use num_traits::{Num, Zero};

use crate::err::LayersError;

/// Return a pair of [`petgraph::Direction`] values corresponding to the "forwards" and "backwards"
/// direction of graph traversal, based on whether the graph is being traved forwards (following
/// the edges) or backward (reversing along edges).  The order of returns is (forwards, backwards).
#[inline(always)]
pub fn traversal_directions(reverse: bool) -> (petgraph::Direction, petgraph::Direction) {
    if reverse {
        (petgraph::Direction::Outgoing, petgraph::Direction::Incoming)
    } else {
        (petgraph::Direction::Incoming, petgraph::Direction::Outgoing)
    }
}

/// An error enumeration returned by topological sort functions.
#[derive(Debug, PartialEq, Eq)]
pub enum TopologicalSortError<E: Error> {
    CycleOrBadInitialState,
    KeyError(E),
}

impl<E: Error> Display for TopologicalSortError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TopologicalSortError::CycleOrBadInitialState => {
                write!(f, "At least one initial node is reachable from another")
            }
            TopologicalSortError::KeyError(ref e) => {
                write!(f, "The key callback failed with: {:?}", e)
            }
        }
    }
}

impl<E: Error> Error for TopologicalSortError<E> {}

/// Get the lexicographical topological sorted nodes from the provided DAG.
///
/// This function returns a list of nodes in a graph lexicographically
/// topologically sorted, using the provided key function as a tie-breaker.
/// A topological sort is a linear ordering of vertices such that for every
/// directed edge from node :math:`u` to node :math:`v`, :math:`u` comes before
/// :math:`v` in the ordering.  If ``reverse`` is set to ``true``, the edges
/// are treated as if they pointed in the opposite direction.
///
/// # Arguments:
///
/// * `dag`: The DAG to get the topological sorted nodes from
/// * `key`: A function that gets passed a single argument, the node id from
///     `dag` and is expected to return a key which will be used for
///     resolving ties in the sorting order.
/// * `reverse`: If `false`, perform a regular topological ordering.  If `true`,
///     return the lexicographical topological order that would have been found
///     if all the edges in the graph were reversed.  This does not affect the
///     comparisons from the `key`.
/// * `initial`: If given, the initial node indices to start the topological
///     ordering from.  If not given, the topological ordering will certainly contain every node in
///     the graph.  If given, only the `initial` nodes and nodes that are dominated by the
///     `initial` set will be in the ordering.  Notably, any node that has a natural in degree of
///     zero will not be in the output ordering if `initial` is given and the zero-in-degree node
///     is not in it.  It is not supported to give an `initial` set where the nodes have even
///     a partial topological order between themselves and `None` will be returned in this case
///
/// # Returns
///
/// `Vec<G::NodeId>` representing the topological ordering of nodes or a
/// `TopologicalSortError` if there is an error.
///
/// # Example
///
/// ```rust
/// use std::convert::Infallible;
///
/// use rustworkx_core::dag_algo::lexicographical_topological_sort;
/// use rustworkx_core::petgraph::stable_graph::{StableDiGraph, NodeIndex};
///
/// let mut graph: StableDiGraph<u8, ()> = StableDiGraph::new();
/// let mut nodes: Vec<NodeIndex> = Vec::new();
/// for weight in 0..9 {
///     nodes.push(graph.add_node(weight));
/// }
/// let edges = [
///         (nodes[0], nodes[1]),
///         (nodes[0], nodes[2]),
///         (nodes[1], nodes[3]),
///         (nodes[2], nodes[4]),
///         (nodes[3], nodes[4]),
///         (nodes[4], nodes[5]),
///         (nodes[5], nodes[6]),
///         (nodes[4], nodes[7]),
///         (nodes[6], nodes[8]),
///         (nodes[7], nodes[8]),
/// ];
/// for (source, target) in edges {
///     graph.add_edge(source, target, ());
/// }
/// let sort_fn = |index: NodeIndex| -> Result<String, Infallible> { Ok(graph[index].to_string()) };
/// let initial = [nodes[6], nodes[7]];
/// let result = lexicographical_topological_sort(&graph, sort_fn, true, Some(&initial));
/// let expected = vec![
///     nodes[6],
///     nodes[5],
///     nodes[7],
///     nodes[4],
///     nodes[2],
///     nodes[3],
///     nodes[1],
///     nodes[0]
/// ];
/// assert_eq!(result, Ok(expected));
///
/// ```
pub fn lexicographical_topological_sort<G, F, K, E>(
    dag: G,
    mut key: F,
    reverse: bool,
    initial: Option<&[G::NodeId]>,
) -> Result<Vec<G::NodeId>, TopologicalSortError<E>>
where
    G: GraphProp<EdgeType = Directed>
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + IntoEdgesDirected
        + NodeCount,
    <G as GraphBase>::NodeId: Hash + Eq + Ord,
    F: FnMut(G::NodeId) -> Result<K, E>,
    K: Ord,
    E: Error,
{
    // HashMap of node_index indegree
    let node_count = dag.node_count();
    let (in_dir, out_dir) = traversal_directions(reverse);

    #[derive(Clone, Eq, PartialEq)]
    struct State<K: Ord, N: Eq + PartialOrd> {
        key: K,
        node: N,
    }

    impl<K: Ord, N: Eq + Ord> Ord for State<K, N> {
        fn cmp(&self, other: &State<K, N>) -> Ordering {
            // Notice that we flip the ordering on costs.
            // In case of a tie, we compare positions - this step is necessary
            // to make implementations of `PartialEq` and `Ord` consistent.
            other
                .key
                .cmp(&self.key)
                .then_with(|| other.node.cmp(&self.node))
        }
    }

    // `PartialOrd` needs to be implemented as well.
    impl<K: Ord, N: Eq + Ord> PartialOrd for State<K, N> {
        fn partial_cmp(&self, other: &State<K, N>) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut in_degree_map: HashMap<G::NodeId, usize> = HashMap::with_capacity(node_count);
    if let Some(initial) = initial {
        // In this case, we don't iterate through all the nodes in the graph, and most nodes aren't
        // in `in_degree_map`; we'll fill in the relevant edge counts lazily.
        for node in initial {
            in_degree_map.insert(*node, 0);
        }
    } else {
        for node in dag.node_identifiers() {
            in_degree_map.insert(node, dag.edges_directed(node, in_dir).count());
        }
    }

    let mut zero_indegree = BinaryHeap::with_capacity(node_count);
    for (node, degree) in in_degree_map.iter() {
        if *degree == 0 {
            let map_key = key(*node).map_err(|e| TopologicalSortError::KeyError(e))?;
            zero_indegree.push(State {
                key: map_key,
                node: *node,
            });
        }
    }
    let mut out_list: Vec<G::NodeId> = Vec::with_capacity(node_count);
    while let Some(State { node, .. }) = zero_indegree.pop() {
        let neighbors = dag.neighbors_directed(node, out_dir);
        for child in neighbors {
            let child_degree = in_degree_map
                .entry(child)
                .or_insert_with(|| dag.edges_directed(child, in_dir).count());
            if *child_degree == 0 {
                return Err(TopologicalSortError::CycleOrBadInitialState);
            } else if *child_degree == 1 {
                let map_key = key(child).map_err(|e| TopologicalSortError::KeyError(e))?;
                zero_indegree.push(State {
                    key: map_key,
                    node: child,
                });
                in_degree_map.remove(&child);
            } else {
                *child_degree -= 1;
            }
        }
        out_list.push(node)
    }
    Ok(out_list)
}

// Type aliases for readability
type NodeId<G> = <G as GraphBase>::NodeId;
type LongestPathResult<G, T, E> = Result<Option<(Vec<NodeId<G>>, T)>, E>;

/// Calculates the longest path in a directed acyclic graph (DAG).
///
/// This function computes the longest path by weight in a given DAG. It will return the longest path
/// along with its total weight, or `None` if the graph contains cycles which make the longest path
/// computation undefined.
///
/// # Arguments
/// * `graph`: Reference to a directed graph.
/// * `weight_fn` - An input callable that will be passed the `EdgeRef` for each edge in the graph.
///  The callable should return the weight of the edge as `Result<T, E>`. The weight must be a type that implements
/// `Num`, `Zero`, `PartialOrd`, and `Copy`.
///
/// # Type Parameters
/// * `G`: Type of the graph. Must be a directed graph.
/// * `F`: Type of the weight function.
/// * `T`: The type of the edge weight. Must implement `Num`, `Zero`, `PartialOrd`, and `Copy`.
/// * `E`: The type of the error that the weight function can return.
///
/// # Returns
/// * `None` if the graph contains a cycle.
/// * `Some((Vec<NodeId<G>>, T))` representing the longest path as a sequence of nodes and its total weight.
/// * `Err(E)` if there is an error computing the weight of any edge.
///
/// # Example
/// ```
/// use petgraph::graph::DiGraph;
/// use petgraph::Directed;
/// use rustworkx_core::dag_algo::longest_path;
///
/// let mut graph: DiGraph<(), i32> = DiGraph::new();
/// let n0 = graph.add_node(());
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// graph.add_edge(n0, n1, 1);
/// graph.add_edge(n0, n2, 3);
/// graph.add_edge(n1, n2, 1);
///
/// let weight_fn = |edge: petgraph::graph::EdgeReference<i32>| Ok::<i32, &str>(*edge.weight());
/// let result = longest_path(&graph, weight_fn).unwrap();
/// assert_eq!(result, Some((vec![n0, n2], 3)));
/// ```
pub fn longest_path<G, F, T, E>(graph: G, mut weight_fn: F) -> LongestPathResult<G, T, E>
where
    G: GraphProp<EdgeType = Directed> + IntoNodeIdentifiers + IntoEdgesDirected + Visitable,
    F: FnMut(G::EdgeRef) -> Result<T, E>,
    T: Num + Zero + PartialOrd + Copy,
    <G as GraphBase>::NodeId: Hash + Eq + PartialOrd,
{
    let mut path: Vec<NodeId<G>> = Vec::new();
    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_) => return Ok(None), // Return None if the graph contains a cycle
    };

    if nodes.is_empty() {
        return Ok(Some((path, T::zero())));
    }

    let mut dist: HashMap<G::NodeId, (T, G::NodeId)> = HashMap::with_capacity(nodes.len()); // Stores the distance and the previous node

    // Iterate over nodes in topological order
    for node in nodes {
        let parents = graph.edges_directed(node, petgraph::Direction::Incoming);
        let mut incoming_path: Vec<(T, G::NodeId)> = Vec::new(); // Stores the distance and the previous node for each parent
        for p_edge in parents {
            let p_node = p_edge.source();
            let weight: T = weight_fn(p_edge)?;
            let length = dist[&p_node].0 + weight;
            incoming_path.push((length, p_node));
        }
        // Determine the maximum distance and corresponding parent node
        let max_path: (T, G::NodeId) = incoming_path
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap_or((T::zero(), node)); // If there are no incoming edges, the distance is zero

        // Store the maximum distance and the corresponding parent node for the current node
        dist.insert(node, max_path);
    }
    let (first, _) = dist
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let mut v = *first;
    let mut u: Option<G::NodeId> = None;
    // Backtrack from this node to find the path
    while u.map_or(true, |u| u != v) {
        path.push(v);
        u = Some(v);
        v = dist[&v].1;
    }
    path.reverse(); // Reverse the path to get the correct order
    let path_weight = dist[first].0; // The total weight of the longest path

    Ok(Some((path, path_weight)))
}

/// Return an iterator of graph layers
///
/// A layer is a subgraph whose nodes are disjoint, i.e.,
/// a layer has depth 1. The layers are constructed using a greedy algorithm.
///
/// Arguments:
///
/// * `graph` - The graph to get the layers from
/// * `first_layer` - A list of node ids for the first layer. This
///     will be the first layer in the output
///
/// Will `panic!` if a provided node is not in the graph.
/// ```
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::dag_algo::layers;
/// use rustworkx_core::dictmap::*;
///
/// let edge_list = vec![
///  (0, 1),
///  (1, 2),
///  (2, 3),
///  (3, 4),
/// ];
///
/// let graph = DiGraph::<u32, u32>::from_edges(&edge_list);
/// let layers: Vec<Vec<NodeIndex>> = layers(&graph, vec![0.into(),]).map(|layer| layer.unwrap()).collect();
/// let expected_layers: Vec<Vec<NodeIndex>> = vec![
///     vec![0.into(),],
///     vec![1.into(),],
///     vec![2.into(),],
///     vec![3.into(),],
///     vec![4.into()]
/// ];
/// assert_eq!(layers, expected_layers)
/// ```
pub fn layers<G>(
    graph: G,
    first_layer: Vec<G::NodeId>,
) -> impl Iterator<Item = Result<Vec<G::NodeId>, LayersError>>
where
    G: NodeIndexable // Used in from_index and to_index.
        + IntoNodeIdentifiers // Used for .node_identifiers
        + IntoNeighborsDirected // Used for .neighbors_directed
        + IntoEdgesDirected, // Used for .edged_directed
    <G as GraphBase>::NodeId: Debug + Copy + Eq + Hash,
{
    LayersIter {
        graph,
        cur_layer: first_layer,
        next_layer: vec![],
        predecessor_count: HashMap::new(),
        first_iter: true,
        cycle_check: HashSet::default(),
    }
}

#[derive(Debug, Clone)]
struct LayersIter<G, N> {
    graph: G,
    cur_layer: Vec<N>,
    next_layer: Vec<N>,
    predecessor_count: HashMap<N, usize>,
    first_iter: bool,
    cycle_check: HashSet<N>, // TODO: Figure out why some cycles cannot be detected
}

impl<G, N> Iterator for LayersIter<G, N>
where
    G: NodeIndexable // Used in from_index and to_index.
        + IntoNodeIdentifiers // Used for .node_identifiers
        + IntoNeighborsDirected // Used for .neighbors_directed
        + IntoEdgesDirected // Used for .edged_directed
        + GraphBase<NodeId = N>,
    N: Debug + Copy + Eq + Hash,
{
    type Item = Result<Vec<N>, LayersError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.first_iter {
            self.first_iter = false;
            for node in &self.cur_layer {
                if self.graph.to_index(*node) >= self.graph.node_bound() {
                    panic!("Node {:#?} is not present in the graph.", node);
                }
                if self.cycle_check.contains(node) {
                    return Some(Err(LayersError(format!(
                        "An invalid first layer was provided: {:#?} appears more than once.",
                        node
                    ))));
                }
                self.cycle_check.insert(*node);
            }
            Some(Ok(self.cur_layer.clone()))
        } else if self.cur_layer.is_empty() {
            None
        } else {
            for node in &self.cur_layer {
                if self.graph.to_index(*node) >= self.graph.node_bound() {
                    panic!("Node {:#?} is not present in the graph.", node);
                }
                let children = self
                    .graph
                    .neighbors_directed(*node, petgraph::Direction::Outgoing);
                let mut used_indices: HashSet<G::NodeId> = HashSet::new();
                for succ in children {
                    // Skip duplicate successors
                    if used_indices.contains(&succ) {
                        continue;
                    }
                    used_indices.insert(succ);
                    let mut multiplicity: usize = 0;
                    let raw_edges: G::EdgesDirected = self
                        .graph
                        .edges_directed(*node, petgraph::Direction::Outgoing);
                    for edge in raw_edges {
                        if edge.target() == succ {
                            multiplicity += 1;
                        }
                    }
                    self.predecessor_count
                        .entry(succ)
                        .and_modify(|e| *e -= multiplicity)
                        .or_insert(
                            // Get the number of incoming edges to the successor
                            self.graph
                                .edges_directed(succ, petgraph::Direction::Incoming)
                                .count()
                                - multiplicity,
                        );
                    if *self.predecessor_count.get(&succ).unwrap() == 0 {
                        if self.cycle_check.contains(&succ) {
                            return Some(Err(LayersError("The provided graph contains a cycle or an invalid first layer was provided.".to_string())));
                        }
                        self.next_layer.push(succ);
                        self.cycle_check.insert(succ);
                        self.predecessor_count.remove(&succ);
                    }
                }
            }
            swap(&mut self.cur_layer, &mut self.next_layer);
            self.next_layer.clear();
            if self.cur_layer.is_empty() {
                None
            } else {
                Some(Ok(self.cur_layer.clone()))
            }
        }
    }
}

/// Collect runs that match a filter function given edge colors.
///
/// A bicolor run is a list of groups of nodes connected by edges of exactly
/// two colors. In addition, all nodes in the group must match the given
/// condition. Each node in the graph can appear in only a single group
/// in the bicolor run.
///
/// # Arguments:
///
/// * `graph`: The DAG to find bicolor runs in
/// * `filter_fn`: The filter function to use for matching nodes. It takes
///     in one argument, the node data payload/weight object, and will return a
///     boolean whether the node matches the conditions or not.
///     If it returns ``true``, it will continue the bicolor chain.
///     If it returns ``false``, it will stop the bicolor chain.
///     If it returns ``None`` it will skip that node.
/// * `color_fn`: The function that gives the color of the edge. It takes
///     in one argument, the edge data payload/weight object, and will
///     return a non-negative integer, the edge color. If the color is None,
///     the edge is ignored.
///
/// # Returns:
///
/// * `Vec<Vec<G::NodeId>>`: a list of groups with exactly two edge colors, where each group
///     is a list of node data payload/weight for the nodes in the bicolor run
/// * `None` if a cycle is found in the graph
/// * Raises an error if found computing the bicolor runs
///
/// # Example:
///
/// ```rust
/// use rustworkx_core::dag_algo::collect_bicolor_runs;
/// use petgraph::graph::{DiGraph, NodeIndex};
/// use std::convert::Infallible;
/// use std::error::Error;
///
/// let mut graph = DiGraph::new();
/// let n0 = graph.add_node(0);
/// let n1 = graph.add_node(0);
/// let n2 = graph.add_node(1);
/// let n3 = graph.add_node(1);
/// let n4 = graph.add_node(0);
/// let n5 = graph.add_node(0);
/// graph.add_edge(n0, n2, 0);
/// graph.add_edge(n1, n2, 1);
/// graph.add_edge(n2, n3, 0);
/// graph.add_edge(n2, n3, 1);
/// graph.add_edge(n3, n4, 0);
/// graph.add_edge(n3, n5, 1);
///
/// let filter_fn = |node_id| -> Result<Option<bool>, Infallible> {
///     Ok(Some(*graph.node_weight(node_id).unwrap() > 0))
/// };
///
/// let color_fn = |edge_id| -> Result<Option<usize>, Infallible> {
///     Ok(Some(*graph.edge_weight(edge_id).unwrap() as usize))
/// };
///
/// let result = collect_bicolor_runs(&graph, filter_fn, color_fn).unwrap();
/// let expected: Vec<Vec<NodeIndex>> = vec![vec![n2, n3]];
/// assert_eq!(result, Some(expected))
/// ```
pub fn collect_bicolor_runs<G, F, C, E>(
    graph: G,
    filter_fn: F,
    color_fn: C,
) -> Result<Option<Vec<Vec<G::NodeId>>>, E>
where
    F: Fn(<G as GraphBase>::NodeId) -> Result<Option<bool>, E>,
    C: Fn(<G as GraphBase>::EdgeId) -> Result<Option<usize>, E>,
    G: IntoNodeIdentifiers // Used in toposort
        + IntoNeighborsDirected // Used in toposort
        + IntoEdgesDirected // Used for .edges_directed
        + Visitable // Used in toposort
        + DataMap, // Used for .node_weight
    <G as GraphBase>::NodeId: Eq + Hash,
    <G as GraphBase>::EdgeId: Eq + Hash,
{
    let mut pending_list: Vec<Vec<G::NodeId>> = Vec::new();
    let mut block_id: Vec<Option<usize>> = Vec::new();
    let mut block_list: Vec<Vec<G::NodeId>> = Vec::new();

    let nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_) => return Ok(None), // Return None if the graph contains a cycle
    };

    // Utility for ensuring pending_list has the color index
    macro_rules! ensure_vector_has_index {
        ($pending_list: expr, $block_id: expr, $color: expr) => {
            if $color >= $pending_list.len() {
                $pending_list.resize($color + 1, Vec::new());
                $block_id.resize($color + 1, None);
            }
        };
    }

    for node in nodes {
        if let Some(is_match) = filter_fn(node)? {
            let raw_edges = graph.edges_directed(node, petgraph::Direction::Outgoing);

            // Remove all edges that yield errors from color_fn
            let colors = raw_edges
                .map(|edge| color_fn(edge.id()))
                .collect::<Result<Vec<Option<usize>>, _>>()?;

            // Remove null edges from color_fn
            let colors = colors.into_iter().flatten().collect::<Vec<usize>>();

            if colors.len() <= 2 && is_match {
                if colors.len() == 1 {
                    let c0 = colors[0];
                    ensure_vector_has_index!(pending_list, block_id, c0);
                    if let Some(c0_block_id) = block_id[c0] {
                        block_list[c0_block_id].push(node);
                    } else {
                        pending_list[c0].push(node);
                    }
                } else if colors.len() == 2 {
                    let c0 = colors[0];
                    let c1 = colors[1];
                    ensure_vector_has_index!(pending_list, block_id, c0);
                    ensure_vector_has_index!(pending_list, block_id, c1);

                    if block_id[c0].is_some()
                        && block_id[c1].is_some()
                        && block_id[c0] == block_id[c1]
                    {
                        block_list[block_id[c0].unwrap_or_default()].push(node);
                    } else {
                        let mut new_block: Vec<G::NodeId> =
                            Vec::with_capacity(pending_list[c0].len() + pending_list[c1].len() + 1);

                        // Clears pending lits and add to new block
                        new_block.append(&mut pending_list[c0]);
                        new_block.append(&mut pending_list[c1]);

                        new_block.push(node);

                        // Create new block, assign its id to color pair
                        block_id[c0] = Some(block_list.len());
                        block_id[c1] = Some(block_list.len());
                        block_list.push(new_block);
                    }
                }
            } else {
                for color in colors {
                    ensure_vector_has_index!(pending_list, block_id, color);
                    if let Some(color_block_id) = block_id[color] {
                        block_list[color_block_id].append(&mut pending_list[color]);
                    }
                    block_id[color] = None;
                    pending_list[color].clear();
                }
            }
        }
    }

    Ok(Some(block_list))
}

/// Collect runs that match a filter function
///
/// A run is a path of nodes where there is only a single successor and all
/// nodes in the path match the given condition. Each node in the graph can
/// appear in only a single run.
///
/// # Arguments:
///
/// * `graph`: The DAG to collect runs from
/// * `include_node_fn`: A filter function used for matching nodes. It takes
///     in one argument, the node data payload/weight object, and returns a
///     boolean whether the node matches the conditions or not.
///     If it returns ``false``, the node will be skipped, cutting the run it's part of.
///
/// # Returns:
///
/// * An Iterator object for extracting the runs one by one. Each run is of type `Result<Vec<G::NodeId>>`.
/// * `None` if a cycle is found in the graph.
///
/// # Example
///
/// ```rust
/// use petgraph::graph::DiGraph;
/// use rustworkx_core::dag_algo::collect_runs;
///
/// let mut graph: DiGraph<i32, ()> = DiGraph::new();
/// let n1 = graph.add_node(-1);
/// let n2 = graph.add_node(2);
/// let n3 = graph.add_node(3);
/// graph.add_edge(n1, n2, ());
/// graph.add_edge(n1, n3, ());
///
/// let positive_payload = |n| -> Result<bool, ()> {Ok(*graph.node_weight(n).expect("i32") > 0)};
/// let mut runs = collect_runs(&graph, positive_payload).expect("Some");
///
/// assert_eq!(runs.next(), Some(Ok(vec![n3])));
/// assert_eq!(runs.next(), Some(Ok(vec![n2])));
/// assert_eq!(runs.next(), None);
/// ```
///
pub fn collect_runs<G, F, E>(
    graph: G,
    include_node_fn: F,
) -> Option<impl Iterator<Item = Result<Vec<G::NodeId>, E>>>
where
    G: GraphProp<EdgeType = Directed>
        + IntoNeighborsDirected
        + IntoNodeIdentifiers
        + Visitable
        + NodeCount,
    F: Fn(G::NodeId) -> Result<bool, E>,
    <G as GraphBase>::NodeId: Hash + Eq,
{
    let mut nodes = match algo::toposort(graph, None) {
        Ok(nodes) => nodes,
        Err(_) => return None,
    };

    nodes.reverse(); // reversing so that pop() in Runs::next obeys the topological order

    let runs = Runs {
        graph,
        seen: HashSet::with_capacity(nodes.len()),
        sorted_nodes: nodes,
        include_node_fn,
    };

    Some(runs)
}

/// Auxiliary struct to make the output of [`collect_runs`] iteratable
///
/// If the filtering function passed to [`collect_runs`] returns an error, it is propagated
/// through `next` as `Err`. In this case the run in which the error occurred will be skipped
/// but the iterator can be used further until consumed.
///
struct Runs<G, F, E>
where
    G: GraphProp<EdgeType = Directed>
        + IntoNeighborsDirected
        + IntoNodeIdentifiers
        + Visitable
        + NodeCount,
    F: Fn(G::NodeId) -> Result<bool, E>,
{
    graph: G,
    sorted_nodes: Vec<G::NodeId>, // topologically-sorted nodes
    seen: HashSet<G::NodeId>,
    include_node_fn: F, // filtering function of the nodes
}

impl<G, F, E> Iterator for Runs<G, F, E>
where
    G: GraphProp<EdgeType = Directed>
        + IntoNeighborsDirected
        + IntoNodeIdentifiers
        + Visitable
        + NodeCount,
    F: Fn(G::NodeId) -> Result<bool, E>,
    <G as GraphBase>::NodeId: Hash + Eq,
{
    // This is a run, wrapped in Result for catching filter function errors
    type Item = Result<Vec<G::NodeId>, E>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.sorted_nodes.pop() {
            if self.seen.contains(&node) {
                continue;
            }
            self.seen.insert(node);

            match (self.include_node_fn)(node) {
                Ok(false) => continue,
                Err(e) => return Some(Err(e)),
                _ => (),
            }

            let mut run: Vec<G::NodeId> = vec![node];
            loop {
                let mut successors: Vec<G::NodeId> = self
                    .graph
                    .neighbors_directed(*run.last().unwrap(), petgraph::Direction::Outgoing)
                    .collect();
                successors.dedup();

                if successors.len() != 1 || self.seen.contains(&successors[0]) {
                    break;
                }

                self.seen.insert(successors[0]);

                match (self.include_node_fn)(successors[0]) {
                    Ok(false) => continue,
                    Err(e) => return Some(Err(e)),
                    _ => (),
                }

                run.push(successors[0]);
            }

            if !run.is_empty() {
                return Some(Ok(run));
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Lower bound is 0 in case all remaining nodes are filtered out
        // Upper bound is the remaining unprocessed nodes (some of which may be seen already), potentially all resulting with singleton runs
        (0, Some(self.sorted_nodes.len()))
    }
}

// Tests for longest_path
#[cfg(test)]
mod test_longest_path {
    use super::*;
    use petgraph::graph::DiGraph;
    use petgraph::stable_graph::StableDiGraph;

    #[test]
    fn test_empty_graph() {
        let graph: DiGraph<(), ()> = DiGraph::new();
        let weight_fn = |_: petgraph::graph::EdgeReference<()>| Ok::<i32, &str>(0);
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Ok(Some((vec![], 0))));
    }

    #[test]
    fn test_single_node_graph() {
        let mut graph: DiGraph<(), ()> = DiGraph::new();
        let n0 = graph.add_node(());
        let weight_fn = |_: petgraph::graph::EdgeReference<()>| Ok::<i32, &str>(0);
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Ok(Some((vec![n0], 0))));
    }

    #[test]
    fn test_dag_with_multiple_paths() {
        let mut graph: DiGraph<(), i32> = DiGraph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());
        let n5 = graph.add_node(());
        graph.add_edge(n0, n1, 3);
        graph.add_edge(n0, n2, 2);
        graph.add_edge(n1, n2, 1);
        graph.add_edge(n1, n3, 4);
        graph.add_edge(n2, n3, 2);
        graph.add_edge(n3, n4, 2);
        graph.add_edge(n2, n5, 1);
        graph.add_edge(n4, n5, 3);
        let weight_fn = |edge: petgraph::graph::EdgeReference<i32>| Ok::<i32, &str>(*edge.weight());
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Ok(Some((vec![n0, n1, n3, n4, n5], 12))));
    }

    #[test]
    fn test_graph_with_cycle() {
        let mut graph: DiGraph<(), i32> = DiGraph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, 1);
        graph.add_edge(n1, n0, 1); // Creates a cycle

        let weight_fn = |edge: petgraph::graph::EdgeReference<i32>| Ok::<i32, &str>(*edge.weight());
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Ok(None));
    }

    #[test]
    fn test_negative_weights() {
        let mut graph: DiGraph<(), i32> = DiGraph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, -1);
        graph.add_edge(n0, n2, 2);
        graph.add_edge(n1, n2, -2);
        let weight_fn = |edge: petgraph::graph::EdgeReference<i32>| Ok::<i32, &str>(*edge.weight());
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Ok(Some((vec![n0, n2], 2))));
    }

    #[test]
    fn test_longest_path_in_stable_digraph() {
        let mut graph: StableDiGraph<(), i32> = StableDiGraph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, 1);
        graph.add_edge(n0, n2, 3);
        graph.add_edge(n1, n2, 1);
        let weight_fn =
            |edge: petgraph::stable_graph::EdgeReference<'_, i32>| Ok::<i32, &str>(*edge.weight());
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Ok(Some((vec![n0, n2], 3))));
    }

    #[test]
    fn test_error_handling() {
        let mut graph: DiGraph<(), i32> = DiGraph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, 1);
        graph.add_edge(n0, n2, 2);
        graph.add_edge(n1, n2, 1);
        let weight_fn = |edge: petgraph::graph::EdgeReference<i32>| {
            if *edge.weight() == 2 {
                Err("Error: edge weight is 2")
            } else {
                Ok::<i32, &str>(*edge.weight())
            }
        };
        let result = longest_path(&graph, weight_fn);
        assert_eq!(result, Err("Error: edge weight is 2"));
    }
}

// Tests for lexicographical_topological_sort
#[cfg(test)]
mod test_lexicographical_topological_sort {
    use super::*;
    use petgraph::graph::{DiGraph, NodeIndex};
    use petgraph::stable_graph::StableDiGraph;
    use std::convert::Infallible;

    #[test]
    fn test_empty_graph() {
        let graph: DiGraph<(), ()> = DiGraph::new();
        let sort_fn = |_: NodeIndex| -> Result<String, Infallible> { Ok("a".to_string()) };
        let result = lexicographical_topological_sort(&graph, sort_fn, false, None);
        assert_eq!(result, Ok(vec![]));
    }

    #[test]
    fn test_empty_stable_graph() {
        let graph: StableDiGraph<(), ()> = StableDiGraph::new();
        let sort_fn = |_: NodeIndex| -> Result<String, Infallible> { Ok("a".to_string()) };
        let result = lexicographical_topological_sort(&graph, sort_fn, false, None);
        assert_eq!(result, Ok(vec![]));
    }

    #[test]
    fn test_simple_layer() {
        let mut graph: DiGraph<String, ()> = DiGraph::new();
        let mut nodes: Vec<NodeIndex> = Vec::new();
        nodes.push(graph.add_node("a".to_string()));
        for i in 0..5 {
            nodes.push(graph.add_node(i.to_string()));
        }
        nodes.push(graph.add_node("A parent".to_string()));
        for (source, target) in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (6, 3)] {
            graph.add_edge(nodes[source], nodes[target], ());
        }
        let sort_fn = |index: NodeIndex| -> Result<String, Infallible> { Ok(graph[index].clone()) };
        let result = lexicographical_topological_sort(&graph, sort_fn, false, None);
        assert_eq!(
            result,
            Ok(vec![
                NodeIndex::new(6),
                NodeIndex::new(0),
                NodeIndex::new(1),
                NodeIndex::new(2),
                NodeIndex::new(3),
                NodeIndex::new(4),
                NodeIndex::new(5)
            ])
        )
    }

    #[test]
    fn test_simple_layer_stable() {
        let mut graph: StableDiGraph<String, ()> = StableDiGraph::new();
        let mut nodes: Vec<NodeIndex> = Vec::new();
        nodes.push(graph.add_node("a".to_string()));
        for i in 0..5 {
            nodes.push(graph.add_node(i.to_string()));
        }
        nodes.push(graph.add_node("A parent".to_string()));
        for (source, target) in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (6, 3)] {
            graph.add_edge(nodes[source], nodes[target], ());
        }
        let sort_fn = |index: NodeIndex| -> Result<String, Infallible> { Ok(graph[index].clone()) };
        let result = lexicographical_topological_sort(&graph, sort_fn, false, None);
        assert_eq!(
            result,
            Ok(vec![
                NodeIndex::new(6),
                NodeIndex::new(0),
                NodeIndex::new(1),
                NodeIndex::new(2),
                NodeIndex::new(3),
                NodeIndex::new(4),
                NodeIndex::new(5)
            ])
        )
    }

    #[test]
    fn test_reverse_graph() {
        let mut graph: DiGraph<String, ()> = DiGraph::new();
        let mut nodes: Vec<NodeIndex> = Vec::new();
        for weight in ["a", "b", "c", "d", "e", "f"] {
            nodes.push(graph.add_node(weight.to_string()));
        }
        let edges = [
            (nodes[0], nodes[1]),
            (nodes[0], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[2], nodes[3]),
            (nodes[1], nodes[4]),
            (nodes[2], nodes[5]),
        ];

        for (source, target) in edges {
            graph.add_edge(source, target, ());
        }
        let sort_fn = |index: NodeIndex| -> Result<String, Infallible> { Ok(graph[index].clone()) };
        let result = lexicographical_topological_sort(&graph, sort_fn, true, None);
        graph.reverse();
        let sort_fn = |index: NodeIndex| -> Result<String, Infallible> { Ok(graph[index].clone()) };
        let expected = lexicographical_topological_sort(&graph, sort_fn, false, None);
        assert_eq!(result, expected,)
    }

    #[test]
    fn test_reverse_graph_stable() {
        let mut graph: StableDiGraph<String, ()> = StableDiGraph::new();
        let mut nodes: Vec<NodeIndex> = Vec::new();
        for weight in ["a", "b", "c", "d", "e", "f"] {
            nodes.push(graph.add_node(weight.to_string()));
        }
        let edges = [
            (nodes[0], nodes[1]),
            (nodes[0], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[2], nodes[3]),
            (nodes[1], nodes[4]),
            (nodes[2], nodes[5]),
        ];

        for (source, target) in edges {
            graph.add_edge(source, target, ());
        }
        let sort_fn = |index: NodeIndex| -> Result<String, Infallible> { Ok(graph[index].clone()) };
        let result = lexicographical_topological_sort(&graph, sort_fn, true, None);
        graph.reverse();
        let sort_fn = |index: NodeIndex| -> Result<String, Infallible> { Ok(graph[index].clone()) };
        let expected = lexicographical_topological_sort(&graph, sort_fn, false, None);
        assert_eq!(result, expected,)
    }

    #[test]
    fn test_initial() {
        let mut graph: StableDiGraph<u8, ()> = StableDiGraph::new();
        let mut nodes: Vec<NodeIndex> = Vec::new();
        for weight in 0..9 {
            nodes.push(graph.add_node(weight));
        }
        let edges = [
            (nodes[0], nodes[1]),
            (nodes[0], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[2], nodes[4]),
            (nodes[3], nodes[4]),
            (nodes[4], nodes[5]),
            (nodes[5], nodes[6]),
            (nodes[4], nodes[7]),
            (nodes[6], nodes[8]),
            (nodes[7], nodes[8]),
        ];
        for (source, target) in edges {
            graph.add_edge(source, target, ());
        }
        let sort_fn =
            |index: NodeIndex| -> Result<String, Infallible> { Ok(graph[index].to_string()) };
        let initial = [nodes[6], nodes[7]];
        let result = lexicographical_topological_sort(&graph, sort_fn, false, Some(&initial));
        assert_eq!(result, Ok(vec![nodes[6], nodes[7], nodes[8]]));
        let initial = [nodes[0]];
        let result = lexicographical_topological_sort(&graph, sort_fn, false, Some(&initial));
        assert_eq!(
            result,
            lexicographical_topological_sort(&graph, sort_fn, false, None)
        );
        let initial = [nodes[7]];
        let result = lexicographical_topological_sort(&graph, sort_fn, false, Some(&initial));
        assert_eq!(result, Ok(vec![nodes[7]]));
    }
}

#[cfg(test)]
mod test_layers {
    use super::*;
    use petgraph::{
        graph::{DiGraph, NodeIndex},
        stable_graph::StableDiGraph,
    };

    #[test]
    fn test_empty_graph() {
        let graph: DiGraph<(), ()> = DiGraph::new();
        let result: Vec<Vec<NodeIndex>> = layers(&graph, vec![]).flatten().collect();
        assert_eq!(result, vec![vec![]]);
    }

    #[test]
    fn test_empty_stable_graph() {
        let graph: StableDiGraph<(), ()> = StableDiGraph::new();
        let result: Vec<Vec<NodeIndex>> = layers(&graph, vec![]).flatten().collect();
        assert_eq!(result, vec![vec![]]);
    }

    #[test]
    fn test_simple_layer() {
        let mut graph: DiGraph<String, ()> = DiGraph::new();
        let mut nodes: Vec<NodeIndex> = Vec::new();
        nodes.push(graph.add_node("a".to_string()));
        for i in 0..5 {
            nodes.push(graph.add_node(i.to_string()));
        }
        nodes.push(graph.add_node("A parent".to_string()));
        for (source, target) in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (6, 3)] {
            graph.add_edge(nodes[source], nodes[target], ());
        }
        let expected: Vec<Vec<NodeIndex>> = vec![
            vec![0.into(), 6.into()],
            vec![5.into(), 4.into(), 2.into(), 1.into(), 3.into()],
        ];
        let result: Vec<Vec<NodeIndex>> =
            layers(&graph, vec![0.into(), 6.into()]).flatten().collect();
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_missing_node() {
        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let graph = DiGraph::<u32, u32>::from_edges(&edge_list);
        layers(&graph, vec![4.into(), 5.into()]).for_each(|layer| match layer {
            Err(e) => panic!("{}", e.0),
            Ok(layer) => drop(layer),
        });
    }

    #[test]
    fn test_dag_with_multiple_paths() {
        let mut graph: DiGraph<(), ()> = DiGraph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());
        let n5 = graph.add_node(());
        graph.add_edge(n0, n1, ());
        graph.add_edge(n0, n2, ());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());
        graph.add_edge(n2, n3, ());
        graph.add_edge(n3, n4, ());
        graph.add_edge(n2, n5, ());
        graph.add_edge(n4, n5, ());

        let result: Vec<Vec<NodeIndex>> = layers(&graph, vec![0.into()]).flatten().collect();
        assert_eq!(
            result,
            vec![vec![n0], vec![n1], vec![n2], vec![n3], vec![n4], vec![n5]]
        );
    }

    #[test]
    #[should_panic]
    fn test_graph_with_cycle() {
        let mut graph: DiGraph<(), i32> = DiGraph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, 1);
        graph.add_edge(n1, n0, 1);

        layers(&graph, vec![0.into()]).for_each(|layer| match layer {
            Err(e) => panic!("{}", e.0),
            Ok(layer) => drop(layer),
        });
    }
}

// Tests for collect_bicolor_runs
#[cfg(test)]
mod test_collect_bicolor_runs {

    use super::*;
    use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
    use std::error::Error;

    #[test]
    fn test_cycle() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(0);
        let n2 = graph.add_node(0);
        graph.add_edge(n0, n1, 1);
        graph.add_edge(n1, n2, 1);
        graph.add_edge(n2, n0, 1);

        let test_filter_fn =
            |_node_id: NodeIndex| -> Result<Option<bool>, Box<dyn Error>> { Ok(Some(true)) };
        let test_color_fn =
            |_edge_id: EdgeIndex| -> Result<Option<usize>, Box<dyn Error>> { Ok(Some(1)) };
        let result = match collect_bicolor_runs(&graph, test_filter_fn, test_color_fn) {
            Ok(Some(_value)) => "Not None",
            Ok(None) => "None",
            Err(_) => "Error",
        };
        assert_eq!(result, "None")
    }

    #[test]
    fn test_filter_function_inner_exception() {
        let mut graph = DiGraph::new();
        graph.add_node(0);

        let fail_function = |node_id: NodeIndex| -> Result<Option<bool>, Box<dyn Error>> {
            let node_weight: &i32 = graph.node_weight(node_id).expect("Invalid NodeId");
            if *node_weight > 0 {
                Ok(Some(true))
            } else {
                Err(Box::from("Failed!"))
            }
        };
        let test_color_fn = |edge_id: EdgeIndex| -> Result<Option<usize>, Box<dyn Error>> {
            let edge_weight: &i32 = graph.edge_weight(edge_id).expect("Invalid Edge");
            Ok(Some(*edge_weight as usize))
        };
        let result = match collect_bicolor_runs(&graph, fail_function, test_color_fn) {
            Ok(Some(_value)) => "Not None",
            Ok(None) => "None",
            Err(_) => "Error",
        };
        assert_eq!(result, "Error")
    }

    #[test]
    fn test_empty() {
        let graph = DiGraph::new();
        let test_filter_fn = |node_id: NodeIndex| -> Result<Option<bool>, Box<dyn Error>> {
            let node_weight: &i32 = graph.node_weight(node_id).expect("Invalid NodeId");
            Ok(Some(*node_weight > 1))
        };
        let test_color_fn = |edge_id: EdgeIndex| -> Result<Option<usize>, Box<dyn Error>> {
            let edge_weight: &i32 = graph.edge_weight(edge_id).expect("Invalid Edge");
            Ok(Some(*edge_weight as usize))
        };
        let result = collect_bicolor_runs(&graph, test_filter_fn, test_color_fn).unwrap();
        let expected: Vec<Vec<NodeIndex>> = vec![];
        assert_eq!(result, Some(expected))
    }

    #[test]
    fn test_two_colors() {
        /* Based on the following graph from the Python unit tests:
        Input:
                ┌─────────────┐                 ┌─────────────┐
                │             │                 │             │
                │    q0       │                 │    q1       │
                │             │                 │             │
                └───┬─────────┘                 └──────┬──────┘
                    │          ┌─────────────┐         │
                q0  │          │             │         │  q1
                    │          │             │         │
                    └─────────►│     cx      │◄────────┘
                    ┌──────────┤             ├─────────┐
                    │          │             │         │
                q0  │          └─────────────┘         │  q1
                    │                                  │
                    │          ┌─────────────┐         │
                    │          │             │         │
                    └─────────►│      cz     │◄────────┘
                     ┌─────────┤             ├─────────┐
                     │         └─────────────┘         │
                 q0  │                                 │ q1
                     │                                 │
                 ┌───▼─────────┐                ┌──────▼──────┐
                 │             │                │             │
                 │    q0       │                │    q1       │
                 │             │                │             │
                 └─────────────┘                └─────────────┘
        Expected: [[cx, cz]]
        */
        let mut graph = DiGraph::new();
        let n0 = graph.add_node(0); //q0
        let n1 = graph.add_node(0); //q1
        let n2 = graph.add_node(1); //cx
        let n3 = graph.add_node(1); //cz
        let n4 = graph.add_node(0); //q0_1
        let n5 = graph.add_node(0); //q1_1
        graph.add_edge(n0, n2, 0); //q0 -> cx
        graph.add_edge(n1, n2, 1); //q1 -> cx
        graph.add_edge(n2, n3, 0); //cx -> cz
        graph.add_edge(n2, n3, 1); //cx -> cz
        graph.add_edge(n3, n4, 0); //cz -> q0_1
        graph.add_edge(n3, n5, 1); //cz -> q1_1

        // Filter out q0, q1, q0_1 and q1_1
        let test_filter_fn = |node_id: NodeIndex| -> Result<Option<bool>, Box<dyn Error>> {
            let node_weight: &i32 = graph.node_weight(node_id).expect("Invalid NodeId");
            Ok(Some(*node_weight > 0))
        };
        // The edge color will match its weight
        let test_color_fn = |edge_id: EdgeIndex| -> Result<Option<usize>, Box<dyn Error>> {
            let edge_weight: &i32 = graph.edge_weight(edge_id).expect("Invalid Edge");
            Ok(Some(*edge_weight as usize))
        };
        let result = collect_bicolor_runs(&graph, test_filter_fn, test_color_fn).unwrap();
        let expected: Vec<Vec<NodeIndex>> = vec![vec![n2, n3]]; //[[cx, cz]]
        assert_eq!(result, Some(expected))
    }

    #[test]
    fn test_two_colors_with_pending() {
        /* Based on the following graph from the Python unit tests:
        Input:
                ┌─────────────┐
                │             │
                │    q0       │
                │             │
                └───┬─────────┘
                 | q0
                 │
                ┌───▼─────────┐
                │             │
                │    h        │
                │             │
                └───┬─────────┘
                    | q0
                    │                           ┌─────────────┐
                    │                           │             │
                    │                           │    q1       │
                    │                           │             │
                    |                           └──────┬──────┘
                    │          ┌─────────────┐         │
                q0  │          │             │         │  q1
                    │          │             │         │
                    └─────────►│     cx      │◄────────┘
                    ┌──────────┤             ├─────────┐
                    │          │             │         │
                q0  │          └─────────────┘         │  q1
                    │                                  │
                    │          ┌─────────────┐         │
                    │          │             │         │
                    └─────────►│      cz     │◄────────┘
                     ┌─────────┤             ├─────────┐
                     │         └─────────────┘         │
                 q0  │                                 │ q1
                     │                                 │
                 ┌───▼─────────┐                ┌──────▼──────┐
                 │             │                │             │
                 │    q0       │                │    y        │
                 │             │                │             │
                 └─────────────┘                └─────────────┘
                                                    | q1
                                                    │
                                                ┌───▼─────────┐
                                                │             │
                                                │    q1       │
                                                │             │
                                                └─────────────┘
        Expected: [[h, cx, cz, y]]
        */
        let mut graph = DiGraph::new();
        let n0 = graph.add_node(0); //q0
        let n1 = graph.add_node(0); //q1
        let n2 = graph.add_node(1); //h
        let n3 = graph.add_node(1); //cx
        let n4 = graph.add_node(1); //cz
        let n5 = graph.add_node(1); //y
        let n6 = graph.add_node(0); //q0_1
        let n7 = graph.add_node(0); //q1_1
        graph.add_edge(n0, n2, 0); //q0 -> h
        graph.add_edge(n2, n3, 0); //h -> cx
        graph.add_edge(n1, n3, 1); //q1 -> cx
        graph.add_edge(n3, n4, 0); //cx -> cz
        graph.add_edge(n3, n4, 1); //cx -> cz
        graph.add_edge(n4, n6, 0); //cz -> q0_1
        graph.add_edge(n4, n5, 1); //cz -> y
        graph.add_edge(n5, n7, 1); //y -> q1_1

        // Filter out q0, q1, q0_1 and q1_1
        let test_filter_fn = |node_id: NodeIndex| -> Result<Option<bool>, Box<dyn Error>> {
            let node_weight: &i32 = graph.node_weight(node_id).expect("Invalid NodeId");
            Ok(Some(*node_weight > 0))
        };
        // The edge color will match its weight
        let test_color_fn = |edge_id: EdgeIndex| -> Result<Option<usize>, Box<dyn Error>> {
            let edge_weight: &i32 = graph.edge_weight(edge_id).expect("Invalid Edge");
            Ok(Some(*edge_weight as usize))
        };
        let result = collect_bicolor_runs(&graph, test_filter_fn, test_color_fn).unwrap();
        let expected: Vec<Vec<NodeIndex>> = vec![vec![n2, n3, n4, n5]]; //[[h, cx, cz, y]]
        assert_eq!(result, Some(expected))
    }
}

#[cfg(test)]
mod test_collect_runs {
    use super::collect_runs;
    use petgraph::{graph::DiGraph, visit::GraphBase};

    type BareDiGraph = DiGraph<(), ()>;
    type RunResult = Result<Vec<<BareDiGraph as GraphBase>::NodeId>, ()>;

    #[test]
    fn test_empty_graph() {
        let graph: BareDiGraph = DiGraph::new();

        let mut runs = collect_runs(&graph, |_| -> Result<bool, ()> { Ok(true) }).expect("Some");

        let run = runs.next();
        assert!(run == None);

        let runs = collect_runs(&graph, |_| -> Result<bool, ()> { Ok(true) }).expect("Some");

        let runs: Vec<RunResult> = runs.collect();

        assert_eq!(runs, Vec::<RunResult>::new());
    }

    #[test]
    fn test_simple_run_w_filter() {
        let mut graph: BareDiGraph = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n3, ());

        let mut runs = collect_runs(&graph, |_| -> Result<bool, ()> { Ok(true) }).expect("Some");

        let the_run = runs.next().expect("Some").expect("3 nodes");
        assert_eq!(the_run.len(), 3);
        assert_eq!(runs.next(), None);

        assert_eq!(the_run, vec![n1, n2, n3]);

        // Now with some filters
        let mut runs = collect_runs(&graph, |_| -> Result<bool, ()> { Ok(false) }).expect("Some");

        assert_eq!(runs.next(), None);

        let mut runs = collect_runs(&graph, |n| -> Result<bool, ()> { Ok(n != n2) }).expect("Some");

        assert_eq!(runs.next(), Some(Ok(vec![n1])));
        assert_eq!(runs.next(), Some(Ok(vec![n3])));
    }

    #[test]
    fn test_multiple_runs_w_filter() {
        let mut graph: BareDiGraph = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());
        let n5 = graph.add_node(());
        let n6 = graph.add_node(());
        let n7 = graph.add_node(());

        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n3, ());
        graph.add_edge(n3, n7, ());
        graph.add_edge(n4, n3, ());
        graph.add_edge(n4, n7, ());
        graph.add_edge(n5, n4, ());
        graph.add_edge(n6, n5, ());

        let runs: Vec<RunResult> = collect_runs(&graph, |_| -> Result<bool, ()> { Ok(true) })
            .expect("Some")
            .collect();

        assert_eq!(runs, vec![Ok(vec![n6, n5, n4]), Ok(vec![n1, n2, n3, n7])]);

        // And now with some filter
        let runs: Vec<RunResult> =
            collect_runs(&graph, |n| -> Result<bool, ()> { Ok(n != n4 && n != n2) })
                .expect("Some")
                .collect();

        assert_eq!(runs, vec![Ok(vec![n6, n5]), Ok(vec![n1]), Ok(vec![n3, n7])]);
    }

    #[test]
    fn test_singleton_runs_w_filter() {
        let mut graph: BareDiGraph = DiGraph::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());

        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());

        let mut runs = collect_runs(&graph, |_| -> Result<bool, ()> { Ok(true) }).expect("Some");

        assert_eq!(runs.next().expect("n1"), Ok(vec![n1]));
        assert_eq!(runs.next().expect("n3"), Ok(vec![n3]));
        assert_eq!(runs.next().expect("n2"), Ok(vec![n2]));

        // And now with some filter
        let runs: Vec<RunResult> = collect_runs(&graph, |n| -> Result<bool, ()> { Ok(n != n1) })
            .expect("Some")
            .collect();

        assert_eq!(runs, vec![Ok(vec![n3]), Ok(vec![n2])]);
    }

    #[test]
    fn test_error_propagation() {
        let mut graph: BareDiGraph = DiGraph::new();
        graph.add_node(());

        let mut runs = collect_runs(&graph, |_| -> Result<bool, ()> { Err(()) }).expect("Some");

        assert!(runs.next().expect("Some").is_err());
        assert_eq!(runs.next(), None);
    }
}
