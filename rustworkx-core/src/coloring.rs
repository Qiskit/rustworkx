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

use ahash::RandomState;
use priority_queue::PriorityQueue;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::convert::Infallible;
use std::hash::Hash;

use crate::dictmap::*;
use crate::line_graph::line_graph;

use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use petgraph::graph::NodeIndex;
use petgraph::visit::{
    EdgeCount, EdgeIndexable, EdgeRef, GraphBase, GraphProp, IntoEdges, IntoNeighborsDirected,
    IntoNodeIdentifiers, NodeCount, NodeIndexable,
};
use petgraph::{Incoming, Outgoing};
use rayon::prelude::*;

/// Compute a two-coloring of a graph
///
/// If a two coloring is not possible for the input graph (meaning it is not
/// bipartite), `None` is returned.
///
/// Arguments:
///
/// * `graph` - The graph to find the coloring for
///
/// # Example
///
/// ```rust
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::coloring::two_color;
/// use rustworkx_core::dictmap::*;
///
/// let edge_list = vec![
///  (0, 1),
///  (1, 2),
///  (2, 3),
///  (3, 4),
/// ];
///
/// let graph = UnGraph::<i32, i32>::from_edges(&edge_list);
/// let coloring = two_color(&graph).unwrap();
/// let mut expected_colors = DictMap::new();
/// expected_colors.insert(NodeIndex::new(0), 1);
/// expected_colors.insert(NodeIndex::new(1), 0);
/// expected_colors.insert(NodeIndex::new(2), 1);
/// expected_colors.insert(NodeIndex::new(3), 0);
/// expected_colors.insert(NodeIndex::new(4), 1);
/// assert_eq!(coloring, expected_colors)
/// ```
pub fn two_color<G>(graph: G) -> Option<DictMap<G::NodeId, u8>>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoNeighborsDirected
        + GraphBase
        + GraphProp
        + NodeCount,
    <G as GraphBase>::NodeId: std::cmp::Eq + Hash,
{
    let mut colors = DictMap::with_capacity(graph.node_count());
    for node in graph.node_identifiers() {
        if colors.contains_key(&node) {
            continue;
        }
        let mut queue = vec![node];
        colors.insert(node, 1);
        while let Some(v) = queue.pop() {
            let v_color: u8 = *colors.get(&v).unwrap();
            let color: u8 = 1 - v_color;
            for w in graph
                .neighbors_directed(v, Outgoing)
                .chain(graph.neighbors_directed(v, Incoming))
            {
                if let Some(color_w) = colors.get(&w) {
                    if *color_w == v_color {
                        return None;
                    }
                } else {
                    colors.insert(w, color);
                    queue.push(w);
                }
            }
        }
    }
    Some(colors)
}

/// coloring strategies for greedy node- and edge- coloring algorithms
#[derive(Clone, PartialEq)]
pub enum ColoringStrategy {
    Degree,
    Saturation,
    IndependentSet,
}

fn inner_greedy_node_color<G, F, E>(
    graph: G,
    preset_color_fn: F,
    strategy: ColoringStrategy,
) -> Result<DictMap<G::NodeId, usize>, E>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::NodeId: Hash + Eq + Send + Sync,
    F: FnMut(G::NodeId) -> Result<Option<usize>, E>,
{
    match strategy {
        ColoringStrategy::Degree => inner_greedy_node_color_strategy_degree(graph, preset_color_fn),
        ColoringStrategy::Saturation => {
            inner_greedy_node_color_strategy_saturation(graph, preset_color_fn)
        }
        ColoringStrategy::IndependentSet => {
            inner_greedy_node_color_strategy_independent_set(graph, preset_color_fn)
        }
    }
}

fn inner_greedy_node_color_strategy_degree<G, F, E>(
    graph: G,
    mut preset_color_fn: F,
) -> Result<DictMap<G::NodeId, usize>, E>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::NodeId: Hash + Eq + Send + Sync,
    F: FnMut(G::NodeId) -> Result<Option<usize>, E>,
{
    let mut colors: DictMap<G::NodeId, usize> = DictMap::with_capacity(graph.node_count());
    let mut node_vec: Vec<G::NodeId> = Vec::with_capacity(graph.node_count());
    let mut sort_map: HashMap<G::NodeId, usize> = HashMap::with_capacity(graph.node_count());
    for k in graph.node_identifiers() {
        if let Some(color) = preset_color_fn(k)? {
            colors.insert(k, color);
            continue;
        }
        node_vec.push(k);
        sort_map.insert(k, graph.edges(k).count());
    }
    node_vec.par_sort_by_key(|k| Reverse(sort_map.get(k)));

    for node in node_vec {
        let mut neighbor_colors: HashSet<usize> = HashSet::new();
        for edge in graph.edges(node) {
            let target = edge.target();
            let existing_color = match colors.get(&target) {
                Some(color) => color,
                None => continue,
            };
            neighbor_colors.insert(*existing_color);
        }
        let mut current_color: usize = 0;
        loop {
            if !neighbor_colors.contains(&current_color) {
                break;
            }
            current_color += 1;
        }
        colors.insert(node, current_color);
    }
    Ok(colors)
}

/// Data associated to nodes for the greedy coloring algorithm
/// using the "saturation first" strategy: always picking the node that
/// has the largest number of different colors already assigned to its
/// neighbors, and, in case of a tie, the node that has the largest number
/// of uncolored neighbors.
#[derive(Clone, Eq, PartialEq)]
struct SaturationStrategyData {
    // degree of a node: number of neighbors without color
    degree: usize,
    // saturation degree of a node: number of colors used by neighbors
    saturation: usize,
}

impl Ord for SaturationStrategyData {
    fn cmp(&self, other: &SaturationStrategyData) -> Ordering {
        self.saturation
            .cmp(&other.saturation)
            .then_with(|| self.degree.cmp(&other.degree))
    }
}

impl PartialOrd for SaturationStrategyData {
    fn partial_cmp(&self, other: &SaturationStrategyData) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn inner_greedy_node_color_strategy_saturation<G, F, E>(
    graph: G,
    mut preset_color_fn: F,
) -> Result<DictMap<G::NodeId, usize>, E>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::NodeId: Hash + Eq + Send + Sync,
    F: FnMut(G::NodeId) -> Result<Option<usize>, E>,
{
    let mut colors: DictMap<G::NodeId, usize> = DictMap::with_capacity(graph.node_count());
    let mut nbd_colors: HashMap<G::NodeId, HashSet<usize>> = graph
        .node_identifiers()
        .map(|k| (k, HashSet::new()))
        .collect();

    let mut pq: PriorityQueue<G::NodeId, SaturationStrategyData> =
        PriorityQueue::with_capacity(graph.node_count());

    // Handle preset nodes
    for k in graph.node_identifiers() {
        if let Some(color) = preset_color_fn(k)? {
            colors.insert(k, color);
            for v in graph.neighbors(k) {
                nbd_colors.get_mut(&v).unwrap().insert(color);
            }
        }
    }

    // Add non-preset nodes to priority queue
    for k in graph.node_identifiers() {
        if colors.get(&k).is_none() {
            let degree = graph
                .neighbors(k)
                .filter(|v| colors.get(v).is_none())
                .count();
            let saturation = nbd_colors.get(&k).unwrap().len();
            pq.push(k, SaturationStrategyData { degree, saturation });
        }
    }

    // Greedily process nodes
    while let Some((k, _)) = pq.pop() {
        let neighbor_colors = nbd_colors.get(&k).unwrap();
        let mut current_color: usize = 0;
        while neighbor_colors.contains(&current_color) {
            current_color += 1;
        }

        colors.insert(k, current_color);
        for v in graph.neighbors(k) {
            if colors.get(&v).is_none() {
                nbd_colors.get_mut(&v).unwrap().insert(current_color);
                let (_, vdata) = pq.get(&v).unwrap();

                pq.push(
                    v,
                    SaturationStrategyData {
                        degree: vdata.degree - 1,
                        saturation: nbd_colors.get(&v).unwrap().len(),
                    },
                );
            }
        }
    }

    Ok(colors)
}

fn inner_greedy_node_color_strategy_independent_set<G, F, E>(
    graph: G,
    mut preset_color_fn: F,
) -> Result<DictMap<G::NodeId, usize>, E>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::NodeId: Hash + Eq + Send + Sync,
    F: FnMut(G::NodeId) -> Result<Option<usize>, E>,
{
    let mut colors: DictMap<G::NodeId, usize> = DictMap::with_capacity(graph.node_count());

    let mut preset: HashSet<G::NodeId> = HashSet::new();
    let mut unprocessed: IndexSet<G::NodeId, RandomState> =
        IndexSet::with_hasher(RandomState::default());

    // Handle preset nodes
    for k in graph.node_identifiers() {
        if let Some(color) = preset_color_fn(k)? {
            colors.insert(k, color);
            preset.insert(k);
        } else {
            unprocessed.insert(k);
        }
    }

    let mut current_color = 0;
    while !unprocessed.is_empty() {
        let mut remaining: IndexSet<G::NodeId, RandomState> =
            IndexSet::with_hasher(RandomState::default());

        // Remove neighbors of preset nodes with the given color
        for k in &preset {
            if colors.get(k) == Some(&current_color) {
                for u in graph.neighbors(*k) {
                    if unprocessed.swap_take(&u).is_some() {
                        remaining.insert(u);
                    }
                }
            }
        }

        // Greedily extract maximal independent set
        while !unprocessed.is_empty() {
            // Greedily take any node
            // Possible optimization is to choose node with smallest degree among unprocessed
            let k = *unprocessed.iter().next().unwrap();
            colors.insert(k, current_color);
            unprocessed.swap_remove(&k);
            for u in graph.neighbors(k) {
                if unprocessed.swap_take(&u).is_some() {
                    remaining.insert(u);
                }
            }
        }

        unprocessed = remaining;
        current_color += 1;
    }

    Ok(colors)
}

/// Color a graph using a greedy graph coloring algorithm.
///
/// This function uses a `largest-first` strategy as described in:
///
/// Adrian Kosowski, and Krzysztof Manuszewski, Classical Coloring of Graphs,
/// Graph Colorings, 2-19, 2004. ISBN 0-8218-3458-4.
///
/// to color the nodes with higher degree first.
///
/// The coloring problem is NP-hard and this is a heuristic algorithm
/// which may not return an optimal solution.
///
/// # Note:
/// Please consider using ``greedy_node_color_with_coloring_strategy``, which is
/// a more general version of this function.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
///
/// # Example
/// ```rust
///
/// use petgraph::graph::Graph;
/// use petgraph::graph::NodeIndex;
/// use petgraph::Undirected;
/// use rustworkx_core::dictmap::*;
/// use rustworkx_core::coloring::greedy_node_color;
///
/// let g = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2)]);
/// let colors = greedy_node_color(&g);
/// let mut expected_colors = DictMap::new();
/// expected_colors.insert(NodeIndex::new(0), 0);
/// expected_colors.insert(NodeIndex::new(1), 1);
/// expected_colors.insert(NodeIndex::new(2), 1);
/// assert_eq!(colors, expected_colors);
/// ```
///
pub fn greedy_node_color<G>(graph: G) -> DictMap<G::NodeId, usize>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::NodeId: Hash + Eq + Send + Sync,
{
    inner_greedy_node_color(
        graph,
        |_| Ok::<Option<usize>, Infallible>(None),
        ColoringStrategy::Degree,
    )
    .unwrap()
}

/// Color a graph using a greedy graph coloring algorithm with preset colors
///
/// This function uses a `largest-first` strategy as described in:
///
/// Adrian Kosowski, and Krzysztof Manuszewski, Classical Coloring of Graphs,
/// Graph Colorings, 2-19, 2004. ISBN 0-8218-3458-4.
///
/// to color the nodes with higher degree first.
///
/// The coloring problem is NP-hard and this is a heuristic algorithm
/// which may not return an optimal solution.
///
/// # Note:
/// Please consider using ``greedy_node_color_with_coloring_strategy``, which is
/// a more general version of this function.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
/// * `preset_color_fn` - A callback function that will recieve the node identifier
///     for each node in the graph and is expected to return an `Option<usize>`
///     (wrapped in a `Result`) that is `None` if the node has no preset and
///     the usize represents the preset color.
///
/// # Example
/// ```rust
///
/// use petgraph::graph::Graph;
/// use petgraph::graph::NodeIndex;
/// use petgraph::Undirected;
/// use rustworkx_core::dictmap::*;
/// use std::convert::Infallible;
/// use rustworkx_core::coloring::greedy_node_color_with_preset_colors;
///
/// let preset_color_fn = |node_idx: NodeIndex| -> Result<Option<usize>, Infallible> {
///     if node_idx.index() == 0 {
///         Ok(Some(1))
///     } else {
///         Ok(None)
///     }
/// };
///
/// let g = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2)]);
/// let colors = greedy_node_color_with_preset_colors(&g, preset_color_fn).unwrap();
/// let mut expected_colors = DictMap::new();
/// expected_colors.insert(NodeIndex::new(0), 1);
/// expected_colors.insert(NodeIndex::new(1), 0);
/// expected_colors.insert(NodeIndex::new(2), 0);
/// assert_eq!(colors, expected_colors);
/// ```
pub fn greedy_node_color_with_preset_colors<G, F, E>(
    graph: G,
    preset_color_fn: F,
) -> Result<DictMap<G::NodeId, usize>, E>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::NodeId: Hash + Eq + Send + Sync,
    F: FnMut(G::NodeId) -> Result<Option<usize>, E>,
{
    inner_greedy_node_color(graph, preset_color_fn, ColoringStrategy::Degree)
}

/// Color a graph using a greedy graph coloring algorithm with preset colors.
///
/// This function uses one of several greedy strategies described in:
///
/// Adrian Kosowski, and Krzysztof Manuszewski, Classical Coloring of Graphs,
/// Graph Colorings, 2-19, 2004. ISBN 0-8218-3458-4.
///
/// The `Degree` (aka `largest-first`) strategy colors the nodes with higher degree
/// first. The `Saturation` (aka `DSATUR` and `SLF`) strategy dynamically
/// chooses the vertex that has the largest number of different colors already
/// assigned to its neighbors, and, in case of a tie, the vertex that has the
/// largest number of uncolored neighbors. The `IndependentSet` strategy finds
/// independent subsets of the graph and assigns a different color to each of these
/// subsets.
///
/// to color the nodes with higher degree first.
///
/// The coloring problem is NP-hard and this is a heuristic algorithm
/// which may not return an optimal solution.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on.
/// * `preset_color_fn` - A callback function that will receive the node identifier
///     for each node in the graph and is expected to return an `Option<usize>`
///     (wrapped in a `Result`) that is `None` if the node has no preset and
///     the usize represents the preset color.
/// * `strategy` - The greedy strategy used by the algorithm.
///
/// # Example
/// ```rust
///
/// use petgraph::graph::Graph;
/// use petgraph::graph::NodeIndex;
/// use petgraph::Undirected;
/// use rustworkx_core::dictmap::*;
/// use std::convert::Infallible;
/// use rustworkx_core::coloring::{greedy_node_color_with_coloring_strategy, ColoringStrategy};
///
/// let preset_color_fn = |node_idx: NodeIndex| -> Result<Option<usize>, Infallible> {
///     if node_idx.index() == 0 {
///         Ok(Some(1))
///     } else {
///         Ok(None)
///     }
/// };
///
/// let g = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2)]);
/// let colors = greedy_node_color_with_coloring_strategy(&g, preset_color_fn, ColoringStrategy::Degree).unwrap();
/// let mut expected_colors = DictMap::new();
/// expected_colors.insert(NodeIndex::new(0), 1);
/// expected_colors.insert(NodeIndex::new(1), 0);
/// expected_colors.insert(NodeIndex::new(2), 0);
/// assert_eq!(colors, expected_colors);
/// ```
pub fn greedy_node_color_with_coloring_strategy<G, F, E>(
    graph: G,
    preset_color_fn: F,
    strategy: ColoringStrategy,
) -> Result<DictMap<G::NodeId, usize>, E>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::NodeId: Hash + Eq + Send + Sync,
    F: FnMut(G::NodeId) -> Result<Option<usize>, E>,
{
    inner_greedy_node_color(graph, preset_color_fn, strategy)
}

/// Color edges of a graph using a greedy approach.
///
/// This function works by greedily coloring the line graph of the given graph.
///
/// The coloring problem is NP-hard and this is a heuristic algorithm
/// which may not return an optimal solution.
///
/// # Note:
/// Please consider using ``greedy_edge_color_with_coloring_strategy``, which is
/// a more general version of this function.
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
///
/// # Example
/// ```rust
///
/// use petgraph::graph::Graph;
/// use petgraph::graph::EdgeIndex;
/// use petgraph::Undirected;
/// use rustworkx_core::dictmap::*;
/// use rustworkx_core::coloring::greedy_edge_color;
///
/// let g = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (1, 2), (0, 2), (2, 3)]);
/// let colors = greedy_edge_color(&g);
/// let mut expected_colors = DictMap::new();
/// expected_colors.insert(EdgeIndex::new(0), 2);
/// expected_colors.insert(EdgeIndex::new(1), 0);
/// expected_colors.insert(EdgeIndex::new(2), 1);
/// expected_colors.insert(EdgeIndex::new(3), 2);
/// assert_eq!(colors, expected_colors);
/// ```
///
pub fn greedy_edge_color<G>(graph: G) -> DictMap<G::EdgeId, usize>
where
    G: EdgeCount + IntoNodeIdentifiers + IntoEdges,
    G::EdgeId: Hash + Eq,
{
    let (new_graph, edge_to_node_map): (
        petgraph::graph::UnGraph<(), ()>,
        HashMap<G::EdgeId, NodeIndex>,
    ) = line_graph(&graph, || (), || ());

    let colors = greedy_node_color(&new_graph);

    let mut edge_colors: DictMap<G::EdgeId, usize> = DictMap::with_capacity(graph.edge_count());

    for edge in graph.edge_references() {
        let edge_index = edge.id();
        let node_index = edge_to_node_map.get(&edge_index).unwrap();
        let edge_color = colors.get(node_index).unwrap();
        edge_colors.insert(edge_index, *edge_color);
    }

    edge_colors
}

/// Color edges of a graph using a greedy graph coloring algorithm with preset
/// colors.
///
/// This function works by greedily coloring the line graph of the given graph.
///
/// The coloring problem is NP-hard and this is a heuristic algorithm
/// which may not return an optimal solution.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on.
/// * `preset_color_fn` - A callback function that will receive the edge identifier
///     for each edge in the graph and is expected to return an `Option<usize>`
///     (wrapped in a `Result`) that is `None` if the edge has no preset and
///     the usize represents the preset color.
/// * `strategy` - The greedy strategy used by the algorithm.
///
/// # Example
/// ```rust
///
/// use petgraph::graph::Graph;
/// use petgraph::graph::EdgeIndex;
/// use petgraph::Undirected;
/// use rustworkx_core::dictmap::*;
/// use rustworkx_core::coloring::{greedy_edge_color_with_coloring_strategy, ColoringStrategy};
/// use std::convert::Infallible;
///
/// let g = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (1, 2), (0, 2), (2, 3)]);
/// let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);
/// let colors = greedy_edge_color_with_coloring_strategy(&g, preset_color_fn, ColoringStrategy::Degree).unwrap();
/// let mut expected_colors = DictMap::new();
/// expected_colors.insert(EdgeIndex::new(0), 2);
/// expected_colors.insert(EdgeIndex::new(1), 0);
/// expected_colors.insert(EdgeIndex::new(2), 1);
/// expected_colors.insert(EdgeIndex::new(3), 2);
/// assert_eq!(colors, expected_colors);
/// ```
///
pub fn greedy_edge_color_with_coloring_strategy<G, F, E>(
    graph: G,
    preset_color_fn: F,
    strategy: ColoringStrategy,
) -> Result<DictMap<G::EdgeId, usize>, E>
where
    G: EdgeCount + IntoNodeIdentifiers + IntoEdges,
    G::EdgeId: Hash + Eq,
    F: Fn(G::EdgeId) -> Result<Option<usize>, E>,
{
    let (new_graph, edge_to_node_map): (
        petgraph::graph::UnGraph<(), ()>,
        HashMap<G::EdgeId, NodeIndex>,
    ) = line_graph(&graph, || (), || ());

    let node_to_edge_map: HashMap<&NodeIndex, &G::EdgeId> =
        edge_to_node_map.iter().map(|(k, v)| (v, k)).collect();
    let new_graph_preset_color_fn =
        |x: NodeIndex| preset_color_fn(**node_to_edge_map.get(&x).unwrap());

    let colors = inner_greedy_node_color(&new_graph, new_graph_preset_color_fn, strategy)?;

    let mut edge_colors: DictMap<G::EdgeId, usize> = DictMap::with_capacity(graph.edge_count());

    for edge in graph.edge_references() {
        let edge_index = edge.id();
        let node_index = edge_to_node_map.get(&edge_index).unwrap();
        let edge_color = colors.get(node_index).unwrap();
        edge_colors.insert(edge_index, *edge_color);
    }

    Ok(edge_colors)
}

struct MisraGries<G: GraphBase> {
    // The input graph
    graph: G,
    // Maximum node degree in the graph
    max_node_degree: usize,
    // Partially assigned colors (indexed by internal edge index)
    colors: Vec<Option<usize>>,
    // Performance optimization: explicitly storing edge colors used at each node
    node_used_colors: Vec<HashSet<usize>>,
}

impl<G> MisraGries<G>
where
    G: EdgeIndexable + IntoEdges + NodeIndexable + IntoNodeIdentifiers,
{
    pub fn new(graph: G) -> Self {
        let colors = vec![None; graph.edge_bound()];
        let max_node_degree = graph
            .node_identifiers()
            .map(|node| graph.edges(node).count())
            .max()
            .unwrap_or(0);
        let empty_set = HashSet::new();
        let node_used_colors = vec![empty_set; graph.node_bound()];

        MisraGries {
            graph,
            max_node_degree,
            colors,
            node_used_colors,
        }
    }

    // Updates edge colors for a set of edges while keeping track of
    // explicitly stored used node colors
    fn update_edge_colors(&mut self, new_colors: &Vec<(G::EdgeRef, usize)>) {
        // First, remove node colors that are going to be unassigned
        for (e, _) in new_colors {
            if let Some(old_color) = self.get_edge_color(*e) {
                self.remove_node_used_color(e.source(), old_color);
                self.remove_node_used_color(e.target(), old_color);
            }
        }
        // Next, add node colors that are going to be assigned
        for (e, c) in new_colors {
            self.add_node_used_color(e.source(), *c);
            self.add_node_used_color(e.target(), *c);
        }
        for (e, c) in new_colors {
            self.colors[EdgeIndexable::to_index(&self.graph, e.id())] = Some(*c);
        }
    }

    // Updates used node colors at u adding c
    fn add_node_used_color(&mut self, u: G::NodeId, c: usize) {
        let uindex = NodeIndexable::to_index(&self.graph, u);
        self.node_used_colors[uindex].insert(c);
    }

    // Updates used node colors at u removing c
    fn remove_node_used_color(&mut self, u: G::NodeId, c: usize) {
        let uindex = NodeIndexable::to_index(&self.graph, u);
        self.node_used_colors[uindex].remove(&c);
    }

    // Gets edge color
    fn get_edge_color(&self, e: G::EdgeRef) -> Option<usize> {
        self.colors[EdgeIndexable::to_index(&self.graph, e.id())]
    }

    // Returns colors used at node u
    fn get_used_colors(&self, u: G::NodeId) -> &HashSet<usize> {
        let uindex = NodeIndexable::to_index(&self.graph, u);
        &self.node_used_colors[uindex]
    }

    // Returns the smallest free (aka unused) color at node u
    fn get_free_color(&self, u: G::NodeId) -> usize {
        let used_colors = self.get_used_colors(u);
        let free_color: usize = (0..self.max_node_degree + 1)
            .position(|color| !used_colors.contains(&color))
            .unwrap();
        free_color
    }

    // Returns true iff color c is free at node u
    fn is_free_color(&self, u: G::NodeId, c: usize) -> bool {
        !self.get_used_colors(u).contains(&c)
    }

    // Returns the maximal fan on edge (u, v) at u
    fn get_maximal_fan(&self, e: G::EdgeRef, u: G::NodeId, v: G::NodeId) -> Vec<G::EdgeRef> {
        let mut fan: Vec<G::EdgeRef> = vec![e];

        let mut neighbors: Vec<G::EdgeRef> = self.graph.edges(u).collect();

        let mut last_node_in_fan = v;
        neighbors.remove(
            neighbors
                .iter()
                .position(|x| x.target() == last_node_in_fan)
                .unwrap(),
        );

        let mut fan_extended: bool = true;
        while fan_extended {
            fan_extended = false;

            for edge in &neighbors {
                if let Some(color) = self.get_edge_color(*edge) {
                    if self.is_free_color(last_node_in_fan, color) {
                        fan.push(*edge);
                        last_node_in_fan = edge.target();
                        fan_extended = true;
                        neighbors.remove(
                            neighbors
                                .iter()
                                .position(|x| x.target() == last_node_in_fan)
                                .unwrap(),
                        );
                        break;
                    }
                }
            }
        }

        fan
    }

    // Assuming that color is either c or d, returns the other color
    fn flip_color(&self, c: usize, d: usize, e: usize) -> usize {
        if e == c {
            d
        } else {
            c
        }
    }

    // Returns the longest path starting at node u with alternating colors c, d, c, d, c, etc.
    fn get_cdu_path(&self, u: G::NodeId, c: usize, d: usize) -> Vec<(G::EdgeRef, usize)> {
        let mut path: Vec<(G::EdgeRef, usize)> = Vec::new();
        let mut cur_node = u;
        let mut cur_color = c;
        let mut path_extended = true;

        while path_extended {
            path_extended = false;
            for edge in self.graph.edges(cur_node) {
                if let Some(color) = self.get_edge_color(edge) {
                    if color == cur_color {
                        path_extended = true;
                        path.push((edge, cur_color));
                        cur_node = edge.target();
                        cur_color = self.flip_color(c, d, cur_color);
                        break;
                    }
                }
            }
        }
        path
    }

    // Main function
    pub fn run_algorithm(&mut self) -> &Vec<Option<usize>> {
        for edge in self.graph.edge_references() {
            let u: G::NodeId = edge.source();
            let v: G::NodeId = edge.target();
            let fan = self.get_maximal_fan(edge, u, v);
            let c = self.get_free_color(u);
            let d = self.get_free_color(fan.last().unwrap().target());

            // find cdu-path
            let cdu_path = self.get_cdu_path(u, d, c);

            // invert colors on cdu-path
            let mut new_cdu_path_colors: Vec<(G::EdgeRef, usize)> =
                Vec::with_capacity(cdu_path.len());
            for (cdu_edge, color) in cdu_path {
                let flipped_color = self.flip_color(c, d, color);
                new_cdu_path_colors.push((cdu_edge, flipped_color));
            }
            self.update_edge_colors(&new_cdu_path_colors);

            // find sub-fan fan[0..w] such that d is free on fan[w]
            let mut w = 0;
            for (i, x) in fan.iter().enumerate() {
                if self.is_free_color(x.target(), d) {
                    w = i;
                    break;
                }
            }

            // rotate fan + fill additional color
            let mut new_fan_colors: Vec<(G::EdgeRef, usize)> = Vec::with_capacity(w + 1);
            for i in 1..w + 1 {
                let next_color = self.get_edge_color(fan[i]).unwrap();
                new_fan_colors.push((fan[i - 1], next_color));
            }
            new_fan_colors.push((fan[w], d));
            self.update_edge_colors(&new_fan_colors);
        }

        &self.colors
    }
}

/// Color edges of a graph using the Misra-Gries edge coloring algorithm.
///
/// Based on the paper: "A constructive proof of Vizing's theorem" by
/// Misra and Gries, 1992.
/// <https://www.cs.utexas.edu/users/misra/psp.dir/vizing.pdf>
///
/// The coloring produces at most d + 1 colors where d is the maximum degree
/// of the graph.
///
/// The coloring problem is NP-hard and this is a heuristic algorithm
/// which may not return an optimal solution.
///
/// Arguments:
///
/// * `graph` - The graph object to run the algorithm on
///
/// # Example
/// ```rust
///
/// use petgraph::graph::Graph;
/// use petgraph::graph::EdgeIndex;
/// use petgraph::Undirected;
/// use rustworkx_core::dictmap::*;
/// use rustworkx_core::coloring::misra_gries_edge_color;///
/// let g = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (1, 2), (0, 2), (2, 3)]);
/// let colors = misra_gries_edge_color(&g);
///
/// let expected_colors: DictMap<EdgeIndex, usize> = [
///     (EdgeIndex::new(0), 2),
///     (EdgeIndex::new(1), 1),
///     (EdgeIndex::new(2), 0),
///     (EdgeIndex::new(3), 2),
/// ]
/// .into_iter()
/// .collect();
///
///  assert_eq!(colors, expected_colors);
/// ```
///
pub fn misra_gries_edge_color<G>(graph: G) -> DictMap<G::EdgeId, usize>
where
    G: EdgeIndexable + IntoEdges + EdgeCount + NodeIndexable + IntoNodeIdentifiers,
    G::EdgeId: Eq + Hash,
{
    let mut mg: MisraGries<G> = MisraGries::new(graph);
    let colors = mg.run_algorithm();

    let mut edge_colors: DictMap<G::EdgeId, usize> = DictMap::with_capacity(graph.edge_count());
    for edge in graph.edge_references() {
        let edge_index = edge.id();
        let color = colors[EdgeIndexable::to_index(&graph, edge_index)].unwrap();
        edge_colors.insert(edge_index, color);
    }
    edge_colors
}

#[cfg(test)]
mod test_node_coloring {
    use crate::coloring::{
        greedy_node_color, greedy_node_color_with_coloring_strategy, two_color, ColoringStrategy,
    };
    use crate::dictmap::*;
    use crate::generators::{complete_graph, cycle_graph, heavy_hex_graph, path_graph};
    use std::convert::Infallible;
    use std::hash::Hash;

    use crate::petgraph::graph::NodeIndex;
    use crate::petgraph::prelude::*;
    use crate::petgraph::Undirected;
    use petgraph::visit::{IntoEdgeReferences, IntoNodeIdentifiers};

    /// Helper function to check validity of node coloring
    fn check_node_colors<G>(graph: G, colors: &DictMap<G::NodeId, usize>)
    where
        G: IntoNodeIdentifiers + IntoEdgeReferences,
        G::NodeId: Hash + Eq + Send + Sync,
    {
        // Check that every node has valid color
        for k in graph.node_identifiers() {
            if !colors.contains_key(&k) {
                panic!("Problem: some nodes have no color assigned.");
            } else {
                println!("Valid color: ok");
            }
        }

        // Check that nodes connected by an edge have different colors
        for e in graph.edge_references() {
            if colors.get(&e.source()) == colors.get(&e.target()) {
                panic!("Problem: same color for connected nodes.");
            } else {
                println!("Connected nodes: ok");
            }
        }
    }

    /// Helper function to check validity of node coloring with preset colors
    fn check_preset_colors<G, F, E>(
        graph: G,
        colors: &DictMap<G::NodeId, usize>,
        mut preset_color_fn: F,
    ) where
        G: IntoNodeIdentifiers + IntoEdgeReferences,
        G::NodeId: Hash + Eq + Send + Sync,
        F: FnMut(G::NodeId) -> Result<Option<usize>, E>,
    {
        // Check preset values
        for k in graph.node_identifiers() {
            if let Ok(Some(color)) = preset_color_fn(k) {
                if *colors.get(&k).unwrap() != color {
                    panic!("Problem: colors are different from preset vales.");
                } else {
                    println!("Preset values: ok");
                }
            }
        }
    }

    #[test]
    fn test_legacy_greedy_node_color_empty_graph() {
        // Empty graph
        let graph = Graph::<(), (), Undirected>::new_undirected();
        let colors = greedy_node_color(&graph);
        let expected_colors: DictMap<NodeIndex, usize> = [].into_iter().collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_legacy_greedy_node_color_simple_graph() {
        // Simple graph
        let graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (0, 2)]);
        let colors = greedy_node_color(&graph);
        let expected_colors: DictMap<NodeIndex, usize> = [
            (NodeIndex::new(0), 0),
            (NodeIndex::new(1), 1),
            (NodeIndex::new(2), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_legacy_greedy_node_color_simple_graph_large_degree() {
        // Graph with multiple edges
        let graph = Graph::<(), (), Undirected>::from_edges([
            (0, 1),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
        ]);
        let colors = greedy_node_color(&graph);
        let expected_colors: DictMap<NodeIndex, usize> = [
            (NodeIndex::new(0), 0),
            (NodeIndex::new(1), 1),
            (NodeIndex::new(2), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_node_color_empty_graph() {
        // Empty graph
        let graph = Graph::<(), (), Undirected>::new_undirected();
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        for strategy in vec![
            ColoringStrategy::Degree,
            ColoringStrategy::Saturation,
            ColoringStrategy::IndependentSet,
        ] {
            let colors =
                greedy_node_color_with_coloring_strategy(&graph, preset_color_fn, strategy);
            let expected_colors: DictMap<NodeIndex, usize> = [].into_iter().collect();
            assert_eq!(colors, Ok(expected_colors));
        }
    }

    #[test]
    fn test_greedy_node_color_simple_graph() {
        // Simple graph
        let graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (0, 2)]);
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);
        let colors = greedy_node_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Degree,
        );
        let expected_colors: DictMap<NodeIndex, usize> = [
            (NodeIndex::new(0), 0),
            (NodeIndex::new(1), 1),
            (NodeIndex::new(2), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, Ok(expected_colors));
    }

    #[test]
    fn test_greedy_node_color_simple_graph_large_degree() {
        // Graph with multiple edges
        let graph = Graph::<(), (), Undirected>::from_edges([
            (0, 1),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
        ]);
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);
        let colors = greedy_node_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Degree,
        );
        let expected_colors: DictMap<NodeIndex, usize> = [
            (NodeIndex::new(0), 0),
            (NodeIndex::new(1), 1),
            (NodeIndex::new(2), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, Ok(expected_colors));
    }

    #[test]
    fn test_greedy_node_color_saturation() {
        // Simple graph
        let graph = Graph::<(), (), Undirected>::from_edges([
            (0, 1),
            (0, 2),
            (0, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (5, 7),
        ]);

        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);
        let colors = greedy_node_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Saturation,
        )
        .unwrap();
        check_node_colors(&graph, &colors);

        let expected_colors: DictMap<NodeIndex, usize> = [
            (NodeIndex::new(0), 0),
            (NodeIndex::new(1), 1),
            (NodeIndex::new(2), 1),
            (NodeIndex::new(3), 1),
            (NodeIndex::new(4), 0),
            (NodeIndex::new(5), 1),
            (NodeIndex::new(6), 0),
            (NodeIndex::new(7), 0),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_node_color_saturation_and_preset() {
        // Simple graph
        let graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (0, 2), (2, 3), (2, 4)]);

        let preset_color_fn = |node_idx: NodeIndex| -> Result<Option<usize>, Infallible> {
            if node_idx.index() == 0 {
                Ok(Some(1))
            } else {
                Ok(None)
            }
        };

        let colors = greedy_node_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Saturation,
        )
        .unwrap();
        check_node_colors(&graph, &colors);
        check_preset_colors(&graph, &colors, preset_color_fn);

        let expected_colors: DictMap<NodeIndex, usize> = [
            (NodeIndex::new(0), 1),
            (NodeIndex::new(1), 0),
            (NodeIndex::new(2), 0),
            (NodeIndex::new(3), 1),
            (NodeIndex::new(4), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_node_color_independent_set() {
        // Simple graph
        let graph = Graph::<(), (), Undirected>::from_edges([
            (0, 1),
            (0, 2),
            (0, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (5, 7),
        ]);

        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);
        let colors = greedy_node_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::IndependentSet,
        )
        .unwrap();
        check_node_colors(&graph, &colors);

        let expected_colors: DictMap<NodeIndex, usize> = [
            (NodeIndex::new(0), 0),
            (NodeIndex::new(1), 1),
            (NodeIndex::new(2), 1),
            (NodeIndex::new(3), 1),
            (NodeIndex::new(4), 0),
            (NodeIndex::new(5), 1),
            (NodeIndex::new(6), 0),
            (NodeIndex::new(7), 0),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_node_color_independent_set_and_preset() {
        // Simple graph
        let graph = Graph::<(), (), Undirected>::from_edges([
            (0, 1),
            (0, 2),
            (0, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (5, 7),
        ]);

        let preset_color_fn = |node_idx: NodeIndex| -> Result<Option<usize>, Infallible> {
            if node_idx.index() == 0 {
                Ok(Some(1))
            } else if node_idx.index() == 3 {
                Ok(Some(0))
            } else {
                Ok(None)
            }
        };

        let colors = greedy_node_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::IndependentSet,
        )
        .unwrap();
        check_node_colors(&graph, &colors);
        check_preset_colors(&graph, &colors, preset_color_fn);

        let expected_colors: DictMap<NodeIndex, usize> = [
            (NodeIndex::new(0), 1),
            (NodeIndex::new(1), 0),
            (NodeIndex::new(2), 0),
            (NodeIndex::new(3), 0),
            (NodeIndex::new(4), 1),
            (NodeIndex::new(5), 2),
            (NodeIndex::new(6), 0),
            (NodeIndex::new(7), 0),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_two_color_directed() {
        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 4)];

        let graph = DiGraph::<i32, i32>::from_edges(edge_list);
        let coloring = two_color(&graph).unwrap();
        let mut expected_colors = DictMap::new();
        expected_colors.insert(NodeIndex::new(0), 1);
        expected_colors.insert(NodeIndex::new(1), 0);
        expected_colors.insert(NodeIndex::new(2), 1);
        expected_colors.insert(NodeIndex::new(3), 0);
        expected_colors.insert(NodeIndex::new(4), 1);
        assert_eq!(coloring, expected_colors)
    }

    #[test]
    fn test_two_color_directed_not_bipartite() {
        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 0), (3, 1)];

        let graph = DiGraph::<i32, i32>::from_edges(edge_list);
        let coloring = two_color(&graph);
        assert_eq!(None, coloring)
    }

    #[test]
    fn test_two_color_undirected_not_bipartite() {
        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 0), (3, 1)];

        let graph = UnGraph::<i32, i32>::from_edges(edge_list);
        let coloring = two_color(&graph);
        assert_eq!(None, coloring)
    }

    #[test]
    fn test_two_color_directed_with_isolates() {
        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 4)];

        let mut graph = DiGraph::<i32, i32>::from_edges(edge_list);
        graph.add_node(10);
        graph.add_node(11);
        let coloring = two_color(&graph).unwrap();
        let mut expected_colors = DictMap::new();
        expected_colors.insert(NodeIndex::new(0), 1);
        expected_colors.insert(NodeIndex::new(1), 0);
        expected_colors.insert(NodeIndex::new(2), 1);
        expected_colors.insert(NodeIndex::new(3), 0);
        expected_colors.insert(NodeIndex::new(4), 1);
        expected_colors.insert(NodeIndex::new(5), 1);
        expected_colors.insert(NodeIndex::new(6), 1);
        assert_eq!(coloring, expected_colors)
    }

    #[test]
    fn test_two_color_undirected_with_isolates() {
        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 4)];

        let mut graph = UnGraph::<i32, i32>::from_edges(edge_list);
        graph.add_node(10);
        graph.add_node(11);
        let coloring = two_color(&graph).unwrap();
        let mut expected_colors = DictMap::new();
        expected_colors.insert(NodeIndex::new(0), 1);
        expected_colors.insert(NodeIndex::new(1), 0);
        expected_colors.insert(NodeIndex::new(2), 1);
        expected_colors.insert(NodeIndex::new(3), 0);
        expected_colors.insert(NodeIndex::new(4), 1);
        expected_colors.insert(NodeIndex::new(5), 1);
        expected_colors.insert(NodeIndex::new(6), 1);
        assert_eq!(coloring, expected_colors)
    }

    #[test]
    fn test_path_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            path_graph(Some(7), None, || (), || (), false).unwrap();
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        for strategy in vec![
            ColoringStrategy::Degree,
            ColoringStrategy::Saturation,
            ColoringStrategy::IndependentSet,
        ] {
            let colors =
                greedy_node_color_with_coloring_strategy(&graph, preset_color_fn, strategy)
                    .unwrap();
            check_node_colors(&graph, &colors);
        }
    }

    #[test]
    fn test_cycle_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            cycle_graph(Some(15), None, || (), || (), false).unwrap();
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        for strategy in vec![
            ColoringStrategy::Degree,
            ColoringStrategy::Saturation,
            ColoringStrategy::IndependentSet,
        ] {
            let colors =
                greedy_node_color_with_coloring_strategy(&graph, preset_color_fn, strategy)
                    .unwrap();
            check_node_colors(&graph, &colors);
        }
    }

    #[test]
    fn test_heavy_hex_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            heavy_hex_graph(7, || (), || (), false).unwrap();
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        for strategy in vec![
            ColoringStrategy::Degree,
            ColoringStrategy::Saturation,
            ColoringStrategy::IndependentSet,
        ] {
            let colors =
                greedy_node_color_with_coloring_strategy(&graph, preset_color_fn, strategy)
                    .unwrap();
            check_node_colors(&graph, &colors);
        }
    }

    #[test]
    fn test_complete_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            complete_graph(Some(10), None, || (), || ()).unwrap();
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        for strategy in vec![
            ColoringStrategy::Degree,
            ColoringStrategy::Saturation,
            ColoringStrategy::IndependentSet,
        ] {
            let colors =
                greedy_node_color_with_coloring_strategy(&graph, preset_color_fn, strategy)
                    .unwrap();
            check_node_colors(&graph, &colors);
        }
    }
}

#[cfg(test)]
mod test_edge_coloring {
    use crate::coloring::{
        greedy_edge_color, greedy_edge_color_with_coloring_strategy, ColoringStrategy,
    };
    use crate::dictmap::DictMap;
    use crate::petgraph::Graph;
    use std::convert::Infallible;

    use petgraph::graph::{edge_index, EdgeIndex};
    use petgraph::Undirected;

    #[test]
    fn test_legacy_greedy_edge_color_empty_graph() {
        // Empty graph
        let graph = Graph::<(), (), Undirected>::new_undirected();
        let colors = greedy_edge_color(&graph);
        let expected_colors: DictMap<EdgeIndex, usize> = [].into_iter().collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_legacy_greedy_edge_color_simple_graph() {
        // Graph with an edge removed
        let graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3)]);
        let colors = greedy_edge_color(&graph);
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 1),
            (EdgeIndex::new(1), 0),
            (EdgeIndex::new(2), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_legacy_greedy_edge_color_graph_with_removed_edges() {
        // Simple graph
        let mut graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3), (3, 0)]);
        graph.remove_edge(edge_index(1));
        let colors = greedy_edge_color(&graph);
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 1),
            (EdgeIndex::new(1), 0),
            (EdgeIndex::new(2), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_empty_graph() {
        // Empty graph
        let graph = Graph::<(), (), Undirected>::new_undirected();
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        let colors = greedy_edge_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Degree,
        )
        .unwrap();
        let expected_colors: DictMap<EdgeIndex, usize> = [].into_iter().collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_simple_graph() {
        // Graph with an edge removed
        let graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3)]);
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        let colors = greedy_edge_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Degree,
        )
        .unwrap();
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 1),
            (EdgeIndex::new(1), 0),
            (EdgeIndex::new(2), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_graph_with_removed_edges() {
        // Simple graph
        let mut graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3), (3, 0)]);
        graph.remove_edge(edge_index(1));

        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        let colors = greedy_edge_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Degree,
        )
        .unwrap();

        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 1),
            (EdgeIndex::new(1), 0),
            (EdgeIndex::new(2), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_degree() {
        // Simple graph
        let graph =
            Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]);
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        let colors = greedy_edge_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Degree,
        )
        .unwrap();
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 1),
            (EdgeIndex::new(1), 0),
            (EdgeIndex::new(2), 1),
            (EdgeIndex::new(3), 0),
            (EdgeIndex::new(4), 2),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_degree_with_preset() {
        // Simple graph
        let graph =
            Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]);

        let preset_color_fn = |node_idx: EdgeIndex| -> Result<Option<usize>, Infallible> {
            if node_idx.index() == 1 {
                Ok(Some(1))
            } else {
                Ok(None)
            }
        };

        let colors = greedy_edge_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Degree,
        )
        .unwrap();
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 0),
            (EdgeIndex::new(1), 1),
            (EdgeIndex::new(2), 0),
            (EdgeIndex::new(3), 1),
            (EdgeIndex::new(4), 2),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_saturation() {
        // Simple graph
        let graph =
            Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]);
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        let colors = greedy_edge_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Saturation,
        )
        .unwrap();
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 1),
            (EdgeIndex::new(1), 0),
            (EdgeIndex::new(2), 1),
            (EdgeIndex::new(3), 0),
            (EdgeIndex::new(4), 2),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_saturation_with_preset() {
        // Simple graph
        let graph =
            Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]);

        let preset_color_fn = |node_idx: EdgeIndex| -> Result<Option<usize>, Infallible> {
            if node_idx.index() == 1 {
                Ok(Some(1))
            } else if node_idx.index() == 4 {
                Ok(Some(0))
            } else {
                Ok(None)
            }
        };

        let colors = greedy_edge_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::Saturation,
        )
        .unwrap();
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 0),
            (EdgeIndex::new(1), 1),
            (EdgeIndex::new(2), 2),
            (EdgeIndex::new(3), 1),
            (EdgeIndex::new(4), 0),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_independent_set() {
        // Simple graph
        let graph =
            Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]);
        let preset_color_fn = |_| Ok::<Option<usize>, Infallible>(None);

        let colors = greedy_edge_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::IndependentSet,
        )
        .unwrap();
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 0),
            (EdgeIndex::new(1), 1),
            (EdgeIndex::new(2), 2),
            (EdgeIndex::new(3), 1),
            (EdgeIndex::new(4), 0),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_independent_set_with_preset() {
        // Simple graph
        let graph =
            Graph::<(), (), Undirected>::from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]);

        let preset_color_fn = |node_idx: EdgeIndex| -> Result<Option<usize>, Infallible> {
            if node_idx.index() == 1 {
                Ok(Some(0))
            } else if node_idx.index() == 4 {
                Ok(Some(2))
            } else {
                Ok(None)
            }
        };

        let colors = greedy_edge_color_with_coloring_strategy(
            &graph,
            preset_color_fn,
            ColoringStrategy::IndependentSet,
        )
        .unwrap();
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 1),
            (EdgeIndex::new(1), 0),
            (EdgeIndex::new(2), 1),
            (EdgeIndex::new(3), 0),
            (EdgeIndex::new(4), 2),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }
}

#[cfg(test)]
mod test_misra_gries_edge_coloring {
    use crate::coloring::misra_gries_edge_color;
    use crate::dictmap::DictMap;
    use crate::generators::{complete_graph, cycle_graph, heavy_hex_graph, path_graph};
    use crate::petgraph::Graph;

    use hashbrown::HashSet;
    use petgraph::graph::EdgeIndex;
    use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeIndexable};
    use petgraph::Undirected;
    use std::fmt::Debug;
    use std::hash::Hash;

    fn check_edge_coloring<G>(graph: G, colors: &DictMap<G::EdgeId, usize>)
    where
        G: NodeIndexable + IntoEdges + IntoNodeIdentifiers,
        G::EdgeId: Eq + Hash + Debug,
    {
        // Check that every edge has valid color
        for edge in graph.edge_references() {
            if !colors.contains_key(&edge.id()) {
                panic!("Problem: edge {:?} has no color assigned.", &edge.id());
            }
        }

        // Check that for every node all of its edges have different colors
        // (i.e. the number of used colors is equal to the degree).
        // Also compute maximum color used and maximum node degree.
        let mut max_color = 0;
        let mut max_node_degree = 0;
        let node_indices: Vec<G::NodeId> = graph.node_identifiers().collect();
        for node in node_indices {
            let mut cur_node_degree = 0;
            let mut used_colors: HashSet<usize> = HashSet::new();
            for edge in graph.edges(node) {
                let color = colors.get(&edge.id()).unwrap();
                used_colors.insert(*color);
                cur_node_degree += 1;
                if max_color < *color {
                    max_color = *color;
                }
            }
            if used_colors.len() < cur_node_degree {
                panic!(
                    "Problem: node {:?} does not have enough colors.",
                    NodeIndexable::to_index(&graph, node)
                );
            }

            if cur_node_degree > max_node_degree {
                max_node_degree = cur_node_degree
            }
        }

        // Check that number of colors used is at most max_node_degree + 1
        // (note that number of colors is max_color + 1).
        if max_color > max_node_degree {
            panic!(
                "Problem: too many colors are used ({} colors used, {} max node degree)",
                max_color + 1,
                max_node_degree
            );
        }
    }

    #[test]
    fn test_simple_graph() {
        let graph = Graph::<(), (), Undirected>::from_edges([(0, 1), (0, 2), (0, 3), (3, 4)]);
        let colors = misra_gries_edge_color(&graph);
        check_edge_coloring(&graph, &colors);

        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 0),
            (EdgeIndex::new(1), 2),
            (EdgeIndex::new(2), 1),
            (EdgeIndex::new(3), 3),
        ]
        .into_iter()
        .collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_path_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            path_graph(Some(7), None, || (), || (), false).unwrap();
        let colors = misra_gries_edge_color(&graph);
        check_edge_coloring(&graph, &colors);
    }

    #[test]
    fn test_cycle_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            cycle_graph(Some(15), None, || (), || (), false).unwrap();
        let colors = misra_gries_edge_color(&graph);
        check_edge_coloring(&graph, &colors);
    }

    #[test]
    fn test_heavy_hex_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            heavy_hex_graph(7, || (), || (), false).unwrap();
        let colors = misra_gries_edge_color(&graph);
        check_edge_coloring(&graph, &colors);
    }

    #[test]
    fn test_complete_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            complete_graph(Some(10), None, || (), || ()).unwrap();
        let colors = misra_gries_edge_color(&graph);
        check_edge_coloring(&graph, &colors);
    }
}
