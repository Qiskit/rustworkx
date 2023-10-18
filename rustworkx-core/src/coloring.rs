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

use std::cmp::Reverse;
use std::hash::Hash;

use crate::dictmap::*;
use crate::line_graph::line_graph;

use hashbrown::{HashMap, HashSet};
use petgraph::graph::NodeIndex;
use petgraph::visit::{
    EdgeCount, EdgeRef, GraphBase, GraphProp, IntoEdges, IntoNeighborsDirected,
    IntoNodeIdentifiers, NodeCount, NodeIndexable, EdgeIndexable,
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
    let mut colors: DictMap<G::NodeId, usize> = DictMap::with_capacity(graph.node_count());
    let mut node_vec: Vec<G::NodeId> = graph.node_identifiers().collect();

    let mut sort_map: HashMap<G::NodeId, usize> = HashMap::with_capacity(graph.node_count());
    for k in node_vec.iter() {
        sort_map.insert(*k, graph.edges(*k).count());
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

    colors
}

/// Color edges of a graph using a greedy approach.
///
/// This function works by greedily coloring the line graph of the given graph.
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

struct MisraGries<G: GraphBase> {
    // The input graph
    graph: G,
    // Partially assigned colors (indexed by internal edge index)
    colors: Vec<Option<usize>>,
}

impl<G> MisraGries<G>
where
    G: EdgeIndexable + IntoEdges,
{
    pub fn new(graph: G) -> Self {
        let colors = vec![None; graph.edge_bound()];
        MisraGries { graph, colors }
    }

    // Computes colors used at node u
    fn get_used_colors(&self, u: G::NodeId) -> HashSet<usize> {
        let used_colors: HashSet<usize> = self
            .graph
            .edges(u)
            .filter_map(|edge| self.colors[EdgeIndexable::to_index(&self.graph, edge.id())])
            .collect();
        used_colors
    }

    // Returns the smallest free (aka unused) color at node u
    fn get_free_color(&self, u: G::NodeId) -> usize {
        let used_colors = self.get_used_colors(u);
        let free_color: usize = (0..)
            .position(|color| !used_colors.contains(&color))
            .unwrap();
        free_color
    }

    // Returns true iff color c is free at node u
    fn is_free_color(&self, u: G::NodeId, c: usize) -> bool {
        let used_colors = self.get_used_colors(u);
        !used_colors.contains(&c)
    }

    // Returns the maximal fan on edge (u, v) at u
    fn get_maximal_fan(
        &self,
        eid: G::EdgeId,
        u: G::NodeId,
        v: G::NodeId,
    ) -> Vec<(G::EdgeId, G::NodeId)> {
        let mut fan: Vec<(G::EdgeId, G::NodeId)> = Vec::new();
        fan.push((eid, v));

        let mut neighbors: Vec<(G::EdgeId, G::NodeId)> = self
            .graph
            .edges(u)
            .map(|edge| (edge.id(), edge.target()))
            .collect();

        let mut last_node_in_fan = v;
        neighbors.remove(neighbors.iter().position(|x| x.1 == v).unwrap());

        let mut fan_extended: bool = true;
        while fan_extended {
            fan_extended = false;

            for (edge, node) in &neighbors {
                if let Some(color) = self.colors[EdgeIndexable::to_index(&self.graph, *edge)] {
                    if self.is_free_color(last_node_in_fan, color) {
                        fan.push((*edge, *node));
                        last_node_in_fan = *node;
                        fan_extended = true;
                        neighbors.remove(neighbors.iter().position(|x| x.1 == *node).unwrap());
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
    fn get_cdu_path(&self, u: G::NodeId, c: usize, d: usize) -> Vec<(G::EdgeId, usize)> {
        let mut path: Vec<(G::EdgeId, usize)> = Vec::new();
        let mut cur_node: G::NodeId = u;
        let mut cur_color = c;
        let mut path_extended = true;

        while path_extended {
            path_extended = false;
            for edge in self.graph.edges(cur_node) {
                if let Some(color) = self.colors[EdgeIndexable::to_index(&self.graph, edge.id())] {
                    if color == cur_color {
                        path_extended = true;
                        path.push((edge.id(), cur_color));
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
            let fan = self.get_maximal_fan(edge.id(), u, v);
            let c = self.get_free_color(u);
            let d = self.get_free_color(fan.last().unwrap().1);

            // find cdu-path
            let cdu_path = self.get_cdu_path(u, d, c);

            // invert colors on cdu-path
            for (cdu_edge, color) in cdu_path {
                let flipped_color = self.flip_color(c, d, color);
                self.colors[EdgeIndexable::to_index(&self.graph, cdu_edge)] = Some(flipped_color);
            }

            // find sub-fan fan[0..w] such that d is free on fan[w]
            let mut w = 0;
            for (i, (_, z)) in fan.iter().enumerate() {
                if self.is_free_color(*z, d) {
                    w = i;
                    break;
                }
            }

            // rotate fan
            for i in 1..w + 1 {
                let next_color =
                    self.colors[EdgeIndexable::to_index(&self.graph, fan[i].0)].unwrap();
                self.colors[EdgeIndexable::to_index(&self.graph, fan[i - 1].0)] = Some(next_color);
            }

            // fill additional color
            self.colors[EdgeIndexable::to_index(&self.graph, fan[w].0)] = Some(d);
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
    G: EdgeIndexable + IntoEdges + EdgeCount,
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

    use crate::coloring::greedy_node_color;
    use crate::coloring::two_color;
    use crate::dictmap::*;
    use crate::petgraph::prelude::*;

    use crate::petgraph::graph::NodeIndex;
    use crate::petgraph::Undirected;

    #[test]
    fn test_greedy_node_color_empty_graph() {
        // Empty graph
        let graph = Graph::<(), (), Undirected>::new_undirected();
        let colors = greedy_node_color(&graph);
        let expected_colors: DictMap<NodeIndex, usize> = [].into_iter().collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_node_color_simple_graph() {
        // Simple graph
        let graph = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2)]);
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
    fn test_greedy_node_color_simple_graph_large_degree() {
        // Graph with multiple edges
        let graph = Graph::<(), (), Undirected>::from_edges(&[
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
    fn test_two_color_directed() {
        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 4)];

        let graph = DiGraph::<i32, i32>::from_edges(&edge_list);
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

        let graph = DiGraph::<i32, i32>::from_edges(&edge_list);
        let coloring = two_color(&graph);
        assert_eq!(None, coloring)
    }

    #[test]
    fn test_two_color_undirected_not_bipartite() {
        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 0), (3, 1)];

        let graph = UnGraph::<i32, i32>::from_edges(&edge_list);
        let coloring = two_color(&graph);
        assert_eq!(None, coloring)
    }

    #[test]
    fn test_two_color_directed_with_isolates() {
        let edge_list = vec![(0, 1), (1, 2), (2, 3), (3, 4)];

        let mut graph = DiGraph::<i32, i32>::from_edges(&edge_list);
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

        let mut graph = UnGraph::<i32, i32>::from_edges(&edge_list);
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
}

#[cfg(test)]
mod test_edge_coloring {
    use crate::coloring::greedy_edge_color;
    use crate::dictmap::DictMap;
    use crate::petgraph::Graph;

    use petgraph::graph::{edge_index, EdgeIndex};
    use petgraph::Undirected;

    #[test]
    fn test_greedy_edge_color_empty_graph() {
        // Empty graph
        let graph = Graph::<(), (), Undirected>::new_undirected();
        let colors = greedy_edge_color(&graph);
        let expected_colors: DictMap<EdgeIndex, usize> = [].into_iter().collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_edge_color_simple_graph() {
        // Graph with an edge removed
        let graph = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (1, 2), (2, 3)]);
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
    fn test_greedy_edge_color_graph_with_removed_edges() {
        // Simple graph
        let mut graph = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)]);
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
}

#[cfg(test)]
mod test_misra_gries_edge_coloring {
    use crate::coloring::misra_gries_edge_color;
    use crate::dictmap::DictMap;
    use crate::generators::{cycle_graph, heavy_hex_graph, path_graph};
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
        let graph = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2), (0, 3), (3, 4)]);
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
}
