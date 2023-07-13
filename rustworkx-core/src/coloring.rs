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
use petgraph::visit::{EdgeCount, EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount};
use rayon::prelude::*;

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

#[cfg(test)]

mod test_node_coloring {

    use crate::coloring::greedy_node_color;
    use crate::dictmap::DictMap;
    use crate::petgraph::Graph;

    use petgraph::graph::NodeIndex;
    use petgraph::Undirected;

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
