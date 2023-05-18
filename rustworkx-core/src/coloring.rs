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
use hashbrown::{HashMap, HashSet};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount};
use rayon::prelude::*;

/// Color a graph using a greedy graph coloring algorithm.
///
/// This function uses a `largest-first` strategy as described in [1]_ and colors
/// the nodes with higher degree first.
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
/// use rustworkx_core::coloring::greedy_color;
///
/// let g = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2)]);
/// let colors = greedy_color(&g);
/// let mut expected_colors = DictMap::new();
/// expected_colors.insert(NodeIndex::new(0), 0);
/// expected_colors.insert(NodeIndex::new(1), 1);
/// expected_colors.insert(NodeIndex::new(2), 1);
/// assert_eq!(colors, expected_colors);
/// ```
///
///
/// .. [1] Adrian Kosowski, and Krzysztof Manuszewski, Classical Coloring of Graphs,
///     Graph Colorings, 2-19, 2004. ISBN 0-8218-3458-4.
pub fn greedy_color<G>(graph: G) -> DictMap<G::NodeId, usize>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::NodeId: Hash + Eq + Send + Sync,
{
    let mut colors: DictMap<G::NodeId, usize> = DictMap::new();
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

#[cfg(test)]

mod test_coloring {

    use crate::coloring::greedy_color;
    use crate::dictmap::DictMap;
    use crate::petgraph::Graph;

    use petgraph::graph::NodeIndex;
    use petgraph::Undirected;

    #[test]
    fn test_greedy_color_empty_graph() {
        // Empty graph
        let graph = Graph::<(), (), Undirected>::new_undirected();
        let colors = greedy_color(&graph);
        let expected_colors: DictMap<NodeIndex, usize> = [].into_iter().collect();
        assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_greedy_color_simple_graph() {
        // Simple graph
        let graph = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2)]);
        let colors = greedy_color(&graph);
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
    fn test_greedy_color_simple_graph_large_degree() {
        // Graph with multiple edges
        let graph = Graph::<(), (), Undirected>::from_edges(&[
            (0, 1),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
        ]);
        let colors = greedy_color(&graph);
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
