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
use rayon::prelude::*;
use petgraph::visit::{
    EdgeCount,
    EdgeIndexable,
    EdgeRef,
    GraphBase,
    GraphProp, // allows is_directed
    IntoEdges,
    IntoEdgesDirected,
    IntoNeighbors,
    IntoNeighborsDirected,
    IntoNodeIdentifiers,
    NodeCount,
    NodeIndexable,
    Reversed,
    Visitable,
};
use core::fmt::Debug;



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

struct MisraGries<G: GraphBase>
{
    // The input graph
    graph: G,
    //
    colors: Vec<Option<usize>>,
}

impl<G> MisraGries<G>
where
    G: NodeIndexable + EdgeIndexable + IntoEdges + Sync + EdgeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Debug,
    G::EdgeId: Eq + Hash + Debug,
    G::EdgeRef: Debug,
{
    pub fn new(graph: G) -> Self {
        // todo: change to edge_bound
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
        println!("get_free_color: {:?} -> {:?}", u, free_color);
        free_color
    }

    // Returns if color c is free at node u
    fn is_free_color(&self, u: G::NodeId, c: usize) -> bool {
        let used_colors = self.get_used_colors(u);
        !used_colors.contains(&c)
    }

    // Returns the maximal fan on edge ee = (u, v) at u
    fn get_maximal_fan(
        &self,
        ee: G::EdgeId,
        u: G::NodeId,
        v: G::NodeId,
    ) -> Vec<(G::EdgeId, G::NodeId)>


    {
        let mut fan: Vec<(G::EdgeId, G::NodeId)> = Vec::new();
        fan.push((ee, v));

        let mut neighbors: Vec<(G::EdgeId, G::NodeId)> = self
            .graph
            .edges(u)
            .map(|edge| (edge.id(), edge.target()))
            .collect();

        let mut last_node = v;
        let position_v = neighbors.iter().position(|x| x.1 == v).unwrap();
        neighbors.remove(position_v);

        let mut fan_extended: bool = true;
        while fan_extended {
            fan_extended = false;

            for (edge_index, z) in &neighbors {

                let eee = EdgeIndexable::to_index(&self.graph, *edge_index);
                if let Some(color) = self.colors[eee] {
                    if self.is_free_color(last_node, color) {
                        fan_extended = true;
                        last_node = *z;
                        fan.push((*edge_index, *z));
                        let position_z = neighbors.iter().position(|x| x.1 == *z).unwrap();
                        neighbors.remove(position_z);
                        break;
                    }
                }
            }

            // for (position, (edge_index, z)) in neighbors.iter().enumerate() {
            //     if let Some(color) = self.colors.get(edge_index) {
            //         if self.is_free_color(last_node, *color) {
            //             fan_extended = true;
            //             last_node = *z;
            //             fan.push((*edge_index, *z));
            //             neighbors.remove(position);
            //             break;
            //         }
            //     }
            // }
        }

        fan
    }

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

                let eee = EdgeIndexable::to_index(&self.graph, edge.id());
                if let Some(color) = self.colors[eee] {
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

    fn check_coloring(&self) -> bool {
        for edge in self.graph.edge_references() {
            let eee = EdgeIndexable::to_index(&self.graph, edge.id());
            match self.colors[eee] {
                Some(_color) => (),
                None => {
                    println!("Problem edge {:?} has no color assigned", edge);
                    return false;
                }
            }
        }

        let mut max_color = 0;

        let node_indices: Vec<G::NodeId> = self.graph.node_identifiers().collect();

        for node in node_indices {
            let mut used_colors: HashSet<usize> = HashSet::new();
            let mut num_edges = 0;
            for edge in self.graph.edges(node) {
                num_edges += 1;
                let eee = EdgeIndexable::to_index(&self.graph, edge.id());

                match self.colors[eee] {


                    Some(color) => {
                        used_colors.insert(color);
                        if max_color < color {
                            max_color = color;
                        }
                    }
                    None => {
                        println!("Problem: edge {:?} has no color assigned", edge);
                        return false;
                    }
                }
            }
            if used_colors.len() < num_edges {
                println!("Problem: node {:?} does not have enough colors", node);
                return false;
            }
        }

        println!("Coloring is OK, max_color = {}", max_color);
        true
    }

    pub fn run_algorithm(&mut self) -> &Vec<Option<usize>> {
        println!("run_algorithm 10!");
        for edge in self.graph.edge_references() {
            println!("=> processing edge {:?}", edge);
            let u: G::NodeId = edge.source();
            let v: G::NodeId = edge.target();
            let fan = self.get_maximal_fan(edge.id(), u, v);
            let c = self.get_free_color(u);
            let d = self.get_free_color(fan.last().unwrap().1);
            println!("=> u = {:?}, v = {:?}, fan = {:?}, c = {:?}, d = {:?}", u, v, fan, c, d);


            // find cdu-path
            let cdu_path = self.get_cdu_path(u, d, c);

            // invert colors on cdu-path
            for (edge_index, color) in cdu_path {
                let flipped_color = self.flip_color(c, d, color);
                let eee = EdgeIndexable::to_index(&self.graph, edge_index);
                println!("cdupath: flipping color of edge {:?} to {:?}", eee, flipped_color);
                self.colors[eee] = Some(flipped_color);
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
                let eee = EdgeIndexable::to_index(&self.graph, fan[i].0);
                let next_color = self.colors[eee].unwrap();
                let edge_id = fan[i - 1].0;
                let eee1 = EdgeIndexable::to_index(&self.graph, edge_id);
                println!("rotate: color of edge {:?} to {:?}", eee1, next_color);
                self.colors[eee1] = Some(next_color);
            }

            // fill additional color
            let edge_id = fan[w].0;
            let eee = EdgeIndexable::to_index(&self.graph, edge_id);
            println!("set: color of edge {:?} to {:?}", eee, d);

            self.colors[eee] = Some(d);
        }

        self.check_coloring();

        &self.colors
    }
}





pub fn mg_color<G>(graph: G) -> DictMap<G::EdgeId, usize>
where
    G: NodeIndexable + EdgeIndexable + IntoEdges + Sync + EdgeCount + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Debug,
    G::EdgeId: Eq + Hash + Debug,
    G::EdgeRef: Debug,

{
    println!("In mgcolor_2!");

    // todo: change to &G
    let mut mg: MisraGries<G> = MisraGries::new(graph);

    let colors = mg.run_algorithm();

    let mut edge_colors: DictMap<G::EdgeId, usize> = DictMap::with_capacity(graph.edge_count());

    for edge in graph.edge_references() {
        let edge_index = edge.id();
        let edge_index_value = EdgeIndexable::to_index(&graph, edge_index);
        let color = colors[edge_index_value].unwrap();
        edge_colors.insert(edge_index, color);
        println!("Edge with index {:?}, value {:?}, source {:?}, target {:?} ->  has color {:?}",
                 edge_index, edge_index_value, edge.source(), edge.target(), color);
    }

    // for (edge, color) in colors.iter().enumerate() {
    //     edge_colors.insert(edge, color.unwrap());
    // }

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


#[cfg(test)]
mod test_mg_coloring {
    use crate::coloring::mg_color;
    use crate::dictmap::DictMap;
    use crate::petgraph::Graph;

    use petgraph::graph::{edge_index, EdgeIndex};
    use petgraph::Undirected;
    use crate::generators::{cycle_graph, heavy_hex_graph, path_graph};

    #[test]
    fn test_simple_graph() {
        // Graph with an edge removed
        let graph = Graph::<(), (), Undirected>::from_edges(&[(0, 1), (0, 2), (0, 3), (3, 4)]);
        let colors = mg_color(&graph);
        let expected_colors: DictMap<EdgeIndex, usize> = [
            (EdgeIndex::new(0), 1),
            (EdgeIndex::new(1), 0),
            (EdgeIndex::new(2), 1),
            (EdgeIndex::new(3), 0),
            (EdgeIndex::new(4), 1),

        ]
        .into_iter()
        .collect();
        println!("colors {:?}", colors);
        // assert_eq!(colors, expected_colors);
    }

    #[test]
    fn test_path_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            path_graph(Some(7), None, || (), || (), false).unwrap();

        println!("graph = {:?}", graph);
        let colors = mg_color(&graph);

        println!("colors {:?}", colors);

    }


    #[test]
    fn test_cycle_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> =
            cycle_graph(Some(15), None, || (), || (), false).unwrap();

        println!("graph = {:?}", graph);
        let colors = mg_color(&graph);

        println!("colors {:?}", colors);

    }

    #[test]
    fn test_heavy_hex_graph() {
        let graph: petgraph::graph::UnGraph<(), ()> = heavy_hex_graph(7, || (), || (), false).unwrap();

        println!("graph = {:?}", graph);
        let colors = mg_color(&graph);

        println!("colors {:?}", colors);

    }

}


