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

use crate::{graph, StablePyGraph};
use hashbrown::HashSet;
use rustworkx_core::coloring::greedy_node_color;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;
use rustworkx_core::dictmap::DictMap;

use petgraph::visit::IntoEdgeReferences;
use petgraph::Undirected;

/// Color a :class:`~.PyGraph` object using a greedy graph coloring algorithm.
///
/// This function uses a `largest-first` strategy as described in [1]_ and colors
/// the nodes with higher degree first.
///
/// .. note::
///
///     The coloring problem is NP-hard and this is a heuristic algorithm which
///     may not return an optimal solution.
///
/// :param PyGraph: The input PyGraph object to color
///
/// :returns: A dictionary where keys are node indices and the value is
///     the color
/// :rtype: dict
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///     from rustworkx.visualization import mpl_draw
///
///     graph = rx.generators.generalized_petersen_graph(5, 2)
///     coloring = rx.graph_greedy_color(graph)
///     colors = [coloring[node] for node in graph.node_indices()]
///
///     # Draw colored graph
///     layout = rx.shell_layout(graph, nlist=[[0, 1, 2, 3, 4],[6, 7, 8, 9, 5]])
///     mpl_draw(graph, node_color=colors, pos=layout)
///
///
/// .. [1] Adrian Kosowski, and Krzysztof Manuszewski, Classical Coloring of Graphs,
///     Graph Colorings, 2-19, 2004. ISBN 0-8218-3458-4.
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_greedy_color(py: Python, graph: &graph::PyGraph) -> PyResult<PyObject> {
    let colors = greedy_node_color(&graph.graph);
    let out_dict = PyDict::new(py);
    for (node, color) in colors {
        out_dict.set_item(node.index(), color)?;
    }
    Ok(out_dict.into())
}

struct MisraGriesAlgorithm<'a> {
    graph: &'a StablePyGraph<Undirected>,
    colors: Vec<Option<usize>>,
}

impl<'a> MisraGriesAlgorithm<'a> {
    pub fn new(graph: &'a StablePyGraph<Undirected>) -> Self {
        let colors = vec![None; graph.edge_count()];
        MisraGriesAlgorithm { graph, colors }
    }

    // Computes colors used at node u
    fn get_used_colors(&self, u: NodeIndex) -> HashSet<usize> {
        let used_colors: HashSet<usize> = self
            .graph
            .edges(u)
            .filter_map(|edge| self.colors[edge.id().index()])

            .collect();
        used_colors
    }

    // Returns the smallest free (aka unused) color at node u
    fn get_free_color(&self, u: NodeIndex) -> usize {
        let used_colors = self.get_used_colors(u);
        let free_color: usize = (0..)
            .position(|color| !used_colors.contains(&color))
            .unwrap();
        free_color
    }

    // Returns if color c is free at node u
    fn is_free_color(&self, u: NodeIndex, c: usize) -> bool {
        let used_colors = self.get_used_colors(u);
        !used_colors.contains(&c)
    }

    // Returns the maximal fan on edge ee = (u, v) at u
    fn get_maximal_fan(
        &self,
        ee: EdgeIndex,
        u: NodeIndex,
        v: NodeIndex,
    ) -> Vec<(EdgeIndex, NodeIndex)> {
        let mut fan: Vec<(EdgeIndex, NodeIndex)> = Vec::new();
        fan.push((ee, v));

        let mut neighbors: Vec<(EdgeIndex, NodeIndex)> = self.graph.edges(u).map(
            |edge| (edge.id(), edge.target())
        ).collect();

        let mut last_node = v;
        let position_v = neighbors.iter().position(|x| x.1 == v).unwrap();
        neighbors.remove(position_v);

        let mut fan_extended: bool = true;
        while fan_extended {
            fan_extended = false;

            for (edge_index, z) in &neighbors {
                if let Some(color) = self.colors[edge_index.index()] {
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
    fn get_cdu_path(&self, u: NodeIndex, c: usize, d: usize) -> Vec<(EdgeIndex, usize)> {
        let mut path: Vec<(EdgeIndex, usize)> = Vec::new();
        let mut cur_node: NodeIndex = u;
        let mut cur_color = c;
        let mut path_extended = true;

        while path_extended {
            path_extended = false;
            for edge in self.graph.edges(cur_node) {
                if let Some(color) = self.colors[edge.id().index()] {
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
            match self.colors[edge.id().index()] {
                Some(_color) => (),
                None => {
                    println!("Problem edge {:?} has no color assigned", edge);
                    return false;
                }
            }
        }

        let mut max_color = 0;
        for node in self.graph.node_indices() {
            let mut used_colors: HashSet<usize> = HashSet::new();
            let mut num_edges = 0;
            for edge in self.graph.edges(node) {
                num_edges += 1;
                match self.colors[edge.id().index()] {
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
        println!("run_algorithm!");
        for edge in self.graph.edge_references() {
            let u: NodeIndex = edge.source();
            let v: NodeIndex = edge.target();
            let fan = self.get_maximal_fan(edge.id(), u, v);
            let c = self.get_free_color(u);
            let d = self.get_free_color(fan.last().unwrap().1);

            // find cdu-path
            let cdu_path = self.get_cdu_path(u, d, c);

            // invert colors on cdu-path
            for (edge_index, color) in cdu_path {
                let flipped_color = self.flip_color(c, d, color);
                self.colors[edge_index.index()] = Some(flipped_color);
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
                let next_color = self.colors[fan[i].0.index()].unwrap();
                let edge_id = fan[i-1].0;
                self.colors[edge_id.index()] = Some(next_color);
            }

            // fill additional color
            let edge_id = fan[w].0;
            self.colors[edge_id.index()] = Some(d);
        }

        // self.check_coloring();


        &self.colors
    }
}

#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_misra_gries_edge_color(py: Python, graph: &graph::PyGraph) -> PyResult<PyObject> {
    let mut mg = MisraGriesAlgorithm::new(&graph.graph);

    let colors = mg.run_algorithm();

    let out_dict = PyDict::new(py);
    for (edge, color) in colors.iter().enumerate() {
        out_dict.set_item(edge, color.unwrap())?;
    }
    Ok(out_dict.into())
}
