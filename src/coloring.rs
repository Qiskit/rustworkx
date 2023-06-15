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

use std::collections::HashSet;
use crate::{graph, StablePyGraph};
use rustworkx_core::coloring::greedy_node_color;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{Python};
use rustworkx_core::dictmap::DictMap;
use petgraph::graph::{EdgeIndex, NodeIndex};
use rustworkx_core::dictmap::*;
use petgraph::visit::{EdgeRef};

use petgraph::visit::{IntoEdgeReferences};
use petgraph::{Undirected};
use petgraph::adj::EdgeReference;


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





fn get_used_colors(graph: &StablePyGraph<Undirected>, edge_colors: &DictMap<EdgeIndex, usize>, u: NodeIndex) -> Vec<usize> {
    let mut used_colors: Vec<usize> = Vec::new();
    for edge in graph.edges(u) {
        match edge_colors.get(&edge.id()) {
            Some(color) => {
                used_colors.push(*color);
            },
            None => ()
        }
    }
    used_colors
}

fn get_free_color(graph: &StablePyGraph<Undirected>, edge_colors: &DictMap<EdgeIndex, usize>, u: NodeIndex) -> usize {
    let used_colors = get_used_colors(&graph, &edge_colors, u);
    // println!("used colors for {:?} are {:?}", u, used_colors);

    let mut c: usize = 0;

    while used_colors.contains(&c) {
        c = c + 1;
    }

    c
}

fn is_free_color(graph: &StablePyGraph<Undirected>, edge_colors: &DictMap<EdgeIndex, usize>, u: NodeIndex, c: usize) -> bool {
    let used_colors = get_used_colors(&graph, &edge_colors, u);
    !used_colors.contains(&c)
}



fn get_maximal_fan(graph: &StablePyGraph<Undirected>, edge_colors: &DictMap<EdgeIndex, usize>, ee: EdgeIndex, u: NodeIndex, v: NodeIndex) -> Vec<(EdgeIndex, NodeIndex)> {
    // println!("calling maximal_fan on {:?} and {:?}", u, v);
    let mut fan: Vec<(EdgeIndex, NodeIndex)> = Vec::new();
    fan.push((ee, v));
    // println!("... initial fan = {:?}", fan);

    let mut neighbors: Vec<(EdgeIndex, NodeIndex)> = Vec::new();
    for edge in graph.edges(u) {
        neighbors.push((edge.id(), edge.target()));
    }

    // graph.neighbors(u).collect();
    // println!("... {:?} has neighbors {:?}", u, neighbors);
    let mut last_node = v;

    let position_v = neighbors.iter().position(|x| x.1 == v).unwrap();
    // println!("position_v is {}", position_v);
    neighbors.remove(position_v);
    // println!("... remaining neighbors are {:?}", neighbors);

    let mut fan_extended: bool = true;
    while fan_extended {
        fan_extended = false;

        for (edge_index, z) in &neighbors {
            // println!("... examining nbd {:?} with edge_index {:?}", z, edge_index);
            match edge_colors.get(edge_index) {
                Some(color) => {
                    // println!("...... has color {:?}", color);
                    // println!("...... checking if color {:?} if free on {:?}", *color, last_node);
                    if is_free_color(&graph, &edge_colors, last_node, *color) {
                        // println!("...... free, extending");
                        fan_extended = true;
                        last_node = *z;
                        fan.push((*edge_index, *z));
                        let position_z = neighbors.iter().position(|x| x.1 == *z).unwrap();
                        neighbors.remove(position_z);
                        break;
                    } else {
                        // println!("...... not free");
                    }
                },
                None => (),
            }
        }

        // println!("... after looking at all neighbors, fan is extended: {}", fan_extended);
    }



    fan
}

fn flip_color(c: usize, d:usize, e:usize) -> usize {
    if e == c {
        d
    }
    else {
        c
    }
}


// Find a path starting from node u with alternating colors d, c, d, c, etc.
fn get_cdu_path(graph: &StablePyGraph<Undirected>, edge_colors: &DictMap<EdgeIndex, usize>, u: NodeIndex, c: usize, d: usize) -> Vec<(EdgeIndex, usize)> {
    let mut path: Vec<(EdgeIndex, usize)> = Vec::new();
    let mut cur_node: NodeIndex = u;
    let mut cur_color = d;
    let mut path_extended = true;

    while path_extended {
        path_extended = false;
        for edge in graph.edges(cur_node) {
            match edge_colors.get(&edge.id()) {
                Some(color) => {
                    if *color == cur_color {
                        // can extend the current path
                        cur_node = edge.target();
                        path.push((edge.id(), cur_color));
                        path_extended = true;
                        cur_color = flip_color(c, d, cur_color);
                        break;
                    }
                },
                _ => ()
            }
        }
    }
    path
}


fn check_coloring(graph: &StablePyGraph<Undirected>, edge_colors: &DictMap<EdgeIndex, usize>) -> bool {
    for edge in graph.edge_references() {
        match edge_colors.get(&edge.id()) {
            Some(_color) => (),
            None => { println!("Problem edge {:?} has no color assigned", edge); return false; }
        }
    }

    let mut max_color = 0;
    for node in graph.node_indices() {
        let mut used_colors: HashSet<usize> = HashSet::new();
        let mut num_edges = 0;
        for edge in graph.edges(node) {
            num_edges = num_edges + 1;
            match edge_colors.get(&edge.id()) {
                Some(color) => {
                    used_colors.insert(*color);
                    if max_color < *color {
                        max_color = *color;
                    }
                },
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

#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_misra_gries_edge_color(py: Python, graph: &graph::PyGraph) -> PyResult<PyObject> {

    // Maybe NodeId would be better
    let mut edge_colors: DictMap<EdgeIndex, usize> = DictMap::new();

    println!("================");
    println!("Initially");
    for edge in graph.graph.edge_references() {
        let u: NodeIndex = edge.source();
        let v: NodeIndex = edge.target();
        println!("==> edge_ref {:?} with id {:?} with source {:?} and target {:?}", edge, edge.id(), u, v);
    }


    for edge in graph.graph.edge_references() {
        println!("================");
        let u: NodeIndex = edge.source();
        let v: NodeIndex = edge.target();
        println!("==> edge_ref with source {:?} and target {:?}", u, v);
        let f = get_maximal_fan(&graph.graph, &edge_colors, edge.id(), u, v);
        println!("==> get_maximal_fan: u = {:?}, v = {:?}, fan = {:?}", u, v, f);
        let c = get_free_color(&graph.graph, &edge_colors, u);
        println!("==> c has free color {}", c);
        let n = f.last().unwrap().1;
        // println!("==> last element in fan is {:?}", n);
        let d = get_free_color(&graph.graph, &edge_colors, n);
        println!("==> d has free color {}", d);

        let cdu_path: Vec<(EdgeIndex, usize)> = get_cdu_path(&graph.graph, &edge_colors, u, c, d);
        println!("==> found cdu_path of length {}: {:?}", cdu_path.len(), cdu_path);

        for (edge_index, edge_color) in cdu_path {
            let new_color = flip_color(c, d, edge_color);
            println!("... setting color for {:?} to {:?}", edge_index, new_color);

            edge_colors.insert(edge_index, new_color);
        }
        println!("==> now edge colors are {:?}", edge_colors);

        let mut w = 0;
        for (i, (ee, z)) in f.iter().enumerate() {
            if is_free_color(&graph.graph, &edge_colors, *z, d) {
                w = i;
                break;
            }
        }
        println!("==> w is {}", w);


        // rotating fan
        for i in 1..w+1 {
            let e_prev = f[i-1].0;
            let e_next = f[i].0;
            let color_next = edge_colors.get(&e_next).unwrap();
            println!("... setting color for {:?} to {:?}", e_prev, color_next);
            edge_colors.insert(e_prev, *color_next);
        }
        println!("==> after rotating fan: edge colors are {:?}", edge_colors);

        let e_next = f[w].0;
        edge_colors.insert(e_next, d);
        println!("... setting color for {:?} to {:?}", e_next, d);

        println!("==> after assigning color, colors are {:?}", edge_colors);

        // break;
    }


    check_coloring(&graph.graph, &edge_colors);

    let out_dict = PyDict::new(py);
    for (edge, color) in edge_colors {
        out_dict.set_item(edge.index(), color)?;
    }
    Ok(out_dict.into())
}

