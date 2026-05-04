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

//! Kamada-Kawai force-directed graph layout (Kamada & Kawai 1989).
//!
//! For each connected component, the layout minimises the energy
//!
//!     E = (1/2) sum_{i<j} k_{ij} (|p_i - p_j| - l_{ij})^2
//!
//! where d_{ij} is the graph-theoretic shortest path between i and j,
//! l_{ij} = d_{ij} / d_max is the desired display distance and
//! k_{ij} = 1 / d_{ij}^2 is the spring constant.  Minimisation follows
//! the original 1989 method: at each outer step the node with the
//! largest partial-gradient norm is picked and updated by a 2D Newton
//! step against the local 2x2 Hessian until its gradient drops below
//! `epsilon`; the outer loop then reselects globally and repeats.
//!
//! Disconnected graphs are handled by laying out each component
//! independently and packing the results in a horizontal row so that
//! components don't overlap.  Directed graphs have their distance
//! matrix symmetrised (Kamada-Kawai is fundamentally undirected).

use std::iter::Iterator;

use hashbrown::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use petgraph::EdgeType;
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeIndexable, EdgeRef, IntoEdgeReferences, NodeIndexable};

use rayon::prelude::*;

use rustworkx_core::connectivity::connected_components;
use rustworkx_core::shortest_path::dijkstra;

use crate::StablePyGraph;
use crate::iterators::Pos2DMapping;
use crate::weight_callable;

use super::spring::{Point, recenter, rescale};

type Nt = f64;

const LBOUND: Nt = 1e-8;

fn distance_matrix<Ty: EdgeType + Sync>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    weight_fn: &Option<Py<PyAny>>,
    default_weight: Nt,
) -> PyResult<Vec<Nt>> {
    let n_bound = graph.node_bound();

    let mut edge_weights: Vec<Option<Nt>> = vec![None; graph.edge_bound() + 1];
    for e in graph.edge_references() {
        let w = weight_callable(py, weight_fn, e.weight(), default_weight)?;
        if w < 0.0 {
            return Err(PyValueError::new_err(
                "kamada_kawai_layout requires non-negative edge weights",
            ));
        }
        edge_weights[e.id().index()] = Some(w);
    }

    let edge_cost = |e: petgraph::stable_graph::EdgeIndex| -> PyResult<Nt> {
        edge_weights[e.index()].ok_or_else(|| PyValueError::new_err("Missing edge weight"))
    };

    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    let rows: Vec<(usize, Vec<Option<Nt>>)> = nodes
        .par_iter()
        .map(|src| {
            let lengths: PyResult<Vec<Option<Nt>>> =
                dijkstra(graph, *src, None, |e| edge_cost(e.id()), None);
            (src.index(), lengths.unwrap())
        })
        .collect();

    let mut matrix = vec![Nt::INFINITY; n_bound * n_bound];
    for (i, lengths) in rows {
        matrix[i * n_bound + i] = 0.0;
        for (j, d_opt) in lengths.into_iter().enumerate() {
            if let Some(d) = d_opt {
                matrix[i * n_bound + j] = d;
            }
        }
    }

    if Ty::is_directed() {
        for i in 0..n_bound {
            for j in (i + 1)..n_bound {
                let d_ij = matrix[i * n_bound + j];
                let d_ji = matrix[j * n_bound + i];
                let d = d_ij.min(d_ji);
                matrix[i * n_bound + j] = d;
                matrix[j * n_bound + i] = d;
            }
        }
    }

    Ok(matrix)
}

fn circular_init_subset(node_indices: &[usize], pos: &mut [Point]) {
    let n = node_indices.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        pos[node_indices[0]] = [0.0, 0.0];
        return;
    }
    for (k, &i) in node_indices.iter().enumerate() {
        let theta = 2.0 * std::f64::consts::PI * (k as Nt) / (n as Nt);
        pos[i] = [theta.cos(), theta.sin()];
    }
}

#[allow(clippy::too_many_arguments)]
fn kk_solve_component(
    dist: &[Nt],
    pos: &mut [Point],
    active: &[usize],
    fixed: &HashSet<usize>,
    n_bound: usize,
    epsilon: Nt,
    max_outer: usize,
    max_inner: usize,
) {
    if active.len() < 2 {
        return;
    }

    let mut d_max: Nt = 0.0;
    for &i in active {
        for &j in active {
            if i == j {
                continue;
            }
            let d = dist[i * n_bound + j];
            if d.is_finite() && d > d_max {
                d_max = d;
            }
        }
    }
    if d_max <= 0.0 {
        return;
    }

    let movable: Vec<usize> = active
        .iter()
        .copied()
        .filter(|i| !fixed.contains(i))
        .collect();
    if movable.is_empty() {
        return;
    }

    let delta_norm = |pos: &[Point], m: usize| -> Nt {
        let mut gx = 0.0;
        let mut gy = 0.0;
        let pm = pos[m];
        for &i in active {
            if i == m {
                continue;
            }
            let d = dist[m * n_bound + i];
            if !d.is_finite() || d <= 0.0 {
                continue;
            }
            let l = d / d_max;
            let k = 1.0 / (d * d);
            let dx = pm[0] - pos[i][0];
            let dy = pm[1] - pos[i][1];
            let r = (dx * dx + dy * dy).sqrt().max(LBOUND);
            let coeff = k * (1.0 - l / r);
            gx += coeff * dx;
            gy += coeff * dy;
        }
        (gx * gx + gy * gy).sqrt()
    };

    for _ in 0..max_outer {
        let mut m = movable[0];
        let mut max_d = delta_norm(pos, m);
        for &cand in &movable[1..] {
            let d = delta_norm(pos, cand);
            if d > max_d {
                m = cand;
                max_d = d;
            }
        }
        if max_d < epsilon {
            break;
        }

        for _ in 0..max_inner {
            let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
            let (mut gx, mut gy) = (0.0, 0.0);
            let pm = pos[m];
            for &i in active {
                if i == m {
                    continue;
                }
                let d = dist[m * n_bound + i];
                if !d.is_finite() || d <= 0.0 {
                    continue;
                }
                let l = d / d_max;
                let k = 1.0 / (d * d);
                let dx = pm[0] - pos[i][0];
                let dy = pm[1] - pos[i][1];
                let r2 = (dx * dx + dy * dy).max(LBOUND * LBOUND);
                let r = r2.sqrt();
                let r3 = r2 * r;

                let coeff = k * (1.0 - l / r);
                gx += coeff * dx;
                gy += coeff * dy;

                a += k * (1.0 - l * dy * dy / r3);
                c += k * (1.0 - l * dx * dx / r3);
                b += k * l * dx * dy / r3;
            }

            let det = a * c - b * b;
            if det.abs() < LBOUND {
                let gn = (gx * gx + gy * gy).sqrt().max(LBOUND);
                pos[m][0] -= 0.1 * gx / gn;
                pos[m][1] -= 0.1 * gy / gn;
                break;
            }

            let dx_step = (-c * gx + b * gy) / det;
            let dy_step = (b * gx - a * gy) / det;
            pos[m][0] += dx_step;
            pos[m][1] += dy_step;

            if (gx * gx + gy * gy).sqrt() < epsilon {
                break;
            }
        }
    }
}

fn pack_components(pos: &mut [Point], components: &[Vec<usize>]) {
    if components.len() <= 1 {
        return;
    }

    let padding = 1.0;
    let mut x_offset: Nt = 0.0;

    for component in components {
        if component.is_empty() {
            continue;
        }

        let (mut min_x, mut max_x) = (Nt::INFINITY, Nt::NEG_INFINITY);
        let (mut min_y, mut max_y) = (Nt::INFINITY, Nt::NEG_INFINITY);
        for &i in component {
            let p = pos[i];
            if p[0] < min_x {
                min_x = p[0];
            }
            if p[0] > max_x {
                max_x = p[0];
            }
            if p[1] < min_y {
                min_y = p[1];
            }
            if p[1] > max_y {
                max_y = p[1];
            }
        }
        let cx = (min_x + max_x) / 2.0;
        let cy = (min_y + max_y) / 2.0;
        let width = max_x - min_x;
        let half_w = width / 2.0;

        let target_cx = x_offset + half_w;
        let dx = target_cx - cx;
        let dy = -cy;
        for &i in component {
            pos[i][0] += dx;
            pos[i][1] += dy;
        }
        x_offset += width + padding;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn kamada_kawai_layout<Ty: EdgeType + Sync>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    pos: Option<HashMap<usize, Point>>,
    fixed: Option<HashSet<usize>>,
    weight_fn: Option<Py<PyAny>>,
    default_weight: Nt,
    epsilon: Nt,
    max_outer: usize,
    max_inner: usize,
    scale: Option<Nt>,
    center: Option<Point>,
) -> PyResult<Pos2DMapping> {
    if fixed.is_some() && pos.is_none() {
        return Err(PyValueError::new_err("`fixed` specified but `pos` not."));
    }

    if graph.node_count() == 0 {
        return Ok(Pos2DMapping {
            pos_map: rustworkx_core::dictmap::DictMap::default(),
        });
    }

    let n_bound = graph.node_bound();
    let active: Vec<usize> = graph.node_indices().map(|n| n.index()).collect();

    let dist = distance_matrix(py, graph, &weight_fn, default_weight)?;

    let raw_components = connected_components(graph);
    let components: Vec<Vec<usize>> = raw_components
        .into_iter()
        .map(|c| {
            let mut v: Vec<usize> = c.iter().map(|n| n.index()).collect();
            v.sort_unstable();
            v
        })
        .collect();

    let mut vpos: Vec<Point> = vec![[0.0, 0.0]; n_bound];
    for component in &components {
        circular_init_subset(component, &mut vpos);
    }
    let user_provided_pos = pos.is_some();
    if let Some(provided) = pos {
        for (i, p) in provided {
            if i < n_bound {
                vpos[i] = p;
            }
        }
    }

    let fixed = fixed.unwrap_or_default();

    for component in &components {
        kk_solve_component(
            &dist, &mut vpos, component, &fixed, n_bound, epsilon, max_outer, max_inner,
        );
    }

    let should_pack = components.len() > 1 && fixed.is_empty() && !user_provided_pos;
    if should_pack {
        pack_components(&mut vpos, &components);
    }

    if fixed.is_empty() {
        if let Some(s) = scale {
            rescale(&mut vpos, s, active.clone());
        }
        if let Some(c) = center {
            recenter(&mut vpos, c);
        }
    }

    Ok(Pos2DMapping {
        pos_map: graph
            .node_indices()
            .map(|n| {
                let n = n.index();
                (n, vpos[n])
            })
            .collect(),
    })
}
