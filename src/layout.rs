// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANtIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

use std::iter::Iterator;

use hashbrown::{HashMap, HashSet};

use pyo3::prelude::*;

use petgraph::prelude::*;
use petgraph::EdgeType;

use crate::iterators::Pos2DMapping;

pub type Point = [f64; 2];

pub fn bipartite_layout<Ty: EdgeType>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
    first_nodes: HashSet<usize>,
    horizontal: Option<bool>,
    scale: Option<f64>,
    center: Option<Point>,
    aspect_ratio: Option<f64>,
) -> Pos2DMapping {
    let node_num = graph.node_count();
    if node_num == 0 {
        return Pos2DMapping {
            pos_map: HashMap::new(),
        };
    }
    let left_num = first_nodes.len();
    let right_num = node_num - left_num;
    let mut pos: Vec<Point> = Vec::with_capacity(node_num);

    let (width, height);
    if horizontal == Some(true) {
        // width and height viewed from 90 degrees clockwise rotation
        width = 1.0;
        height = match aspect_ratio {
            Some(aspect_ratio) => aspect_ratio * width,
            None => 4.0 * width / 3.0,
        };
    } else {
        height = 1.0;
        width = match aspect_ratio {
            Some(aspect_ratio) => aspect_ratio * height,
            None => 4.0 * height / 3.0,
        };
    }

    let x_offset = width / 2.0;
    let y_offset = height / 2.0;
    let left_dy = height / (left_num - 1) as f64;
    let right_dy = height / (right_num - 1) as f64;

    let mut lc = 0;
    let mut rc = 0;

    for node in graph.node_indices() {
        let n = node.index();

        let (x, y);
        if first_nodes.contains(&n) {
            x = -x_offset;
            y = lc as f64 * left_dy - y_offset;
            lc += 1;
        } else {
            x = width - x_offset;
            y = rc as f64 * right_dy - y_offset;
            rc += 1;
        }

        if horizontal == Some(true) {
            pos.push([-y, x]);
        } else {
            pos.push([x, y]);
        }
    }

    rescale(&mut pos, scale);
    recenter(&mut pos, center);

    Pos2DMapping {
        pos_map: graph.node_indices().map(|n| n.index()).zip(pos).collect(),
    }
}

pub fn circular_layout<Ty: EdgeType>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {
    let node_num = graph.node_count();
    let mut pos: Vec<Point> = Vec::with_capacity(node_num);
    let pi = std::f64::consts::PI;

    if node_num == 1 {
        pos.push([0.0, 0.0])
    } else {
        for i in 0..node_num {
            let angle = 2.0 * pi * i as f64 / node_num as f64;
            pos.push([angle.cos(), angle.sin()]);
        }
    }

    rescale(&mut pos, scale);
    recenter(&mut pos, center);

    Pos2DMapping {
        pos_map: graph.node_indices().map(|n| n.index()).zip(pos).collect(),
    }
}

pub fn shell_layout<Ty: EdgeType>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
    nlist: Option<Vec<Vec<usize>>>,
    rotate: Option<f64>,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {
    let node_num = graph.node_count();
    let mut pos: Vec<Point> = vec![[0.0, 0.0]; node_num];
    let pi = std::f64::consts::PI;

    let shell_list: Vec<Vec<usize>> = match nlist {
        Some(nlist) => nlist,
        None => vec![(0..node_num).collect()],
    };
    let shell_num = shell_list.len();

    let radius_bump = match scale {
        Some(scale) => scale / shell_num as f64,
        None => 1.0 / shell_num as f64,
    };

    let mut radius = match node_num {
        1 => 0.0,
        _ => radius_bump,
    };

    let rot_angle = match rotate {
        Some(rotate) => rotate,
        None => pi / shell_num as f64,
    };

    let mut first_theta = rot_angle;
    for shell in shell_list {
        let shell_len = shell.len();
        for i in 0..shell_len {
            let angle = 2.0 * pi * i as f64 / shell_len as f64 + first_theta;
            pos[shell[i]] = [radius * angle.cos(), radius * angle.sin()];
        }
        radius += radius_bump;
        first_theta += rot_angle;
    }

    recenter(&mut pos, center);

    Pos2DMapping {
        pos_map: graph.node_indices().map(|n| n.index()).zip(pos).collect(),
    }
}

pub fn spiral_layout<Ty: EdgeType>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
    scale: Option<f64>,
    center: Option<Point>,
    resolution: Option<f64>,
    equidistant: Option<bool>,
) -> Pos2DMapping {
    let node_num = graph.node_count();
    let mut pos: Vec<Point> = Vec::with_capacity(node_num);

    let ros = resolution.unwrap_or(0.35);

    if node_num == 1 {
        pos.push([0.0, 0.0]);
    } else if equidistant == Some(true) {
        let mut theta: f64 = ros;
        let chord = 1.0;
        let step = 0.5;
        for _ in 0..node_num {
            let r = step * theta;
            theta += chord / r;
            pos.push([theta.cos() * r, theta.sin() * r]);
        }
    } else {
        let mut angle: f64 = 0.0;
        let mut dist = 0.0;
        let step = 1.0;
        for _ in 0..node_num {
            pos.push([dist * angle.cos(), dist * angle.sin()]);
            dist += step;
            angle += ros;
        }
    }

    rescale(&mut pos, scale);
    recenter(&mut pos, center);

    Pos2DMapping {
        pos_map: graph.node_indices().map(|n| n.index()).zip(pos).collect(),
    }
}

fn recenter(pos: &mut Vec<Point>, center: Option<Point>) {
    let num_pos = pos.len();
    if num_pos == 0 {
        return;
    }
    let dim = pos[0].len();
    if let Some(center) = center {
        for p in pos.iter_mut() {
            for (d, c) in center.iter().enumerate() {
                p[d] += c;
            }
        }
    }
}

fn rescale(pos: &mut Vec<Point>, scale: Option<f64>) {
    let num_pos = pos.len();
    if num_pos == 0 {
        return;
    }
    let dim = pos[0].len();
    let sc = scale.unwrap_or(1.0);

    let mut lim = 0.0;
    for d in 0..dim {
        let mean = pos.iter().map(|t| t[d]).sum::<f64>() / num_pos as f64;
        for p in pos.iter_mut() {
            p[d] -= mean;
            let pd = p[d].abs();
            if lim < pd {
                lim = pd;
            }
        }
    }
    if lim > 0.0 {
        let mult = sc / lim;
        pos.iter_mut().for_each(|t| {
            for td in t.iter_mut() {
                *td *= mult;
            }
        });
    }
}
