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

use petgraph::visit::NodeIndexable;
use petgraph::EdgeType;

use super::spring::{recenter, Point};
use crate::iterators::Pos2DMapping;
use crate::StablePyGraph;

pub fn shell_layout<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
    nlist: Option<Vec<Vec<usize>>>,
    rotate: Option<f64>,
    scale: Option<f64>,
    center: Option<Point>,
) -> Pos2DMapping {
    let node_num = graph.node_bound();
    let mut pos: Vec<Point> = vec![[0.0, 0.0]; node_num];
    let pi = std::f64::consts::PI;

    let shell_list: Vec<Vec<usize>> = match nlist {
        Some(nlist) => nlist,
        None => vec![graph.node_indices().map(|n| n.index()).collect()],
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

    if let Some(center) = center {
        recenter(&mut pos, center);
    }

    Pos2DMapping {
        pos_map: graph
            .node_indices()
            .map(|n| {
                let n = n.index();
                (n, pos[n])
            })
            .collect(),
    }
}
