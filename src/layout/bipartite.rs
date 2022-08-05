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

use hashbrown::HashSet;

use petgraph::EdgeType;

use super::spring::{recenter, rescale, Point};
use crate::iterators::Pos2DMapping;
use crate::StablePyGraph;
use rustworkx_core::dictmap::*;

pub fn bipartite_layout<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
    first_nodes: HashSet<usize>,
    horizontal: Option<bool>,
    scale: Option<f64>,
    center: Option<Point>,
    aspect_ratio: Option<f64>,
) -> Pos2DMapping {
    let node_num = graph.node_count();
    if node_num == 0 {
        return Pos2DMapping {
            pos_map: DictMap::new(),
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

    let x_offset: f64 = width / 2.0;
    let y_offset: f64 = height / 2.0;
    let left_dy: f64 = match left_num {
        0 | 1 => 0.0,
        _ => height / (left_num - 1) as f64,
    };
    let right_dy: f64 = match right_num {
        0 | 1 => 0.0,
        _ => height / (right_num - 1) as f64,
    };

    let mut lc: f64 = 0.0;
    let mut rc: f64 = 0.0;

    for node in graph.node_indices() {
        let n = node.index();

        let (x, y): (f64, f64);
        if first_nodes.contains(&n) {
            x = -x_offset;
            y = lc * left_dy - y_offset;
            lc += 1.0;
        } else {
            x = width - x_offset;
            y = rc * right_dy - y_offset;
            rc += 1.0;
        }

        if horizontal == Some(true) {
            pos.push([-y, x]);
        } else {
            pos.push([x, y]);
        }
    }

    if let Some(scale) = scale {
        rescale(&mut pos, scale, (0..node_num).collect());
    }

    if let Some(center) = center {
        recenter(&mut pos, center);
    }

    Pos2DMapping {
        pos_map: graph.node_indices().map(|n| n.index()).zip(pos).collect(),
    }
}
