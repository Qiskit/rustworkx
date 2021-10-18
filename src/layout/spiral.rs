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

use petgraph::EdgeType;

use super::spring::{recenter, rescale, Point};
use crate::iterators::Pos2DMapping;
use crate::StablePyGraph;

pub fn spiral_layout<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
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
