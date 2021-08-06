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

use pyo3::prelude::*;

use petgraph::prelude::*;
use petgraph::EdgeType;

use super::spring::{recenter, rescale, Point};
use crate::iterators::Pos2DMapping;

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
