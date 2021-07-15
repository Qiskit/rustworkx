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
use indexmap::IndexMap;

use pyo3::prelude::*;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::NodeIndexable;
use petgraph::EdgeType;

use crate::iterators::Pos2DMapping;

type Nt = f64;
pub type Point = [Nt; 2];
type Graph<Ty> = StableGraph<PyObject, PyObject, Ty>;

const LBOUND: Nt = 1e-8;

#[inline]
fn l2norm(x: Point) -> Nt {
    (x[0] * x[0] + x[1] * x[1]).sqrt()
}

pub trait Force {
    // evaluate force between points x, y
    // given the difference x - y and the l2 - norm ||x - y||.
    fn eval(&self, dif: &Point, dnorm: Nt) -> Nt;

    // total force in Point x
    // from (points, weights) in ys.
    fn total<'a, I>(&self, x: &Point, ys: I) -> [Nt; 2]
    where
        I: Iterator<Item = (&'a Point, Nt)>,
    {
        let mut ftot = [0.0, 0.0];

        for (y, w) in ys {
            let d = [y[0] - x[0], y[1] - x[1]];
            let dnorm = l2norm(d).max(LBOUND);
            let f = w * self.eval(&d, dnorm);

            ftot[0] += f * d[0] / dnorm;
            ftot[1] += f * d[1] / dnorm;
        }

        ftot
    }
}

pub struct RepulsiveForce {
    _c: Nt,
    _k: Nt,
    _p: i32,
}

impl RepulsiveForce {
    pub fn new(k: Nt, p: i32) -> Self {
        RepulsiveForce {
            _c: 0.2,
            _k: k,
            _p: p,
        }
    }
}

impl Force for RepulsiveForce {
    fn eval(&self, _: &Point, dnorm: Nt) -> Nt {
        -self._c * self._k.powi(1_i32 + self._p) / dnorm.powi(self._p)
    }
}

pub struct AttractiveForce {
    _k: Nt,
}

impl AttractiveForce {
    pub fn new(k: Nt) -> Self {
        AttractiveForce { _k: k }
    }
}

impl Force for AttractiveForce {
    fn eval(&self, dif: &Point, _: Nt) -> Nt {
        (dif[0] * dif[0] + dif[1] * dif[1]) / self._k
    }
}

pub trait CoolingScheme {
    fn update_step(&mut self, cost: Nt) -> Nt;
}

pub struct AdaptiveCoolingScheme {
    _step: Nt,
    _tau: Nt,
    _cost: Nt,
    _progress: usize,
}

impl AdaptiveCoolingScheme {
    pub fn new(step: Nt) -> Self {
        AdaptiveCoolingScheme {
            _step: step,
            _tau: 0.9,
            _cost: std::f64::INFINITY,
            _progress: 0,
        }
    }
}

impl CoolingScheme for AdaptiveCoolingScheme {
    fn update_step(&mut self, cost: Nt) -> Nt {
        if cost < self._cost {
            self._progress += 1;
            if self._progress >= 5 {
                self._progress = 0;
                self._step /= self._tau;
            }
        } else {
            self._progress = 0;
            self._step *= self._tau;
        }

        self._cost = cost;
        self._step
    }
}

pub struct LinearCoolingScheme {
    _step: Nt,
    _num_iter: usize,
    _dt: Nt,
}

impl LinearCoolingScheme {
    pub fn new(step: Nt, num_iter: usize) -> Self {
        LinearCoolingScheme {
            _step: step,
            _num_iter: num_iter,
            _dt: step / (num_iter + 1) as Nt,
        }
    }
}

impl CoolingScheme for LinearCoolingScheme {
    fn update_step(&mut self, _: Nt) -> Nt {
        self._step -= self._dt;
        self._step
    }
}

// Rescale so that pos in [-scale, scale].
fn rescale(pos: &mut Vec<Point>, scale: Nt, indices: Vec<usize>) {
    let n = indices.len();
    if n == 0 {
        return;
    }
    // find mean in each dimension
    let mut mu: Point = [0.0, 0.0];
    for &n in &indices {
        mu[0] += pos[n][0];
        mu[1] += pos[n][1];
    }
    mu[0] /= n as Nt;
    mu[1] /= n as Nt;

    // substract mean and find max coordinate for all axes
    let mut lim = std::f64::NEG_INFINITY;
    for n in indices {
        let [px, py] = pos.get_mut(n).unwrap();
        *px -= mu[0];
        *py -= mu[1];

        let pm = px.abs().max(py.abs());
        if lim < pm {
            lim = pm;
        }
    }

    // rescale
    if lim > 0.0 {
        for [px, py] in pos.iter_mut() {
            *px *= scale / lim;
            *py *= scale / lim;
        }
    }
}

fn recenter(pos: &mut Vec<Point>, center: Point) {
    for [px, py] in pos.iter_mut() {
        *px += center[0];
        *py += center[1];
    }
}

#[allow(clippy::too_many_arguments)]
pub fn evolve<Ty, Fa, Fr, C>(
    graph: &Graph<Ty>,
    mut pos: Vec<Point>,
    fixed: HashSet<usize>,
    f_a: Fa,
    f_r: Fr,
    mut cs: C,
    num_iter: usize,
    tol: f64,
    weights: HashMap<(usize, usize), f64>,
    scale: Option<Nt>,
    center: Option<Point>,
) -> Vec<Point>
where
    Ty: EdgeType,
    Fa: Force,
    Fr: Force,
    C: CoolingScheme,
{
    let mut step = cs.update_step(std::f64::INFINITY);

    for _ in 0..num_iter {
        let mut energy = 0.0;
        let mut converged = true;

        for v in graph.node_indices() {
            let v = v.index();
            if fixed.contains(&v) {
                continue;
            }
            // attractive forces
            let ys = graph.neighbors_undirected(NodeIndex::new(v)).map(|n| {
                let n = n.index();
                (&pos[n], weights[&(v, n)])
            });
            let fa = f_a.total(&pos[v], ys);

            // repulsive forces
            let ys =
                graph.node_indices().filter(|&n| n.index() != v).map(|n| {
                    let n = n.index();
                    (&pos[n], 1.0)
                });
            let fr = f_r.total(&pos[v], ys);

            // update current position
            let f = [fa[0] + fr[0], fa[1] + fr[1]];
            let f2 = f[0] * f[0] + f[1] * f[1];
            energy += f2;

            let fnorm = f2.sqrt().max(LBOUND);
            let dx = step * f[0] / fnorm;
            let dy = step * f[1] / fnorm;
            pos[v][0] += dx;
            pos[v][1] += dy;

            if dx * dx + dy * dy > tol {
                converged = false;
            }
        }

        step = cs.update_step(energy);
        if converged {
            break;
        }
    }

    if fixed.is_empty() {
        if let Some(scale) = scale {
            rescale(
                &mut pos,
                scale,
                graph.node_indices().map(|n| n.index()).collect(),
            );
        }

        if let Some(center) = center {
            recenter(&mut pos, center);
        }
    }

    pos
}

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
            pos_map: IndexMap::new(),
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

pub fn shell_layout<Ty: EdgeType>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
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
