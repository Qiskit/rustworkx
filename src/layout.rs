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

use pyo3::prelude::*;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::EdgeType;

type Nt = f64;
pub type Point = [Nt; 2];
type Graph<Ty> = StableGraph<PyObject, PyObject, Ty>;

const LBOUND: Nt = 1e-8;

#[inline]
fn l2norm(x: Point) -> Nt {
    (x[0] * x[0] + x[1] * x[1]).sqrt()
}

#[inline]
fn max(x: Nt, y: Nt) -> Nt {
    if x > y {
        x
    } else {
        y
    }
}

pub trait Force {
    // evaluate force between points x, y
    // given the difference x - y and the l2 - norm ||x - y||.
    fn eval(&self, dif: &Point, dnorm: Nt) -> Nt;

    // total force in Point x.
    fn total<'a, I>(&self, x: &Point, ys: I) -> (Nt, Nt)
    where
        I: Iterator<Item = &'a Point>,
    {
        let mut ftot = (0.0, 0.0);

        for y in ys {
            let d = [y[0] - x[0], y[1] - x[1]];
            let dnorm = max(l2norm(d), LBOUND);
            let f = self.eval(&d, dnorm);

            ftot.0 += f * d[0] / dnorm;
            ftot.1 += f * d[1] / dnorm;
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
    _niter: usize,
    _dt: Nt,
}

impl LinearCoolingScheme {
    pub fn new(step: Nt, niter: usize) -> Self {
        LinearCoolingScheme {
            _step: step,
            _niter: niter,
            _dt: step / (niter + 1) as Nt,
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
    let mut mu: Point = indices
        .iter()
        .map(|&n| pos[n])
        .reduce(|[sumx, sumy], [px, py]| [sumx + px, sumy + py])
        .unwrap();
    mu[0] /= n as Nt;
    mu[1] /= n as Nt;

    // substract mean and find max coordinate for all axes
    let mut lim = std::f64::NEG_INFINITY;
    for n in indices {
        let [px, py] = pos.get_mut(n).unwrap();
        *px -= mu[0];
        *py -= mu[1];

        let pm = max(px.abs(), py.abs());
        if lim < pm {
            lim = pm;
        }
    }

    // rescale
    for [px, py] in pos.iter_mut() {
        *px *= scale / lim;
        *py *= scale / lim;
    }
}

fn recenter(pos: &mut Vec<Point>, center: Point) {
    for [px, py] in pos.iter_mut() {
        *px += center[0];
        *py += center[1];
    }
}

pub fn evolve<Ty, Fa, Fr, C>(
    graph: &Graph<Ty>,
    mut pos: Vec<Point>,
    fixed: HashSet<usize>,
    f_a: Fa,
    f_r: Fr,
    mut cs: C,
    niter: usize,
    tol: f64,
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

    for _ in 0..niter {
        let mut energy = 0.0;
        let mut converged = true;

        for v in graph.node_indices() {
            let v = v.index();
            if fixed.contains(&v) {
                continue;
            }
            // attractive forces
            let ys = graph
                .neighbors_undirected(NodeIndex::new(v))
                .map(|n| &pos[n.index()]);
            let fa = f_a.total(&pos[v], ys);

            // repulsive forces
            let ys = graph
                .node_indices()
                .filter(|&n| n.index() != v)
                .map(|n| &pos[n.index()]);
            let fr = f_r.total(&pos[v], ys);

            // update current position
            let f = (fa.0 + fr.0, fa.1 + fr.1);
            let f2 = f.0 * f.0 + f.1 * f.1;
            energy += f2;

            let fnorm = max(f2.sqrt(), LBOUND);
            let dx = step * f.0 / fnorm;
            let dy = step * f.1 / fnorm;
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
