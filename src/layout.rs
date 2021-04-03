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

use std::iter::Iterator;

use pyo3::prelude::*;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::EdgeType;

type NT = f64;
pub type Point = (NT, NT);
type Graph<Ty> = StableGraph<PyObject, PyObject, Ty>;

fn l2norm(x: Point) -> NT {
    (x.0 * x.0 + x.1 * x.1).sqrt()
}

pub trait Force {
    // evaluate force between points x, y
    // given the difference x - y and the l2 - norm ||x - y||.
    fn eval(&self, dif: &Point, dnorm: NT) -> NT;

    // total force in Point x.
    fn total<'a, I>(&self, x: &Point, ys: I) -> (NT, NT)
    where
        I: Iterator<Item = &'a Point>,
    {
        let mut ftot = (0.0, 0.0);

        for y in ys {
            let d = (y.0 - x.0, y.1 - x.1);
            let dnorm = l2norm(d);
            let f = self.eval(&d, dnorm);

            ftot.0 += f * d.0 / dnorm;
            ftot.1 += f * d.1 / dnorm;
        }

        ftot
    }
}

pub struct RepulsiveForce {
    _c: NT,
    _k: NT,
    _p: i32,
}

impl RepulsiveForce {
    pub fn new(k: NT, p: i32) -> Self {
        RepulsiveForce {
            _c: 0.2,
            _k: k,
            _p: p,
        }
    }
}

impl Force for RepulsiveForce {
    fn eval(&self, _: &Point, dnorm: NT) -> NT {
        -self._c * self._k.powi(1_i32 + self._p) / dnorm.powi(self._p)
    }
}

pub struct AttractiveForce {
    _k: NT,
}

impl AttractiveForce {
    pub fn new(k: NT) -> Self {
        AttractiveForce { _k: k }
    }
}

impl Force for AttractiveForce {
    fn eval(&self, dif: &Point, _: NT) -> NT {
        (dif.0 * dif.0 + dif.1 * dif.1) / self._k
    }
}

pub trait CoolingScheme {
    fn update_step(&mut self, cost: NT) -> NT;
}

pub struct AdaptiveCoolingScheme {
    _step: NT,
    _tau: NT,
    _cost: NT,
    _progress: usize,
}

impl AdaptiveCoolingScheme {
    pub fn new(step: NT) -> Self {
        AdaptiveCoolingScheme {
            _step: step,
            _tau: 0.9,
            _cost: f64::INFINITY,
            _progress: 0,
        }
    }
}

impl CoolingScheme for AdaptiveCoolingScheme {
    fn update_step(&mut self, cost: NT) -> NT {
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
    _step: NT,
    _niter: usize,
    _dt: NT,
}

impl LinearCoolingScheme {
    pub fn new(step: NT, niter: usize) -> Self {
        LinearCoolingScheme {
            _step: step,
            _niter: niter,
            _dt: step / (niter + 1) as NT,
        }
    }
}

impl CoolingScheme for LinearCoolingScheme {
    fn update_step(&mut self, _: NT) -> NT {
        self._step -= self._dt;
        self._step
    }
}

pub fn evolve<Ty, Fa, Fr, C>(
    graph: &Graph<Ty>,
    mut pos: Vec<Point>,
    f_a: Fa,
    f_r: Fr,
    mut cs: C,
    niter: usize,
    tol: f64,
) -> Vec<Point>
where
    Ty: EdgeType,
    Fa: Force,
    Fr: Force,
    C: CoolingScheme,
{
    let nnodes = graph.node_count();

    let mut step = cs.update_step(f64::INFINITY);

    for _ in 0..niter {
        let mut energy = 0.0;
        let mut converged = true;

        for v in 0..nnodes {
            // attractive forces
            let ys = graph
                .neighbors_undirected(NodeIndex::new(v))
                .map(|n| &pos[n.index()]);
            let fa = f_a.total(&pos[v], ys);

            // repulsive forces
            let ys = (0..nnodes).filter(|&n| n != v).map(|n| &pos[n]);
            let fr = f_r.total(&pos[v], ys);

            // update current position
            let f = (fa.0 + fr.0, fa.1 + fr.1);
            let fsq = f.0 * f.0 + f.1 * f.1;
            energy += fsq;

            let fnorm = fsq.sqrt();
            let dx = step * f.0 / fnorm;
            let dy = step * f.1 / fnorm;
            pos[v].0 += dx;
            pos[v].1 += dy;

            if dx * dx + dy * dy > tol {
                converged = false;
            }
        }

        step = cs.update_step(energy);
        if converged {
            break;
        }
    }

    pos
}
