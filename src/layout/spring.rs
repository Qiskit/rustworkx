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

use crate::iterators::Pos2DMapping;
use crate::weight_callable;

use std::iter::Iterator;

use hashbrown::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{IntoEdgeReferences, NodeIndexable};
use petgraph::EdgeType;

use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand_pcg::Pcg64;

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
pub fn rescale(pos: &mut Vec<Point>, scale: Nt, indices: Vec<usize>) {
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

pub fn recenter(pos: &mut Vec<Point>, center: Point) {
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

#[allow(clippy::too_many_arguments)]
pub fn spring_layout<Ty>(
    py: Python,
    graph: &StableGraph<PyObject, PyObject, Ty>,
    pos: Option<HashMap<usize, Point>>,
    fixed: Option<HashSet<usize>>,
    k: Option<f64>,
    repulsive_exponent: Option<i32>,
    adaptive_cooling: Option<bool>,
    num_iter: Option<usize>,
    tol: Option<f64>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    scale: Option<f64>,
    center: Option<Point>,
    seed: Option<u64>,
) -> PyResult<Pos2DMapping>
where
    Ty: EdgeType,
{
    if fixed.is_some() && pos.is_none() {
        return Err(PyValueError::new_err("`fixed` specified but `pos` not."));
    }

    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };

    let dist = Uniform::new(0.0, 1.0);

    let pos = pos.unwrap_or_default();
    let mut vpos: Vec<Point> = (0..graph.node_bound())
        .map(|_| [dist.sample(&mut rng), dist.sample(&mut rng)])
        .collect();
    for (n, p) in pos.into_iter() {
        vpos[n] = p;
    }

    let fixed = fixed.unwrap_or_default();
    let k = k.unwrap_or(1.0 / (graph.node_count() as f64).sqrt());
    let f_a = AttractiveForce::new(k);
    let f_r = RepulsiveForce::new(k, repulsive_exponent.unwrap_or(2));

    let num_iter = num_iter.unwrap_or(50);
    let tol = tol.unwrap_or(1e-6);
    let step = 0.1;

    let mut weights: HashMap<(usize, usize), f64> =
        HashMap::with_capacity(2 * graph.edge_count());
    for e in graph.edge_references() {
        let w = weight_callable(py, &weight_fn, e.weight(), default_weight)?;
        let source = e.source().index();
        let target = e.target().index();

        weights.insert((source, target), w);
        weights.insert((target, source), w);
    }

    let pos = match adaptive_cooling {
        Some(false) => {
            let cs = LinearCoolingScheme::new(step, num_iter);
            evolve(
                graph, vpos, fixed, f_a, f_r, cs, num_iter, tol, weights,
                scale, center,
            )
        }
        _ => {
            let cs = AdaptiveCoolingScheme::new(step);
            evolve(
                graph, vpos, fixed, f_a, f_r, cs, num_iter, tol, weights,
                scale, center,
            )
        }
    };

    Ok(Pos2DMapping {
        pos_map: graph
            .node_indices()
            .map(|n| {
                let n = n.index();
                (n, pos[n])
            })
            .collect(),
    })
}
