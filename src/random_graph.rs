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

#![allow(clippy::float_cmp)]

use crate::{digraph, graph, StablePyGraph};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;

use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand_pcg::Pcg64;

/// Return a :math:`G_{np}` directed random graph, also known as an
/// Erdős-Rényi graph or a binomial graph.
///
/// For number of nodes :math:`n` and probability :math:`p`, the :math:`G_{n,p}`
/// graph algorithm creates :math:`n` nodes, and for all the :math:`n (n - 1)` possible edges,
/// each edge is created independently with probability :math:`p`.
/// In general, for any probability :math:`p`, the expected number of edges returned
/// is :math:`m = p n (n - 1)`. If :math:`p = 0` or :math:`p = 1`, the returned
/// graph is not random and will always be an empty or a complete graph respectively.
/// An empty graph has zero edges and a complete directed graph has :math:`n (n - 1)` edges.
/// The run time is :math:`O(n + m)` where :math:`m` is the expected number of edges mentioned above.
/// When :math:`p = 0`, run time always reduces to :math:`O(n)`, as the lower bound.
/// When :math:`p = 1`, run time always goes to :math:`O(n + n (n - 1))`, as the upper bound.
/// For other probabilities, this algorithm [1]_ runs in :math:`O(n + m)` time.
///
/// For :math:`0 < p < 1`, the algorithm is based on the implementation of the networkx function
/// ``fast_gnp_random_graph`` [2]_
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param float probability: The probability of creating an edge between two nodes
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyDiGraph object
/// :rtype: PyDiGraph
///
/// .. [1] Vladimir Batagelj and Ulrik Brandes,
///    "Efficient generation of large random networks",
///    Phys. Rev. E, 71, 036113, 2005.
/// .. [2] https://github.com/networkx/networkx/blob/networkx-2.4/networkx/generators/random_graphs.py#L49-L120
#[pyfunction]
#[pyo3(text_signature = "(num_nodes, probability, seed=None, /)")]
pub fn directed_gnp_random_graph(
    py: Python,
    num_nodes: isize,
    probability: f64,
    seed: Option<u64>,
) -> PyResult<digraph::PyDiGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StablePyGraph::<Directed>::new();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(PyValueError::new_err(
            "Probability out of range, must be 0 <= p <= 1",
        ));
    }
    if probability > 0.0 {
        if (probability - 1.0).abs() < std::f64::EPSILON {
            for u in 0..num_nodes {
                for v in 0..num_nodes {
                    if u != v {
                        // exclude self-loops
                        let u_index = NodeIndex::new(u as usize);
                        let v_index = NodeIndex::new(v as usize);
                        inner_graph.add_edge(u_index, v_index, py.None());
                    }
                }
            }
        } else {
            let mut v: isize = 0;
            let mut w: isize = -1;
            let lp: f64 = (1.0 - probability).ln();

            let between = Uniform::new(0.0, 1.0);
            while v < num_nodes {
                let random: f64 = between.sample(&mut rng);
                let lr: f64 = (1.0 - random).ln();
                let ratio: isize = (lr / lp) as isize;
                w = w + 1 + ratio;
                // avoid self loops
                if v == w {
                    w += 1;
                }
                while v < num_nodes && num_nodes <= w {
                    w -= v;
                    v += 1;
                    // avoid self loops
                    if v == w {
                        w -= v;
                        v += 1;
                    }
                }
                if v < num_nodes {
                    let v_index = NodeIndex::new(v as usize);
                    let w_index = NodeIndex::new(w as usize);
                    inner_graph.add_edge(v_index, w_index, py.None());
                }
            }
        }
    }

    let graph = digraph::PyDiGraph {
        graph: inner_graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    };
    Ok(graph)
}

/// Return a :math:`G_{np}` random undirected graph, also known as an
/// Erdős-Rényi graph or a binomial graph.
///
/// For number of nodes :math:`n` and probability :math:`p`, the :math:`G_{n,p}`
/// graph algorithm creates :math:`n` nodes, and for all the :math:`n (n - 1)/2` possible edges,
/// each edge is created independently with probability :math:`p`.
/// In general, for any probability :math:`p`, the expected number of edges returned
/// is :math:`m = p n (n - 1)/2`. If :math:`p = 0` or :math:`p = 1`, the returned
/// graph is not random and will always be an empty or a complete graph respectively.
/// An empty graph has zero edges and a complete undirected graph has :math:`n (n - 1)/2` edges.
/// The run time is :math:`O(n + m)` where :math:`m` is the expected number of edges mentioned above.
/// When :math:`p = 0`, run time always reduces to :math:`O(n)`, as the lower bound.
/// When :math:`p = 1`, run time always goes to :math:`O(n + n (n - 1)/2)`, as the upper bound.
/// For other probabilities, this algorithm [1]_ runs in :math:`O(n + m)` time.
///
/// For :math:`0 < p < 1`, the algorithm is based on the implementation of the networkx function
/// ``fast_gnp_random_graph`` [2]_
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param float probability: The probability of creating an edge between two nodes
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyGraph object
/// :rtype: PyGraph
///
/// .. [1] Vladimir Batagelj and Ulrik Brandes,
///    "Efficient generation of large random networks",
///    Phys. Rev. E, 71, 036113, 2005.
/// .. [2] https://github.com/networkx/networkx/blob/networkx-2.4/networkx/generators/random_graphs.py#L49-L120
#[pyfunction]
#[pyo3(text_signature = "(num_nodes, probability, seed=None, /)")]
pub fn undirected_gnp_random_graph(
    py: Python,
    num_nodes: isize,
    probability: f64,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StablePyGraph::<Undirected>::default();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(PyValueError::new_err(
            "Probability out of range, must be 0 <= p <= 1",
        ));
    }
    if probability > 0.0 {
        if (probability - 1.0).abs() < std::f64::EPSILON {
            for u in 0..num_nodes {
                for v in u + 1..num_nodes {
                    let u_index = NodeIndex::new(u as usize);
                    let v_index = NodeIndex::new(v as usize);
                    inner_graph.add_edge(u_index, v_index, py.None());
                }
            }
        } else {
            let mut v: isize = 1;
            let mut w: isize = -1;
            let lp: f64 = (1.0 - probability).ln();

            let between = Uniform::new(0.0, 1.0);
            while v < num_nodes {
                let random: f64 = between.sample(&mut rng);
                let lr = (1.0 - random).ln();
                let ratio: isize = (lr / lp) as isize;
                w = w + 1 + ratio;
                while w >= v && v < num_nodes {
                    w -= v;
                    v += 1;
                }
                if v < num_nodes {
                    let v_index = NodeIndex::new(v as usize);
                    let w_index = NodeIndex::new(w as usize);
                    inner_graph.add_edge(v_index, w_index, py.None());
                }
            }
        }
    }

    let graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    };
    Ok(graph)
}

/// Return a :math:`G_{nm}` directed graph, also known as an
/// Erdős-Rényi graph.
///
/// Generates a random directed graph out of all the possible graphs with :math:`n` nodes and
/// :math:`m` edges. The generated graph will not be a multigraph and will not have self loops.
///
/// For :math:`n` nodes, the maximum edges that can be returned is :math:`n (n - 1)`.
/// Passing :math:`m` higher than that will still return the maximum number of edges.
/// If :math:`m = 0`, the returned graph will always be empty (no edges).
/// When a seed is provided, the results are reproducible. Passing a seed when :math:`m = 0`
/// or :math:`m >= n (n - 1)` has no effect, as the result will always be an empty or a complete graph respectively.
///
/// This algorithm has a time complexity of :math:`O(n + m)`
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param int num_edges: The number of edges to create in the graph
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyDiGraph object
/// :rtype: PyDiGraph
///
#[pyfunction]
#[pyo3(text_signature = "(num_nodes, num_edges, /, seed=None)")]
pub fn directed_gnm_random_graph(
    py: Python,
    num_nodes: isize,
    num_edges: isize,
    seed: Option<u64>,
) -> PyResult<digraph::PyDiGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    if num_edges < 0 {
        return Err(PyValueError::new_err("num_edges must be >= 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StablePyGraph::<Directed>::new();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    // if number of edges to be created is >= max,
    // avoid randomly missed trials and directly add edges between every node
    if num_edges >= num_nodes * (num_nodes - 1) {
        for u in 0..num_nodes {
            for v in 0..num_nodes {
                // avoid self-loops
                if u != v {
                    let u_index = NodeIndex::new(u as usize);
                    let v_index = NodeIndex::new(v as usize);
                    inner_graph.add_edge(u_index, v_index, py.None());
                }
            }
        }
    } else {
        let mut created_edges: isize = 0;
        let between = Uniform::new(0, num_nodes);
        while created_edges < num_edges {
            let u = between.sample(&mut rng);
            let v = between.sample(&mut rng);
            let u_index = NodeIndex::new(u as usize);
            let v_index = NodeIndex::new(v as usize);
            // avoid self-loops and multi-graphs
            if u != v && inner_graph.find_edge(u_index, v_index).is_none() {
                inner_graph.add_edge(u_index, v_index, py.None());
                created_edges += 1;
            }
        }
    }
    let graph = digraph::PyDiGraph {
        graph: inner_graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    };
    Ok(graph)
}

/// Return a :math:`G_{nm}` undirected graph, also known as an
/// Erdős-Rényi graph.
///
/// Generates a random undirected graph out of all the possible graphs with :math:`n` nodes and
/// :math:`m` edges. The generated graph will not be a multigraph and will not have self loops.
///
/// For :math:`n` nodes, the maximum edges that can be returned is :math:`n (n - 1)/2`.
/// Passing :math:`m` higher than that will still return the maximum number of edges.
/// If :math:`m = 0`, the returned graph will always be empty (no edges).
/// When a seed is provided, the results are reproducible. Passing a seed when :math:`m = 0`
/// or :math:`m >= n (n - 1)/2` has no effect, as the result will always be an empty or a complete graph respectively.
///
/// This algorithm has a time complexity of :math:`O(n + m)`
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param int num_edges: The number of edges to create in the graph
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyGraph object
/// :rtype: PyGraph

#[pyfunction]
#[pyo3(text_signature = "(num_nodes, num_edges, /, seed=None)")]
pub fn undirected_gnm_random_graph(
    py: Python,
    num_nodes: isize,
    num_edges: isize,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    if num_nodes <= 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }
    if num_edges < 0 {
        return Err(PyValueError::new_err("num_edges must be >= 0"));
    }
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };
    let mut inner_graph = StablePyGraph::<Undirected>::default();
    for x in 0..num_nodes {
        inner_graph.add_node(x.to_object(py));
    }
    // if number of edges to be created is >= max,
    // avoid randomly missed trials and directly add edges between every node
    if num_edges >= num_nodes * (num_nodes - 1) / 2 {
        for u in 0..num_nodes {
            for v in u + 1..num_nodes {
                let u_index = NodeIndex::new(u as usize);
                let v_index = NodeIndex::new(v as usize);
                inner_graph.add_edge(u_index, v_index, py.None());
            }
        }
    } else {
        let mut created_edges: isize = 0;
        let between = Uniform::new(0, num_nodes);
        while created_edges < num_edges {
            let u = between.sample(&mut rng);
            let v = between.sample(&mut rng);
            let u_index = NodeIndex::new(u as usize);
            let v_index = NodeIndex::new(v as usize);
            // avoid self-loops and multi-graphs
            if u != v && inner_graph.find_edge(u_index, v_index).is_none() {
                inner_graph.add_edge(u_index, v_index, py.None());
                created_edges += 1;
            }
        }
    }
    let graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    };
    Ok(graph)
}

#[inline]
fn pnorm(x: f64, p: f64) -> f64 {
    if p == 1.0 || p == std::f64::INFINITY {
        x.abs()
    } else if p == 2.0 {
        x * x
    } else {
        x.abs().powf(p)
    }
}

fn distance(x: &[f64], y: &[f64], p: f64) -> f64 {
    let it = x.iter().zip(y.iter()).map(|(xi, yi)| pnorm(xi - yi, p));

    if p == std::f64::INFINITY {
        it.fold(-1.0, |max, x| if x > max { x } else { max })
    } else {
        it.sum()
    }
}

/// Returns a random geometric graph in the unit cube of dimensions `dim`.
///
/// The random geometric graph model places `num_nodes` nodes uniformly at
/// random in the unit cube. Two nodes are joined by an edge if the
/// distance between the nodes is at most `radius`.
///
/// Each node has a node attribute ``'pos'`` that stores the
/// position of that node in Euclidean space as provided by the
/// ``pos`` keyword argument or, if ``pos`` was not provided, as
/// generated by this function.
///
/// :param int num_nodes: The number of nodes to create in the graph
/// :param float radius: Distance threshold value
/// :param int dim: Dimension of node positions. Default: 2
/// :param list pos: Optional list with node positions as values
/// :param float p: Which Minkowski distance metric to use.  `p` has to meet the condition
///     ``1 <= p <= infinity``.
///     If this argument is not specified, the :math:`L^2` metric
///     (the Euclidean distance metric), p = 2 is used.
/// :param int seed: An optional seed to use for the random number generator
///
/// :return: A PyGraph object
/// :rtype: PyGraph
#[pyfunction(dim = "2", p = "2.0")]
#[pyo3(text_signature = "(num_nodes, radius, /, dim=2, pos=None, p=2.0, seed=None)")]
pub fn random_geometric_graph(
    py: Python,
    num_nodes: usize,
    radius: f64,
    dim: usize,
    pos: Option<Vec<Vec<f64>>>,
    p: f64,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    if num_nodes == 0 {
        return Err(PyValueError::new_err("num_nodes must be > 0"));
    }

    let mut inner_graph = StablePyGraph::<Undirected>::default();

    let radius_p = pnorm(radius, p);
    let mut rng: Pcg64 = match seed {
        Some(seed) => Pcg64::seed_from_u64(seed),
        None => Pcg64::from_entropy(),
    };

    let dist = Uniform::new(0.0, 1.0);
    let pos = pos.unwrap_or_else(|| {
        (0..num_nodes)
            .map(|_| (0..dim).map(|_| dist.sample(&mut rng)).collect())
            .collect()
    });

    if num_nodes != pos.len() {
        return Err(PyValueError::new_err(
            "number of elements in pos and num_nodes must be equal",
        ));
    }

    for pval in pos.iter() {
        let pos_dict = PyDict::new(py);
        pos_dict.set_item("pos", pval.to_object(py))?;

        inner_graph.add_node(pos_dict.into());
    }

    for u in 0..(num_nodes - 1) {
        for v in (u + 1)..num_nodes {
            if distance(&pos[u], &pos[v], p) < radius_p {
                inner_graph.add_edge(NodeIndex::new(u), NodeIndex::new(v), py.None());
            }
        }
    }

    let graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    };
    Ok(graph)
}
