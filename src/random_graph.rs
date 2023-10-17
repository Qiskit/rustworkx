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

use rustworkx_core::generators as core_generators;

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
#[pyo3(text_signature = "(num_nodes, probability, /, seed=None)")]
pub fn directed_gnp_random_graph(
    py: Python,
    num_nodes: usize,
    probability: f64,
    seed: Option<u64>,
) -> PyResult<digraph::PyDiGraph> {
    let default_fn = || py.None();
    let mut graph: StablePyGraph<Directed> = match core_generators::gnp_random_graph(
        num_nodes,
        probability,
        seed,
        default_fn,
        default_fn,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyValueError::new_err(
                "num_nodes or probability invalid input",
            ))
        }
    };
    // Core function does not put index into node payload, so for backwards compat
    // in the python interface, we do it here.
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    for node in nodes.iter() {
        graph[*node] = node.index().to_object(py);
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph: false,
        attrs: py.None(),
    })
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
#[pyo3(text_signature = "(num_nodes, probability, /, seed=None)")]
pub fn undirected_gnp_random_graph(
    py: Python,
    num_nodes: usize,
    probability: f64,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let mut graph: StablePyGraph<Undirected> = match core_generators::gnp_random_graph(
        num_nodes,
        probability,
        seed,
        default_fn,
        default_fn,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyValueError::new_err(
                "num_nodes or probability invalid input",
            ))
        }
    };
    // Core function does not put index into node payload, so for backwards compat
    // in the python interface, we do it here.
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    for node in nodes.iter() {
        graph[*node] = node.index().to_object(py);
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    })
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
    num_nodes: usize,
    num_edges: usize,
    seed: Option<u64>,
) -> PyResult<digraph::PyDiGraph> {
    let default_fn = || py.None();
    let mut graph: StablePyGraph<Directed> =
        match core_generators::gnm_random_graph(num_nodes, num_edges, seed, default_fn, default_fn)
        {
            Ok(graph) => graph,
            Err(_) => {
                return Err(PyValueError::new_err(
                    "num_nodes or num_edges invalid input",
                ))
            }
        };
    // Core function does not put index into node payload, so for backwards compat
    // in the python interface, we do it here.
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    for node in nodes.iter() {
        graph[*node] = node.index().to_object(py);
    }
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph: false,
        attrs: py.None(),
    })
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
    num_nodes: usize,
    num_edges: usize,
    seed: Option<u64>,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    let mut graph: StablePyGraph<Undirected> =
        match core_generators::gnm_random_graph(num_nodes, num_edges, seed, default_fn, default_fn)
        {
            Ok(graph) => graph,
            Err(_) => {
                return Err(PyValueError::new_err(
                    "num_nodes or num_edges invalid input",
                ))
            }
        };
    // Core function does not put index into node payload, so for backwards compat
    // in the python interface, we do it here.
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    for node in nodes.iter() {
        graph[*node] = node.index().to_object(py);
    }
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    })
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
#[pyfunction]
#[pyo3(
    signature=(num_nodes, radius, dim=2, pos=None, p=2.0, seed=None),
    text_signature = "(num_nodes, radius, /, dim=2, pos=None, p=2.0, seed=None)"
)]
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

/// Generate a random graph using Barabási–Albert preferential attachment
///
/// A graph is grown to $n$ nodes by adding new nodes each with $m$ edges that
/// are preferentially attached to existing nodes with high degree. All the edges
/// and nodes added to this graph will have weights of ``None``.
///
/// The algorithm performed by this function are described in:
///
/// A. L. Barabási and R. Albert "Emergence of scaling in random networks",
/// Science 286, pp 509-512, 1999.
///
/// :param int n: The number of nodes to extend the graph to.
/// :param int m: The number of edges to attach from a new node to existing nodes.
/// :param int seed: An optional seed to use for the random number generator
/// :param PyGraph initial_graph: An optional initial graph to use as a starting
///     point. :func:`.star_graph` is used to create an ``m`` node star graph
///     to use as a starting point. If specified the input graph will be
///     modified in place.
///
/// :return: A PyGraph object
/// :rtype: PyGraph
#[pyfunction]
pub fn barabasi_albert_graph(
    py: Python,
    n: usize,
    m: usize,
    seed: Option<u64>,
    initial_graph: Option<graph::PyGraph>,
) -> PyResult<graph::PyGraph> {
    let default_fn = || py.None();
    if m < 1 {
        return Err(PyValueError::new_err("m must be > 0"));
    }
    if m >= n {
        return Err(PyValueError::new_err("m must be < n"));
    }
    let graph = match core_generators::barabasi_albert_graph(
        n,
        m,
        seed,
        initial_graph.map(|x| x.graph),
        default_fn,
        default_fn,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyValueError::new_err(
                "initial_graph has either less nodes than m, or more nodes than n",
            ))
        }
    };
    Ok(graph::PyGraph {
        graph,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    })
}

/// Generate a random graph using Barabási–Albert preferential attachment
///
/// A graph is grown to $n$ nodes by adding new nodes each with $m$ edges that
/// are preferentially attached to existing nodes with high degree. All the edges
/// and nodes added to this graph will have weights of ``None``. For the purposes
/// of the extension algorithm all edges are treated as weak (meaning directionality
/// isn't considered).
///
/// The algorithm performed by this function are described in:
///
/// A. L. Barabási and R. Albert "Emergence of scaling in random networks",
/// Science 286, pp 509-512, 1999.
///
/// :param int n: The number of nodes to extend the graph to.
/// :param int m: The number of edges to attach from a new node to existing nodes.
/// :param int seed: An optional seed to use for the random number generator
/// :param PyDiGraph initial_graph: An optional initial graph to use as a starting
///     point. :func:`.star_graph` is used to create an ``m`` node star graph
///     to use as a starting point. If specified the input graph will be
///     modified in place.
///
/// :return: A PyDiGraph object
/// :rtype: PyDiGraph
#[pyfunction]
pub fn directed_barabasi_albert_graph(
    py: Python,
    n: usize,
    m: usize,
    seed: Option<u64>,
    initial_graph: Option<digraph::PyDiGraph>,
) -> PyResult<digraph::PyDiGraph> {
    let default_fn = || py.None();
    if m < 1 {
        return Err(PyValueError::new_err("m must be > 0"));
    }
    if m >= n {
        return Err(PyValueError::new_err("m must be < n"));
    }
    let graph = match core_generators::barabasi_albert_graph(
        n,
        m,
        seed,
        initial_graph.map(|x| x.graph),
        default_fn,
        default_fn,
    ) {
        Ok(graph) => graph,
        Err(_) => {
            return Err(PyValueError::new_err(
                "initial_graph has either less nodes than m, or more nodes than n",
            ))
        }
    };
    Ok(digraph::PyDiGraph {
        graph,
        node_removed: false,
        check_cycle: false,
        cycle_state: algo::DfsSpace::default(),
        multigraph: false,
        attrs: py.None(),
    })
}
