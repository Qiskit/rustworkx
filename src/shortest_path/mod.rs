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

mod all_pairs_bellman_ford;
pub mod all_pairs_dijkstra;
mod average_length;
mod distance_matrix;
mod floyd_warshall;
mod num_shortest_path;

use std::convert::TryFrom;

use crate::{digraph, edge_weights_from_callable, graph, CostFn, NegativeCycle, NoPathFound};

use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::EdgeIndex;
use petgraph::visit::NodeCount;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyValueError;

use numpy::IntoPyArray;

use rustworkx_core::dictmap::*;
use rustworkx_core::shortest_path::{
    astar, bellman_ford, dijkstra, k_shortest_path, negative_cycle_finder,
};

use crate::iterators::{
    AllPairsPathLengthMapping, AllPairsPathMapping, NodeIndices, NodesCountMapping,
    PathLengthMapping, PathMapping,
};

/// Find the shortest path from a node
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param PyGraph graph:
/// :param int source: The node index to find paths from
/// :param int target: An optional target to find a path to
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
/// :param bool as_undirected: If set to true the graph will be treated as
///     undirected for finding the shortest path.
///
/// :return: Dictionary of paths. The keys are destination node indices and
///     the dict values are lists of node indices making the path.
/// :rtype: dict
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction(default_weight = "1.0", as_undirected = "false")]
#[pyo3(text_signature = "(graph, source, /, target=None weight_fn=None, default_weight=1.0)")]
pub fn graph_dijkstra_shortest_paths(
    py: Python,
    graph: &graph::PyGraph,
    source: usize,
    target: Option<usize>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<PathMapping> {
    let start = NodeIndex::new(source);
    let goal_index: Option<NodeIndex> = target.map(NodeIndex::new);
    let mut paths: DictMap<NodeIndex, Vec<NodeIndex>> = DictMap::with_capacity(graph.node_count());

    let cost_fn = CostFn::try_from((weight_fn, default_weight))?;

    (dijkstra(
        &graph.graph,
        start,
        goal_index,
        |e| cost_fn.call(py, e.weight()),
        Some(&mut paths),
    ) as PyResult<Vec<Option<f64>>>)?;

    Ok(PathMapping {
        paths: paths
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == source || target.is_some() && target.unwrap() != k_int {
                    None
                } else {
                    Some((
                        k.index(),
                        v.iter().map(|x| x.index()).collect::<Vec<usize>>(),
                    ))
                }
            })
            .collect(),
    })
}

/// Find the shortest path from a node
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param PyDiGraph graph:
/// :param int source: The node index to find paths from
/// :param int target: An optional target path to find the path
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
/// :param bool as_undirected: If set to true the graph will be treated as
///     undirected for finding the shortest path.
///
/// :return: Dictionary of paths. The keys are destination node indices and
///     the dict values are lists of node indices making the path.
/// :rtype: dict
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction(default_weight = "1.0", as_undirected = "false")]
#[pyo3(
    text_signature = "(graph, source, /, target=None weight_fn=None, default_weight=1.0, as_undirected=False)"
)]
pub fn digraph_dijkstra_shortest_paths(
    py: Python,
    graph: &digraph::PyDiGraph,
    source: usize,
    target: Option<usize>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    as_undirected: bool,
) -> PyResult<PathMapping> {
    let start = NodeIndex::new(source);
    let goal_index: Option<NodeIndex> = target.map(NodeIndex::new);
    let mut paths: DictMap<NodeIndex, Vec<NodeIndex>> = DictMap::with_capacity(graph.node_count());
    let cost_fn = CostFn::try_from((weight_fn, default_weight))?;

    if as_undirected {
        (dijkstra(
            // TODO: Use petgraph undirected adapter after
            // https://github.com/petgraph/petgraph/pull/318 is available in
            // a petgraph release.
            &graph.to_undirected(py, true, None)?.graph,
            start,
            goal_index,
            |e| cost_fn.call(py, e.weight()),
            Some(&mut paths),
        ) as PyResult<Vec<Option<f64>>>)?;
    } else {
        (dijkstra(
            &graph.graph,
            start,
            goal_index,
            |e| cost_fn.call(py, e.weight()),
            Some(&mut paths),
        ) as PyResult<Vec<Option<f64>>>)?;
    }
    Ok(PathMapping {
        paths: paths
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == source || target.is_some() && target.unwrap() != k_int {
                    None
                } else {
                    Some((k_int, v.iter().map(|x| x.index()).collect::<Vec<usize>>()))
                }
            })
            .collect(),
    })
}

/// Compute the lengths of the shortest paths for a PyGraph object using
/// Dijkstra's algorithm
///
/// :param PyGraph graph: The input graph to use
/// :param int node: The node index to use as the source for finding the
///     shortest paths from
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the
///     cost/weight of that edge. It must be non-negative
/// :param int goal: An optional node index to use as the end of the path.
///     When specified the traversal will stop when the goal is reached and
///     the output dictionary will only have a single entry with the length
///     of the shortest path to the goal node.
///
/// :returns: A dictionary of the shortest paths from the provided node where
///     the key is the node index of the end of the path and the value is the
///     cost/sum of the weights of path
/// :rtype: PathLengthMapping
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, node, edge_cost_fn, /, goal=None)")]
pub fn graph_dijkstra_shortest_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let edge_cost_callable = CostFn::from(edge_cost_fn);
    let start = NodeIndex::new(node);
    let goal_index: Option<NodeIndex> = goal.map(NodeIndex::new);

    let res: Vec<Option<f64>> = dijkstra(
        &graph.graph,
        start,
        goal_index,
        |e| edge_cost_callable.call(py, e.weight()),
        None,
    )?;

    if let Some(goal_usize) = goal {
        return Ok(PathLengthMapping {
            path_lengths: match res[goal_usize] {
                Some(goal_length) => {
                    let mut ans = DictMap::new();
                    ans.insert(goal_usize, goal_length);
                    ans
                }
                None => DictMap::new(),
            },
        });
    }

    Ok(PathLengthMapping {
        path_lengths: res
            .into_iter()
            .enumerate()
            .filter_map(|(k_int, opt_v)| {
                if k_int != node {
                    opt_v.map(|v| (k_int, v))
                } else {
                    None
                }
            })
            .collect(),
    })
}

/// Compute the lengths of the shortest paths for a PyDiGraph object using
/// Dijkstra's algorithm
///
/// :param PyDiGraph graph: The input graph to use
/// :param int node: The node index to use as the source for finding the
///     shortest paths from
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the
///     cost/weight of that edge. It must be non-negative
/// :param int goal: An optional node index to use as the end of the path.
///     When specified the traversal will stop when the goal is reached and
///     the output dictionary will only have a single entry with the length
///     of the shortest path to the goal node.
///
/// :returns: A dictionary of the shortest paths from the provided node where
///     the key is the node index of the end of the path and the value is the
///     cost/sum of the weights of path
/// :rtype: PathLengthMapping
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, node, edge_cost_fn, /, goal=None)")]
pub fn digraph_dijkstra_shortest_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let edge_cost_callable = CostFn::from(edge_cost_fn);

    let start = NodeIndex::new(node);
    let goal_index: Option<NodeIndex> = goal.map(NodeIndex::new);

    let res: Vec<Option<f64>> = dijkstra(
        &graph.graph,
        start,
        goal_index,
        |e| edge_cost_callable.call(py, e.weight()),
        None,
    )?;

    if let Some(goal_usize) = goal {
        return Ok(PathLengthMapping {
            path_lengths: match res[goal_usize] {
                Some(goal_length) => {
                    let mut ans = DictMap::new();
                    ans.insert(goal_usize, goal_length);
                    ans
                }
                None => DictMap::new(),
            },
        });
    }

    Ok(PathLengthMapping {
        path_lengths: res
            .into_iter()
            .enumerate()
            .filter_map(|(k_int, opt_v)| {
                if k_int != node {
                    opt_v.map(|v| (k_int, v))
                } else {
                    None
                }
            })
            .collect(),
    })
}

/// For each node in the graph, calculates the lengths of the shortest paths
/// to all others in a :class:`~rustworkx.PyDiGraph` object
///
/// This function will calculate the shortest path lengths from all nodes in the
/// graph using Dijkstra's algorithm. This function is multithreaded and will
/// launch a thread pool with threads equal to the number of CPUs by
/// default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// :param graph: The input :class:`~rustworkx.PyDiGraph` to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {1: 2.0, 2: 2.0},
///             1: {2: 1.0},
///             2: {0: 1.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn digraph_all_pairs_dijkstra_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    all_pairs_dijkstra::all_pairs_dijkstra_path_lengths(py, &graph.graph, edge_cost_fn)
}

/// For each node in the graph, finds the shortest paths to all others in a
/// :class:`~rustworkx.PyDiGraph` object
///
/// This function will generate the shortest paths from all nodes in the graph
/// Dijkstra's algorithm. This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// :param graph: The input :class:`~rustworkx.PyDiGraph` object to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of paths. The keys are source node indices
///     and the values are dicts of the target node and the list of the
///     node indices making up the shortest path to that node. For example::
///
///         {
///             0: {1: [0, 1],  2: [0, 1, 2]},
///             1: {2: [1, 2]},
///             2: {0: [2, 0]},
///         }
///
/// :rtype: AllPairsPathMapping
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn digraph_all_pairs_dijkstra_shortest_paths(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    all_pairs_dijkstra::all_pairs_dijkstra_shortest_paths(py, &graph.graph, edge_cost_fn, None)
}

/// For each node in the graph, calculates the lengths of the shortest paths
/// to all others in a :class:`~rustworkx.PyGraph` object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param graph: The input :class:`~rustworkx.PyGraph` to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {1: 2.0, 2: 2.0},
///             1: {2: 1.0},
///             2: {0: 1.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn graph_all_pairs_dijkstra_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    all_pairs_dijkstra::all_pairs_dijkstra_path_lengths(py, &graph.graph, edge_cost_fn)
}

/// For each node in the graph, finds the shortest paths to all others in a
/// :class:`~rustworkx.PyGraph` object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param graph: The input :class:`~rustworkx.PyGraph` object to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of paths. The keys are destination node
///     indices and the values are dicts of the target node and the list of the
///     node indices making up the shortest path to that node. For example::
///
///         {
///             0: {1: [0, 1],  2: [0, 1, 2]},
///             1: {2: [1, 2]},
///             2: {0: [2, 0]},
///         }
///
/// :rtype: AllPairsPathMapping
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn graph_all_pairs_dijkstra_shortest_paths(
    py: Python,
    graph: &graph::PyGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    all_pairs_dijkstra::all_pairs_dijkstra_shortest_paths(py, &graph.graph, edge_cost_fn, None)
}

/// Compute the A* shortest path for a PyDiGraph
///
/// :param PyDiGraph graph: The input graph to use
/// :param int node: The node index to compute the path from
/// :param goal_fn: A python callable that will take in 1 parameter, a node's
///     data object and will return a boolean which will be True if it is the
///     finish node.
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the cost of
///     that edge. It must be non-negative.
/// :param estimate_cost_fn: A python callable that will take in 1 parameter, a
///     node's data object and will return a float which represents the
///     estimated cost for the next node. The return must be non-negative. For
///     the algorithm to find the actual shortest path, it should be
///     admissible, meaning that it should never overestimate the actual cost
///     to get to the nearest goal node.
///
/// :return: The computed shortest path between node and finish as a list
///     of node indices.
/// :rtype: NodeIndices
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, node, goal_fn, edge_cost_fn, estimate_cost_fn, /)")]
pub fn digraph_astar_shortest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<NodeIndices> {
    let goal_fn_callable = |a: &PyObject| -> PyResult<bool> {
        let res = goal_fn.call1(py, (a,))?;
        let output: bool = res.extract(py)?;
        Ok(output)
    };

    let edge_cost_callable = CostFn::from(edge_cost_fn);
    let estimate_cost_callable = CostFn::from(estimate_cost_fn);
    let start = NodeIndex::new(node);

    let astar_res = astar(
        &graph.graph,
        start,
        |f| goal_fn_callable(graph.graph.node_weight(f).unwrap()),
        |e| edge_cost_callable.call(py, e.weight()),
        |estimate| estimate_cost_callable.call(py, &graph.graph[estimate]),
    )?;
    let path = match astar_res {
        Some(path) => path,
        None => return Err(NoPathFound::new_err("No path found that satisfies goal_fn")),
    };
    Ok(NodeIndices {
        nodes: path.1.into_iter().map(|x| x.index()).collect(),
    })
}

/// Compute the A* shortest path for a PyGraph
///
/// :param PyGraph graph: The input graph to use
/// :param int node: The node index to compute the path from
/// :param goal_fn: A python callable that will take in 1 parameter, a node's data
///     object and will return a boolean which will be True if it is the finish
///     node.
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an edge's
///     data object and will return a float that represents the cost of that
///     edge. It must be non-negative.
/// :param estimate_cost_fn: A python callable that will take in 1 parameter, a
///     node's data object and will return a float which represents the estimated
///     cost for the next node. The return must be non-negative. For the
///     algorithm to find the actual shortest path, it should be admissible,
///     meaning that it should never overestimate the actual cost to get to the
///     nearest goal node.
///
/// :returns: The computed shortest path between node and finish as a list
///     of node indices.
/// :rtype: NodeIndices
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, node, goal_fn, edge_cost_fn, estimate_cost_fn /)")]
pub fn graph_astar_shortest_path(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<NodeIndices> {
    let goal_fn_callable = |a: &PyObject| -> PyResult<bool> {
        let res = goal_fn.call1(py, (a,))?;
        let output: bool = res.extract(py)?;
        Ok(output)
    };

    let edge_cost_callable = CostFn::from(edge_cost_fn);
    let estimate_cost_callable = CostFn::from(estimate_cost_fn);
    let start = NodeIndex::new(node);

    let astar_res = astar(
        &graph.graph,
        start,
        |f| goal_fn_callable(graph.graph.node_weight(f).unwrap()),
        |e| edge_cost_callable.call(py, e.weight()),
        |estimate| estimate_cost_callable.call(py, &graph.graph[estimate]),
    )?;
    let path = match astar_res {
        Some(path) => path,
        None => return Err(NoPathFound::new_err("No path found that satisfies goal_fn")),
    };
    Ok(NodeIndices {
        nodes: path.1.into_iter().map(|x| x.index()).collect(),
    })
}

/// Compute the length of the kth shortest path
///
/// Computes the lengths of the kth shortest path from ``start`` to every
/// reachable node.
///
/// Computes in :math:`O(k * (|E| + |V|*log(|V|)))` time (average).
///
/// :param PyGraph graph: The graph to find the shortest paths in
/// :param int start: The node index to find the shortest paths from
/// :param int k: The kth shortest path to find the lengths of
/// :param edge_cost: A python callable that will receive an edge payload and
///     return a float for the cost of that eedge
/// :param int goal: An optional goal node index, if specified the output
///     dictionary
///
/// :returns: A dict of lengths where the key is the destination node index and
///     the value is the length of the path.
/// :rtype: PathLengthMapping
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, start, k, edge_cost, /, goal=None)")]
pub fn digraph_k_shortest_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    start: usize,
    k: usize,
    edge_cost: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let out_goal = goal.map(NodeIndex::new);
    let edge_cost_callable = CostFn::from(edge_cost);

    let out_map: Vec<Option<f64>> =
        k_shortest_path(&graph.graph, NodeIndex::new(start), out_goal, k, |e| {
            edge_cost_callable.call(py, e.weight())
        })?;

    if let Some(goal_usize) = goal {
        return Ok(PathLengthMapping {
            path_lengths: match out_map[goal_usize] {
                Some(goal_length) => {
                    let mut ans = DictMap::new();
                    ans.insert(goal_usize, goal_length);
                    ans
                }
                None => DictMap::new(),
            },
        });
    }

    Ok(PathLengthMapping {
        path_lengths: out_map
            .into_iter()
            .enumerate()
            .filter_map(|(k_int, opt_v)| opt_v.map(|v| (k_int, v)))
            .collect(),
    })
}

/// Compute the length of the kth shortest path
///
/// Computes the lengths of the kth shortest path from ``start`` to every
/// reachable node.
///
/// Computes in :math:`O(k * (|E| + |V|*log(|V|)))` time (average).
///
/// :param PyGraph graph: The graph to find the shortest paths in
/// :param int start: The node index to find the shortest paths from
/// :param int k: The kth shortest path to find the lengths of
/// :param edge_cost: A python callable that will receive an edge payload and
///     return a float for the cost of that eedge
/// :param int goal: An optional goal node index, if specified the output
///     dictionary
///
/// :returns: A dict of lengths where the key is the destination node index and
///     the value is the length of the path.
/// :rtype: PathLengthMapping
/// :raises ValueError: when an edge weight with NaN or negative value
///     is provided.
#[pyfunction]
#[pyo3(text_signature = "(graph, start, k, edge_cost, /, goal=None)")]
pub fn graph_k_shortest_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    start: usize,
    k: usize,
    edge_cost: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let out_goal = goal.map(NodeIndex::new);
    let edge_cost_callable = CostFn::from(edge_cost);

    let out_map: Vec<Option<f64>> =
        k_shortest_path(&graph.graph, NodeIndex::new(start), out_goal, k, |e| {
            edge_cost_callable.call(py, e.weight())
        })?;

    if let Some(goal_usize) = goal {
        return Ok(PathLengthMapping {
            path_lengths: match out_map[goal_usize] {
                Some(goal_length) => {
                    let mut ans = DictMap::new();
                    ans.insert(goal_usize, goal_length);
                    ans
                }
                None => DictMap::new(),
            },
        });
    }

    Ok(PathLengthMapping {
        path_lengths: out_map
            .into_iter()
            .enumerate()
            .filter_map(|(k_int, opt_v)| opt_v.map(|v| (k_int, v)))
            .collect(),
    })
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// This function is multithreaded and will launch a pool with threads equal
/// to the number of CPUs by default if the number of nodes in the graph is
/// above the value of ``parallel_threshold`` (it defaults to 300).
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads if parallelization was enabled.
///
/// :param PyDiGraph graph: The directed graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells rustworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         digraph_floyd_warshall(graph, weight_fn= lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         digraph_floyd_warshall(graph, weight_fn=float)
///
///     to cast the edge object as a float as the weight.
/// :param as_undirected: If set to true each directed edge will be treated as
///     bidirectional/undirected.
/// :param int parallel_threshold: The number of nodes to execute
///     the algorithm in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {0: 0.0, 1: 2.0, 2: 2.0},
///             1: {1: 0.0, 2: 1.0},
///             2: {0: 1.0, 2: 0.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
#[pyfunction(
    parallel_threshold = "300",
    as_undirected = "false",
    default_weight = "1.0"
)]
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, as_undirected=False, default_weight=1.0, parallel_threshold=300)"
)]
pub fn digraph_floyd_warshall(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    as_undirected: bool,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<AllPairsPathLengthMapping> {
    floyd_warshall::floyd_warshall(
        py,
        &graph.graph,
        weight_fn,
        as_undirected,
        default_weight,
        parallel_threshold,
    )
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// This function is multithreaded and will launch a pool with threads equal
/// to the number of CPUs by default if the number of nodes in the graph is
/// above the value of ``parallel_threshold`` (it defaults to 300).
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads if parallelization was enabled.
///
/// :param PyGraph graph: The graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells rustworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_floyd_warshall(graph, weight_fn= lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_floyd_warshall(graph, weight_fn=float)
///
///     to cast the edge object as a float as the weight.
/// :param int parallel_threshold: The number of nodes to execute
///     the algorithm in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {0: 0.0, 1: 2.0, 2: 2.0},
///             1: {1: 0.0, 2: 1.0},
///             2: {0: 1.0, 2: 0.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
#[pyfunction(parallel_threshold = "300", default_weight = "1.0")]
#[pyo3(text_signature = "(graph, /, weight_fn=None, default_weight=1.0, parallel_threshold=300)")]
pub fn graph_floyd_warshall(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<AllPairsPathLengthMapping> {
    let as_undirected = true;
    floyd_warshall::floyd_warshall(
        py,
        &graph.graph,
        weight_fn,
        as_undirected,
        default_weight,
        parallel_threshold,
    )
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// This function is multithreaded and will launch a pool with threads equal
/// to the number of CPUs by default if the number of nodes in the graph is
/// above the value of ``parallel_threshold`` (it defaults to 300).
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads if parallelization was enabled.
///
/// :param PyGraph graph: The graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells rustworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight.
/// :param int parallel_threshold: The number of nodes to execute
///     the algorithm in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :returns: A matrix of shortest path distances between nodes. If there is no
///     path between two nodes then the corresponding matrix entry will be
///     ``np.inf``.
/// :rtype: numpy.ndarray
#[pyfunction(parallel_threshold = "300", default_weight = "1.0")]
#[pyo3(text_signature = "(graph, /, weight_fn=None, default_weight=1.0, parallel_threshold=300)")]
pub fn graph_floyd_warshall_numpy(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<PyObject> {
    let matrix = floyd_warshall::floyd_warshall_numpy(
        py,
        &graph.graph,
        weight_fn,
        true,
        default_weight,
        parallel_threshold,
    )?;
    Ok(matrix.into_pyarray(py).into())
}

/// Find all-pairs shortest path lengths using Floyd's algorithm
///
/// Floyd's algorithm is used for finding shortest paths in dense graphs
/// or graphs with negative weights (where Dijkstra's algorithm fails).
///
/// This function is multithreaded and will launch a pool with threads equal
/// to the number of CPUs by default if the number of nodes in the graph is
/// above the value of ``parallel_threshold`` (it defaults to 300).
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads if parallelization was enabled.
///
/// :param PyDiGraph graph: The directed graph to run Floyd's algorithm on
/// :param weight_fn: A callable object (function, lambda, etc) which
///     will be passed the edge object and expected to return a ``float``. This
///     tells rustworkx/rust how to extract a numerical weight as a ``float``
///     for edge object. Some simple examples are::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: 1)
///
///     to return a weight of 1 for all edges. Also::
///
///         graph_floyd_warshall_numpy(graph, weight_fn: lambda x: float(x))
///
///     to cast the edge object as a float as the weight.
/// :param as_undirected: If set to true each directed edge will be treated as
///     bidirectional/undirected.
/// :param int parallel_threshold: The number of nodes to execute
///     the algorithm in parallel at. It defaults to 300, but this can
///     be tuned
///
/// :returns: A matrix of shortest path distances between nodes. If there is no
///     path between two nodes then the corresponding matrix entry will be
///     ``np.inf``.
/// :rtype: numpy.ndarray
#[pyfunction(
    parallel_threshold = "300",
    as_undirected = "false",
    default_weight = "1.0"
)]
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, as_undirected=False, default_weight=1.0, parallel_threshold=300)"
)]
pub fn digraph_floyd_warshall_numpy(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
    as_undirected: bool,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<PyObject> {
    let matrix = floyd_warshall::floyd_warshall_numpy(
        py,
        &graph.graph,
        weight_fn,
        as_undirected,
        default_weight,
        parallel_threshold,
    )?;
    Ok(matrix.into_pyarray(py).into())
}

/// Get the number of unweighted shortest paths from a source node
///
/// :param PyDiGraph graph: The graph to find the number of shortest paths on
/// :param int source: The source node to find the shortest paths from
///
/// :returns: A mapping of target node indices to the number of shortest paths
///     from ``source`` to that node. If there is no path from ``source`` to
///     a node in the graph that node will not be preset in the output mapping.
/// :rtype: NodesCountMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, source, /)")]
pub fn digraph_num_shortest_paths_unweighted(
    graph: &digraph::PyDiGraph,
    source: usize,
) -> PyResult<NodesCountMapping> {
    Ok(NodesCountMapping {
        map: num_shortest_path::num_shortest_paths_unweighted(&graph.graph, source)?,
    })
}

/// Get the number of unweighted shortest paths from a source node
///
/// :param PyGraph graph: The graph to find the number of shortest paths on
/// :param int source: The source node to find the shortest paths from
///
/// :returns: A mapping of target node indices to the number of shortest paths
///     from ``source`` to that node. If there is no path from ``source`` to
///     a node in the graph that node will not be preset in the output mapping.
/// :rtype: NumPathsMapping
#[pyfunction]
#[pyo3(text_signature = "(graph, source, /)")]
pub fn graph_num_shortest_paths_unweighted(
    graph: &graph::PyGraph,
    source: usize,
) -> PyResult<NodesCountMapping> {
    Ok(NodesCountMapping {
        map: num_shortest_path::num_shortest_paths_unweighted(&graph.graph, source)?,
    })
}

/// Get the distance matrix for a directed graph
///
/// This differs from functions like digraph_floyd_warshall_numpy in that the
/// edge weight/data payload is not used and each edge is treated as a
/// distance of 1.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold`` (it
/// defaults to 300). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// :param PyDiGraph graph: The graph to get the distance matrix for
/// :param int parallel_threshold: The number of nodes to calculate the
///     the distance matrix in parallel at. It defaults to 300, but this can
///     be tuned
/// :param bool as_undirected: If set to ``True`` the input directed graph
///     will be treat as if each edge was bidirectional/undirected in the
///     output distance matrix.
/// :param float null_value: An optional float that will treated as a null
///     value. This element will be the default in the matrix and represents
///     the absense of a path in the graph. By default this is ``0.0``.
///
/// :returns: The distance matrix
/// :rtype: numpy.ndarray
#[pyfunction(
    parallel_threshold = "300",
    as_undirected = "false",
    null_value = "0.0"
)]
#[pyo3(text_signature = "(graph, /, parallel_threshold=300, as_undirected=False, null_value=0.0)")]
pub fn digraph_distance_matrix(
    py: Python,
    graph: &digraph::PyDiGraph,
    parallel_threshold: usize,
    as_undirected: bool,
    null_value: f64,
) -> PyObject {
    let matrix = distance_matrix::compute_distance_matrix(
        &graph.graph,
        parallel_threshold,
        as_undirected,
        null_value,
    );
    matrix.into_pyarray(py).into()
}

/// Get the distance matrix for an undirected graph
///
/// This differs from functions like digraph_floyd_warshall_numpy in that the
/// edge weight/data payload is not used and each edge is treated as a
/// distance of 1.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``paralllel_threshold`` (it
/// defaults to 300). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
///
/// :param PyGraph graph: The graph to get the distance matrix for
/// :param int parallel_threshold: The number of nodes to calculate the
///     the distance matrix in parallel at. It defaults to 300, but this can
///     be tuned
/// :param float null_value: An optional float that will treated as a null
///     value. This element will be the default in the matrix and represents
///     the absense of a path in the graph. By default this is ``0.0``.
///
/// :returns: The distance matrix
/// :rtype: numpy.ndarray
#[pyfunction(parallel_threshold = "300", null_value = "0.0")]
#[pyo3(text_signature = "(graph, /, parallel_threshold=300, null_value=0.0)")]
pub fn graph_distance_matrix(
    py: Python,
    graph: &graph::PyGraph,
    parallel_threshold: usize,
    null_value: f64,
) -> PyObject {
    let matrix = distance_matrix::compute_distance_matrix(
        &graph.graph,
        parallel_threshold,
        true,
        null_value,
    );
    matrix.into_pyarray(py).into()
}

/// Return the average shortest path length for a :class:`~rustworkx.PyDiGraph`
/// with unweighted edges.
///
/// The average shortest path length is calculated as
///
/// .. math::
///
///     a =\sum_{s,t \in V, s \ne t} \frac{d(s, t)}{n(n-1)}
///
/// where :math:`V` is the set of nodes in ``graph``, :math:`d(s, t)` is the
/// shortest path length from :math:`s` to :math:`t`, and :math:`n` is the
/// number of nodes in ``graph``. If ``disconnected`` is set to ``True``,
/// the average will be taken only between connected nodes.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold`` (it
/// defaults to 300). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
/// By default it will use all available CPUs if the environment variable is
/// not specified.
///
/// :param PyDiGraph graph: The graph to compute the average shortest path length
///     for
/// :param int parallel_threshold: The number of nodes to calculate the
///     the distance matrix in parallel at. It defaults to 300, but this can
///     be tuned to any number of nodes.
/// :param bool as_undirected: If set to ``True`` the input directed graph
///     will be treated as if each edge was bidirectional/undirected while
///     finding the shortest paths. Default: ``False``.
/// :param bool disconnected: If set to ``True`` only connected vertex pairs
///     will be included in the calculation. If ``False``, infinity is returned
///     for disconnected graphs. Default: ``False``.
///
/// :returns: The average shortest path length. If no vertex pairs can be included
///     in the calculation this will return NaN.
/// :rtype: float
#[pyfunction(
    parallel_threshold = "300",
    as_undirected = "false",
    disconnected = "false"
)]
#[pyo3(
    text_signature = "(graph, /, parallel_threshold=300, as_undirected=False, disconnected=False)"
)]
pub fn digraph_unweighted_average_shortest_path_length(
    graph: &digraph::PyDiGraph,
    parallel_threshold: usize,
    as_undirected: bool,
    disconnected: bool,
) -> f64 {
    let n = graph.node_count();
    if n <= 1 {
        return std::f64::NAN;
    }

    let (sum, conn_pairs) =
        average_length::compute_distance_sum(&graph.graph, parallel_threshold, as_undirected);

    let tot_pairs = n * (n - 1);
    if disconnected && conn_pairs == 0 {
        return std::f64::NAN;
    }

    if !disconnected && conn_pairs < tot_pairs {
        return std::f64::INFINITY;
    }

    (sum as f64) / (conn_pairs as f64)
}

/// Return the average shortest path length for a :class:`~rustworkx.PyGraph`
/// with unweighted edges.
///
/// The average shortest path length is calculated as
///
/// .. math::
///
///     a =\sum_{s,t \in V, s \ne t} \frac{d(s, t)}{n(n-1)}
///
/// where :math:`V` is the set of nodes in ``graph``, :math:`d(s, t)` is the
/// shortest path length from node :math:`s` to node :math:`t`, and :math:`n`
/// is the number of nodes in ``graph``. If ``disconnected`` is set to ``True``,
/// the average will be taken only between connected nodes.
///
/// This function is also multithreaded and will run in parallel if the number
/// of nodes in the graph is above the value of ``parallel_threshold`` (it
/// defaults to 300). If the function will be running in parallel the env var
/// ``RAYON_NUM_THREADS`` can be used to adjust how many threads will be used.
/// By default it will use all available CPUs if the environment variable is
/// not specified.
///
/// :param PyGraph graph: The graph to compute the average shortest path length
///     for
/// :param int parallel_threshold: The number of nodes to calculate the
///     the distance matrix in parallel at. It defaults to 300, but this can
///     be tuned to any number of nodes.
/// :param bool disconnected: If set to ``True`` only connected vertex pairs
///     will be included in the calculation. If ``False``, infinity is returned
///     for disconnected graphs. Default: ``False``.
///
/// :returns: The average shortest path length. If no vertex pairs can be included
///     in the calculation this will return NaN.
/// :rtype: float
#[pyfunction(parallel_threshold = "300", disconnected = "false")]
#[pyo3(text_signature = "(graph, /, parallel_threshold=300, disconnected=False)")]
pub fn graph_unweighted_average_shortest_path_length(
    graph: &graph::PyGraph,
    parallel_threshold: usize,
    disconnected: bool,
) -> f64 {
    let n = graph.node_count();
    if n <= 1 {
        return std::f64::NAN;
    }

    let (sum, conn_pairs) =
        average_length::compute_distance_sum(&graph.graph, parallel_threshold, true);

    let tot_pairs = n * (n - 1);
    if disconnected && conn_pairs == 0 {
        return std::f64::NAN;
    }

    if !disconnected && conn_pairs < tot_pairs {
        return std::f64::INFINITY;
    }

    (sum as f64) / (conn_pairs as f64)
}

/// Compute the lengths of the shortest paths for a PyDiGraph object using
/// the Bellman-Ford algorithm with the SPFA heuristic.
///
/// :param PyDiGraph graph: The input graph to use
/// :param int node: The node index to use as the source for finding the
///     shortest paths from
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the
///     cost/weight of that edge. It can be negative.
/// :param int goal: An optional node index to use as the end of the path.
///     When specified the output dictionary will only have a single entry with
///     the length of the shortest path to the goal node.
///
/// :returns: A read-only dictionary of the shortest paths from the provided node where
///     the key is the node index of the end of the path and the value is the
///     cost/sum of the weights of path
/// :rtype: PathLengthMapping
///
/// :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
///     path is not defined.
#[pyfunction]
#[pyo3(text_signature = "(graph, node, edge_cost_fn, /, goal=None)")]
pub fn digraph_bellman_ford_shortest_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let edge_weights: Vec<Option<f64>> =
        edge_weights_from_callable(py, &graph.graph, &Some(edge_cost_fn), 1.0)?;
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };

    let start = NodeIndex::new(node);

    let res: Option<Vec<Option<f64>>> =
        bellman_ford(&graph.graph, start, |e| edge_cost(e.id()), None)?;

    if res.is_none() {
        return Err(NegativeCycle::new_err(
            "The shortest-path is not defined because there is a negative cycle",
        ));
    }

    let res = res.unwrap();

    if let Some(goal_usize) = goal {
        return Ok(PathLengthMapping {
            path_lengths: match res[goal_usize] {
                Some(goal_length) => {
                    let mut ans = DictMap::new();
                    ans.insert(goal_usize, goal_length);
                    ans
                }
                None => DictMap::new(),
            },
        });
    }

    Ok(PathLengthMapping {
        path_lengths: res
            .into_iter()
            .enumerate()
            .filter_map(|(k_int, opt_v)| {
                if k_int != node {
                    opt_v.map(|v| (k_int, v))
                } else {
                    None
                }
            })
            .collect(),
    })
}

/// Compute the lengths of the shortest paths for a PyGraph object using
/// the Bellman-Ford algorithm with the SPFA heuristic.
///
/// :param PyGraph graph: The input graph to use
/// :param int node: The node index to use as the source for finding the
///     shortest paths from
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an
///     edge's data object and will return a float that represents the
///     cost/weight of that edge. It can be negative.
/// :param int goal: An optional node index to use as the end of the path.
///     When specified the output dictionary will only have a single entry with
///     the length of the shortest path to the goal node.
///
/// :returns: A read-only dictionary of the shortest paths from the provided node where
///     the key is the node index of the end of the path and the value is the
///     cost/sum of the weights of path
/// :rtype: PathLengthMapping
///
/// :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
///     path is not defined.
#[pyfunction]
#[pyo3(text_signature = "(graph, node, edge_cost_fn, /, goal=None)")]
pub fn graph_bellman_ford_shortest_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let edge_weights: Vec<Option<f64>> =
        edge_weights_from_callable(py, &graph.graph, &Some(edge_cost_fn), 1.0)?;
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };

    let start = NodeIndex::new(node);

    let res: Option<Vec<Option<f64>>> =
        bellman_ford(&graph.graph, start, |e| edge_cost(e.id()), None)?;

    if res.is_none() {
        return Err(NegativeCycle::new_err(
            "The shortest-path is not defined because there is a negative cycle",
        ));
    }

    let res = res.unwrap();

    if let Some(goal_usize) = goal {
        return Ok(PathLengthMapping {
            path_lengths: match res[goal_usize] {
                Some(goal_length) => {
                    let mut ans = DictMap::new();
                    ans.insert(goal_usize, goal_length);
                    ans
                }
                None => DictMap::new(),
            },
        });
    }

    Ok(PathLengthMapping {
        path_lengths: res
            .into_iter()
            .enumerate()
            .filter_map(|(k_int, opt_v)| {
                if k_int != node {
                    opt_v.map(|v| (k_int, v))
                } else {
                    None
                }
            })
            .collect(),
    })
}

/// Find the shortest path from a node
///
/// This function will generate the shortest path from a source node using
/// the Bellman-Ford algorithm with the SPFA heuristic.
///
/// :param PyGraph graph: The input graph to use
/// :param int source: The node index to find paths from
/// :param int target: An optional target to find a path to
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
/// :param bool as_undirected: If set to true the graph will be treated as
///     undirected for finding the shortest path.
///
/// :return: Read-only dictionary of paths. The keys are destination node indices and
///     the dict values are lists of node indices making the path.
/// :rtype: PathMapping
///
/// :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
///     path is not defined.
#[pyfunction(default_weight = "1.0", as_undirected = "false")]
#[pyo3(text_signature = "(graph, source, /, target=None, weight_fn=None, default_weight=1.0)")]
pub fn graph_bellman_ford_shortest_paths(
    py: Python,
    graph: &graph::PyGraph,
    source: usize,
    target: Option<usize>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<PathMapping> {
    let start = NodeIndex::new(source);
    let mut paths: DictMap<NodeIndex, Vec<NodeIndex>> = DictMap::with_capacity(graph.node_count());

    let edge_weights: Vec<Option<f64>> =
        edge_weights_from_callable(py, &graph.graph, &weight_fn, default_weight)?;
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };

    let res: Option<Vec<Option<f64>>> =
        bellman_ford(&graph.graph, start, |e| edge_cost(e.id()), Some(&mut paths))?;

    if res.is_none() {
        return Err(NegativeCycle::new_err(
            "The shortest-path is not defined because there is a negative cycle",
        ));
    }

    Ok(PathMapping {
        paths: paths
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == source || target.is_some() && target.unwrap() != k_int {
                    None
                } else {
                    Some((
                        k.index(),
                        v.iter().map(|x| x.index()).collect::<Vec<usize>>(),
                    ))
                }
            })
            .collect(),
    })
}

/// Find the shortest path from a node
///
/// This function will generate the shortest path from a source node using
/// the Bellman-Ford algorithm with the SPFA heuristic.
///
/// :param PyDiGraph graph: The input graph to use
/// :param int source: The node index to find paths from
/// :param int target: An optional target to find a path to
/// :param weight_fn: An optional weight function for an edge. It will accept
///     a single argument, the edge's weight object and will return a float which
///     will be used to represent the weight/cost of the edge
/// :param float default_weight: If ``weight_fn`` isn't specified this optional
///     float value will be used for the weight/cost of each edge.
/// :param bool as_undirected: If set to true the graph will be treated as
///     undirected for finding the shortest path.
///
/// :return: Read-only dictionary of paths. The keys are destination node indices and
///     the dict values are lists of node indices making the path.
/// :rtype: PathMapping
///
/// :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
///     path is not defined.
#[pyfunction(default_weight = "1.0", as_undirected = "false")]
#[pyo3(
    text_signature = "(graph, source, /, target=None, weight_fn=None, default_weight=1.0, as_undirected=False)"
)]
pub fn digraph_bellman_ford_shortest_paths(
    py: Python,
    graph: &digraph::PyDiGraph,
    source: usize,
    target: Option<usize>,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    as_undirected: bool,
) -> PyResult<PathMapping> {
    if as_undirected {
        return graph_bellman_ford_shortest_paths(
            py,
            &graph.to_undirected(py, true, None)?,
            source,
            target,
            weight_fn.map(|x| x.clone_ref(py)),
            default_weight,
        );
    }

    let start = NodeIndex::new(source);
    let mut paths: DictMap<NodeIndex, Vec<NodeIndex>> = DictMap::with_capacity(graph.node_count());

    let edge_weights: Vec<Option<f64>> =
        edge_weights_from_callable(py, &graph.graph, &weight_fn, default_weight)?;
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };

    let res: Option<Vec<Option<f64>>> =
        bellman_ford(&graph.graph, start, |e| edge_cost(e.id()), Some(&mut paths))?;

    if res.is_none() {
        return Err(NegativeCycle::new_err(
            "The shortest-path is not defined because there is a negative cycle",
        ));
    }

    Ok(PathMapping {
        paths: paths
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == source || target.is_some() && target.unwrap() != k_int {
                    None
                } else {
                    Some((
                        k.index(),
                        v.iter().map(|x| x.index()).collect::<Vec<usize>>(),
                    ))
                }
            })
            .collect(),
    })
}

/// Check if a negative cycle exists on a graph
///
/// This function will check for the existence of a negative cycle in a graph
/// using the Bellman-Ford algorithm with the SPFA heuristic.
///
/// :param PyDiGraph graph: The input graph to use
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an edge's
///     data object and will return a float that represents the cost of that
///     edge.
///
/// :return: True if there is a negative cycle or False otherwise
/// :rtype: bool
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn negative_edge_cycle(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<bool> {
    let edge_weights: Vec<Option<f64>> =
        edge_weights_from_callable(py, &graph.graph, &Some(edge_cost_fn), 1.0)?;
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };

    let cycle: Option<Vec<_>> = negative_cycle_finder(&graph.graph, |e| edge_cost(e.id()))?;

    Ok(cycle.is_some())
}

/// Find a negative cycle of a graph
///
/// This function will find an arbitrary negative cycle in a graph
/// using the Bellman-Ford algorithm with the SPFA heuristic.
///
/// :param PyDiGraph graph: The input graph to use
/// :param edge_cost_fn: A python callable that will take in 1 parameter, an edge's
///     data object and will return a float that represents the cost of that
///     edge.
///
/// :return: A list of the nodes in an arbitrary negative cycle, if it exists
/// :rtype: NodeIndices
///
/// :raises: ValueError: when there is no cycle in the graph provided
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn find_negative_cycle(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<NodeIndices> {
    let edge_weights: Vec<Option<f64>> =
        edge_weights_from_callable(py, &graph.graph, &Some(edge_cost_fn), 1.0)?;
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };

    let cycle: Option<Vec<_>> = negative_cycle_finder(&graph.graph, |e| edge_cost(e.id()))?;

    if cycle.is_none() {
        return Err(PyValueError::new_err("There is no negative cycle"));
    }

    let cycle = cycle.unwrap();

    Ok(NodeIndices {
        nodes: cycle.into_iter().map(|x| x.index()).collect(),
    })
}

/// For each node in the graph, calculates the lengths of the shortest paths
/// to all others in a :class:`~rustworkx.PyDiGraph` object
///
/// This function will calculate the shortest path lengths from all nodes in the
/// graph using the Bellman-Ford algorithm. This function is multithreaded and will
/// launch a thread pool with threads equal to the number of CPUs by
/// default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// :param graph: The input :class:`~rustworkx.PyDiGraph` to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {1: 2.0, 2: 2.0},
///             1: {2: 1.0},
///             2: {0: 1.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
///
/// :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
///     path is not defined.
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn digraph_all_pairs_bellman_ford_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    all_pairs_bellman_ford::all_pairs_bellman_ford_path_lengths(py, &graph.graph, edge_cost_fn)
}

/// For each node in the graph, finds the shortest paths to all others in a
/// :class:`~rustworkx.PyDiGraph` object
///
/// This function will generate the shortest paths from all nodes in the graph
/// the Bellman-Ford  algorithm. This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// :param graph: The input :class:`~rustworkx.PyDiGraph` object to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of paths. The keys are source node indices
///     and the values are dicts of the target node and the list of the
///     node indices making up the shortest path to that node. For example::
///
///         {
///             0: {1: [0, 1],  2: [0, 1, 2]},
///             1: {2: [1, 2]},
///             2: {0: [2, 0]},
///         }
///
/// :rtype: AllPairsPathMapping
///
/// :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
///     path is not defined.
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn digraph_all_pairs_bellman_ford_shortest_paths(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    all_pairs_bellman_ford::all_pairs_bellman_ford_shortest_paths(py, &graph.graph, edge_cost_fn)
}

/// For each node in the graph, calculates the lengths of the shortest paths
/// to all others in a :class:`~rustworkx.PyGraph` object
///
/// This function will generate the shortest path from a source node using
/// the Bellman-Ford  algorithm.
///
/// :param graph: The input :class:`~rustworkx.PyGraph` to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of path lengths. The keys are source
///     node indices and the values are dicts of the target node and the length
///     of the shortest path to that node. For example::
///
///         {
///             0: {1: 2.0, 2: 2.0},
///             1: {2: 1.0},
///             2: {0: 1.0},
///         }
///
/// :rtype: AllPairsPathLengthMapping
///
/// :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
///     path is not defined.
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn graph_all_pairs_bellman_ford_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    all_pairs_bellman_ford::all_pairs_bellman_ford_path_lengths(py, &graph.graph, edge_cost_fn)
}

/// For each node in the graph, finds the shortest paths to all others in a
/// :class:`~rustworkx.PyGraph` object
///
/// This function will generate the shortest path from a source node using
/// the Bellman-Ford algorithm.
///
/// :param graph: The input :class:`~rustworkx.PyGraph` object to use
/// :param edge_cost_fn: A callable object that acts as a weight function for
///     an edge. It will accept a single positional argument, the edge's weight
///     object and will return a float which will be used to represent the
///     weight/cost of the edge
///
/// :return: A read-only dictionary of paths. The keys are destination node
///     indices and the values are dicts of the target node and the list of the
///     node indices making up the shortest path to that node. For example::
///
///         {
///             0: {1: [0, 1],  2: [0, 1, 2]},
///             1: {2: [1, 2]},
///             2: {0: [2, 0]},
///         }
///
/// :rtype: AllPairsPathMapping
///
/// :raises: :class:`~rustworkx.NegativeCycle`: when there is a negative cycle and the shortest
///     path is not defined.
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn graph_all_pairs_bellman_ford_shortest_paths(
    py: Python,
    graph: &graph::PyGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    all_pairs_bellman_ford::all_pairs_bellman_ford_shortest_paths(py, &graph.graph, edge_cost_fn)
}
