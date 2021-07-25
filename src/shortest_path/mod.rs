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

mod all_pairs_dijkstra;
mod astar;
mod dijkstra;
mod floyd_warshall;
mod k_shortest_path;
mod num_shortest_path;

use hashbrown::{HashMap, HashSet};

use super::weight_callable;
use crate::{digraph, get_edge_iter_with_weights, graph, NoPathFound};

use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::visit::NodeCount;

use ndarray::prelude::*;
use numpy::IntoPyArray;
use rayon::prelude::*;

use crate::iterators::{
    AllPairsPathLengthMapping, AllPairsPathMapping, NodeIndices,
    NodesCountMapping, PathLengthMapping, PathMapping,
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
#[pyfunction(default_weight = "1.0", as_undirected = "false")]
#[pyo3(
    text_signature = "(graph, source, /, target=None weight_fn=None, default_weight=1.0)"
)]
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
    let mut paths: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(graph.node_count());
    dijkstra::dijkstra(
        graph,
        start,
        goal_index,
        |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
        Some(&mut paths),
    )?;

    Ok(PathMapping {
        paths: paths
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == source
                    || target.is_some() && target.unwrap() != k_int
                {
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
    let mut paths: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(graph.node_count());
    if as_undirected {
        dijkstra::dijkstra(
            // TODO: Use petgraph undirected adapter after
            // https://github.com/petgraph/petgraph/pull/318 is available in
            // a petgraph release.
            &graph.to_undirected(py, true, None)?,
            start,
            goal_index,
            |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
            Some(&mut paths),
        )?;
    } else {
        dijkstra::dijkstra(
            graph,
            start,
            goal_index,
            |e| weight_callable(py, &weight_fn, e.weight(), default_weight),
            Some(&mut paths),
        )?;
    }
    Ok(PathMapping {
        paths: paths
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == source
                    || target.is_some() && target.unwrap() != k_int
                {
                    None
                } else {
                    Some((
                        k_int,
                        v.iter().map(|x| x.index()).collect::<Vec<usize>>(),
                    ))
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
#[pyfunction]
#[pyo3(text_signature = "(graph, node, edge_cost_fn, /, goal=None)")]
pub fn graph_dijkstra_shortest_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        raw.extract(py)
    };

    let start = NodeIndex::new(node);
    let goal_index: Option<NodeIndex> = goal.map(NodeIndex::new);

    let res = dijkstra::dijkstra(
        graph,
        start,
        goal_index,
        |e| edge_cost_callable(e.weight()),
        None,
    )?;
    Ok(PathLengthMapping {
        path_lengths: res
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == node || goal.is_some() && goal.unwrap() != k_int {
                    None
                } else {
                    Some((k_int, *v))
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
#[pyfunction]
#[pyo3(text_signature = "(graph, node, edge_cost_fn, /, goal=None)")]
pub fn digraph_dijkstra_shortest_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    edge_cost_fn: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        raw.extract(py)
    };

    let start = NodeIndex::new(node);
    let goal_index: Option<NodeIndex> = goal.map(NodeIndex::new);

    let res = dijkstra::dijkstra(
        graph,
        start,
        goal_index,
        |e| edge_cost_callable(e.weight()),
        None,
    )?;
    Ok(PathLengthMapping {
        path_lengths: res
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if k_int == node || goal.is_some() && goal.unwrap() != k_int {
                    None
                } else {
                    Some((k_int, *v))
                }
            })
            .collect(),
    })
}

/// Calculate the the shortest length from all nodes in a
/// :class:`~retworkx.PyDiGraph` object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm. This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// :param graph: The input :class:`~retworkx.PyDiGraph` to use
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
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn digraph_all_pairs_dijkstra_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    all_pairs_dijkstra::all_pairs_dijkstra_path_lengths(
        py,
        &graph.graph,
        edge_cost_fn,
    )
}

/// Find the shortest path from all nodes in a :class:`~retworkx.PyDiGraph`
/// object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm. This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// :param graph: The input :class:`~retworkx.PyDiGraph` object to use
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
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn digraph_all_pairs_dijkstra_shortest_paths(
    py: Python,
    graph: &digraph::PyDiGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    all_pairs_dijkstra::all_pairs_dijkstra_shortest_paths(
        py,
        &graph.graph,
        edge_cost_fn,
    )
}

/// Calculate the the shortest length from all nodes in a
/// :class:`~retworkx.PyGraph` object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param graph: The input :class:`~retworkx.PyGraph` to use
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
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn graph_all_pairs_dijkstra_path_lengths(
    py: Python,
    graph: &graph::PyGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    all_pairs_dijkstra::all_pairs_dijkstra_path_lengths(
        py,
        &graph.graph,
        edge_cost_fn,
    )
}

/// Find the shortest path from all nodes in a :class:`~retworkx.PyGraph`
/// object
///
/// This function will generate the shortest path from a source node using
/// Dijkstra's algorithm.
///
/// :param graph: The input :class:`~retworkx.PyGraph` object to use
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
#[pyfunction]
#[pyo3(text_signature = "(graph, edge_cost_fn, /)")]
pub fn graph_all_pairs_dijkstra_shortest_paths(
    py: Python,
    graph: &graph::PyGraph,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathMapping> {
    all_pairs_dijkstra::all_pairs_dijkstra_shortest_paths(
        py,
        &graph.graph,
        edge_cost_fn,
    )
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
#[pyfunction]
#[pyo3(text_signature = "(graph, node, goal_fn, edge_cost, estimate_cost, /)")]
fn digraph_astar_shortest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<NodeIndices> {
    let goal_fn_callable = |a: &PyObject| -> PyResult<bool> {
        let res = goal_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: bool = raw.extract(py)?;
        Ok(output)
    };

    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };

    let estimate_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = estimate_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };
    let start = NodeIndex::new(node);

    let astar_res = astar::astar(
        graph,
        start,
        |f| goal_fn_callable(graph.graph.node_weight(f).unwrap()),
        |e| edge_cost_callable(e.weight()),
        |estimate| {
            estimate_cost_callable(graph.graph.node_weight(estimate).unwrap())
        },
    )?;
    let path = match astar_res {
        Some(path) => path,
        None => {
            return Err(NoPathFound::new_err(
                "No path found that satisfies goal_fn",
            ))
        }
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
#[pyfunction]
#[pyo3(text_signature = "(graph, node, goal_fn, edge_cost, estimate_cost, /)")]
fn graph_astar_shortest_path(
    py: Python,
    graph: &graph::PyGraph,
    node: usize,
    goal_fn: PyObject,
    edge_cost_fn: PyObject,
    estimate_cost_fn: PyObject,
) -> PyResult<NodeIndices> {
    let goal_fn_callable = |a: &PyObject| -> PyResult<bool> {
        let res = goal_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: bool = raw.extract(py)?;
        Ok(output)
    };

    let edge_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = edge_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };

    let estimate_cost_callable = |a: &PyObject| -> PyResult<f64> {
        let res = estimate_cost_fn.call1(py, (a,))?;
        let raw = res.to_object(py);
        let output: f64 = raw.extract(py)?;
        Ok(output)
    };
    let start = NodeIndex::new(node);

    let astar_res = astar::astar(
        graph,
        start,
        |f| goal_fn_callable(graph.graph.node_weight(f).unwrap()),
        |e| edge_cost_callable(e.weight()),
        |estimate| {
            estimate_cost_callable(graph.graph.node_weight(estimate).unwrap())
        },
    )?;
    let path = match astar_res {
        Some(path) => path,
        None => {
            return Err(NoPathFound::new_err(
                "No path found that satisfies goal_fn",
            ))
        }
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
#[pyfunction]
#[pyo3(text_signature = "(graph, start, k, edge_cost, /, goal=None)")]
fn digraph_k_shortest_path_lengths(
    py: Python,
    graph: &digraph::PyDiGraph,
    start: usize,
    k: usize,
    edge_cost: PyObject,
    goal: Option<usize>,
) -> PyResult<PathLengthMapping> {
    let out_goal = goal.map(NodeIndex::new);
    let edge_cost_callable = |edge: &PyObject| -> PyResult<f64> {
        let res = edge_cost.call1(py, (edge,))?;
        res.extract(py)
    };

    let out_map = k_shortest_path::k_shortest_path(
        graph,
        NodeIndex::new(start),
        out_goal,
        k,
        edge_cost_callable,
    )?;
    Ok(PathLengthMapping {
        path_lengths: out_map
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if goal.is_some() && goal.unwrap() != k_int {
                    None
                } else {
                    Some((k_int, *v))
                }
            })
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
    let edge_cost_callable = |edge: &PyObject| -> PyResult<f64> {
        let res = edge_cost.call1(py, (edge,))?;
        res.extract(py)
    };

    let out_map = k_shortest_path::k_shortest_path(
        graph,
        NodeIndex::new(start),
        out_goal,
        k,
        edge_cost_callable,
    )?;
    Ok(PathLengthMapping {
        path_lengths: out_map
            .iter()
            .filter_map(|(k, v)| {
                let k_int = k.index();
                if goal.is_some() && goal.unwrap() != k_int {
                    None
                } else {
                    Some((k_int, *v))
                }
            })
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
///     tells retworkx/rust how to extract a numerical weight as a ``float``
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
fn digraph_floyd_warshall(
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
///     tells retworkx/rust how to extract a numerical weight as a ``float``
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
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, default_weight=1.0, parallel_threshold=300)"
)]
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
///     tells retworkx/rust how to extract a numerical weight as a ``float``
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
#[pyo3(
    text_signature = "(graph, /, weight_fn=None, default_weight=1.0, parallel_threshold=300)"
)]
pub fn graph_floyd_warshall_numpy(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
    parallel_threshold: usize,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    // Allocate empty matrix
    let mut mat = Array2::<f64>::from_elem((n, n), std::f64::INFINITY);

    // Build adjacency matrix
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
        mat[[i, j]] = mat[[i, j]].min(edge_weight);
        mat[[j, i]] = mat[[j, i]].min(edge_weight);
    }

    // 0 out the diagonal
    for x in mat.diag_mut() {
        *x = 0.0;
    }
    // Perform the Floyd-Warshall algorithm.
    // In each loop, this finds the shortest path from point i
    // to point j using intermediate nodes 0..k
    if n < parallel_threshold {
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let d_ijk = mat[[i, k]] + mat[[k, j]];
                    if d_ijk < mat[[i, j]] {
                        mat[[i, j]] = d_ijk;
                    }
                }
            }
        }
    } else {
        for k in 0..n {
            let row_k = mat.slice(s![k, ..]).to_owned();
            mat.axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row_i| {
                    let m_ik = row_i[k];
                    row_i.iter_mut().zip(row_k.iter()).for_each(
                        |(m_ij, m_kj)| {
                            let d_ijk = m_ik + *m_kj;
                            if d_ijk < *m_ij {
                                *m_ij = d_ijk;
                            }
                        },
                    )
                })
        }
    }
    Ok(mat.into_pyarray(py).into())
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
///     tells retworkx/rust how to extract a numerical weight as a ``float``
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
    let n = graph.node_count();

    // Allocate empty matrix
    let mut mat = Array2::<f64>::from_elem((n, n), std::f64::INFINITY);

    // Build adjacency matrix
    for (i, j, weight) in get_edge_iter_with_weights(graph) {
        let edge_weight =
            weight_callable(py, &weight_fn, &weight, default_weight)?;
        mat[[i, j]] = mat[[i, j]].min(edge_weight);
        if as_undirected {
            mat[[j, i]] = mat[[j, i]].min(edge_weight);
        }
    }
    // 0 out the diagonal
    for x in mat.diag_mut() {
        *x = 0.0;
    }
    // Perform the Floyd-Warshall algorithm.
    // In each loop, this finds the shortest path from point i
    // to point j using intermediate nodes 0..k
    if n < parallel_threshold {
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let d_ijk = mat[[i, k]] + mat[[k, j]];
                    if d_ijk < mat[[i, j]] {
                        mat[[i, j]] = d_ijk;
                    }
                }
            }
        }
    } else {
        for k in 0..n {
            let row_k = mat.slice(s![k, ..]).to_owned();
            mat.axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row_i| {
                    let m_ik = row_i[k];
                    row_i.iter_mut().zip(row_k.iter()).for_each(
                        |(m_ij, m_kj)| {
                            let d_ijk = m_ik + *m_kj;
                            if d_ijk < *m_ij {
                                *m_ij = d_ijk;
                            }
                        },
                    )
                })
        }
    }
    Ok(mat.into_pyarray(py).into())
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
        map: num_shortest_path::num_shortest_paths_unweighted(
            &graph.graph,
            source,
        )?,
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
        map: num_shortest_path::num_shortest_paths_unweighted(
            &graph.graph,
            source,
        )?,
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
///
/// :returns: The distance matrix
/// :rtype: numpy.ndarray
#[pyfunction(parallel_threshold = "300", as_undirected = "false")]
#[pyo3(
    text_signature = "(graph, /, parallel_threshold=300, as_undirected=False)"
)]
pub fn digraph_distance_matrix(
    py: Python,
    graph: &digraph::PyDiGraph,
    parallel_threshold: usize,
    as_undirected: bool,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    let bfs_traversal = |index: usize, mut row: ArrayViewMut1<f64>| {
        let mut seen: HashMap<NodeIndex, usize> = HashMap::with_capacity(n);
        let start_index = NodeIndex::new(index);
        let mut level = 0;
        let mut next_level: HashSet<NodeIndex> = HashSet::new();
        next_level.insert(start_index);
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<NodeIndex> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    row[[v.index()]] = level as f64;
                }
            }
            if seen.len() == n {
                return;
            }
            for node in found {
                for v in graph
                    .graph
                    .neighbors_directed(node, petgraph::Direction::Outgoing)
                {
                    next_level.insert(v);
                }
                if as_undirected {
                    for v in graph
                        .graph
                        .neighbors_directed(node, petgraph::Direction::Incoming)
                    {
                        next_level.insert(v);
                    }
                }
            }
            level += 1
        }
    };
    if n < parallel_threshold {
        matrix
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    } else {
        // Parallelize by row and iterate from each row index in BFS order
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    }
    Ok(matrix.into_pyarray(py).into())
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
///
/// :returns: The distance matrix
/// :rtype: numpy.ndarray
#[pyfunction(parallel_threshold = "300")]
#[pyo3(text_signature = "(graph, /, parallel_threshold=300)")]
pub fn graph_distance_matrix(
    py: Python,
    graph: &graph::PyGraph,
    parallel_threshold: usize,
) -> PyResult<PyObject> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::zeros((n, n));
    let bfs_traversal = |index: usize, mut row: ArrayViewMut1<f64>| {
        let mut seen: HashMap<NodeIndex, usize> = HashMap::with_capacity(n);
        let start_index = NodeIndex::new(index);
        let mut level = 0;
        let mut next_level: HashSet<NodeIndex> = HashSet::new();
        next_level.insert(start_index);
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<NodeIndex> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    row[[v.index()]] = level as f64;
                }
            }
            if seen.len() == n {
                return;
            }
            for node in found {
                for v in graph.graph.neighbors(node) {
                    next_level.insert(v);
                }
            }
            level += 1
        }
    };
    if n < parallel_threshold {
        matrix
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    } else {
        // Parallelize by row and iterate from each row index in BFS order
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    }
    Ok(matrix.into_pyarray(py).into())
}
