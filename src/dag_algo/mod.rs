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

use super::DictMap;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use rustworkx_core::dictmap::InitWithHasher;

use super::iterators::NodeIndices;
use crate::{digraph, DAGHasCycle, InvalidNode, StablePyGraph};

use rustworkx_core::dag_algo::collect_bicolor_runs as core_collect_bicolor_runs;
use rustworkx_core::dag_algo::lexicographical_topological_sort as core_lexico_topo_sort;
use rustworkx_core::dag_algo::longest_path as core_longest_path;
use rustworkx_core::traversal::dfs_edges;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::EdgeReference;
use petgraph::visit::NodeCount;

use num_traits::{Num, Zero};

/// Calculate the longest path in a directed acyclic graph (DAG).
///
/// This function interfaces with the Python `PyDiGraph` object to compute the longest path
/// using the provided weight function.
///
/// # Arguments
/// * `graph`: Reference to a `PyDiGraph` object.
/// * `weight_fn`: A callable that takes the source node index, target node index, and the weight
///   object and returns the weight of the edge as a `PyResult<T>`.
///
/// # Type Parameters
/// * `F`: Type of the weight function.
/// * `T`: The type of the edge weight. Must implement `Num`, `Zero`, `PartialOrd`, and `Copy`.
///
/// # Returns
/// * `PyResult<(Vec<G::NodeId>, T)>` representing the longest path as a sequence of node indices and its total weight.
fn longest_path<F, T>(graph: &digraph::PyDiGraph, mut weight_fn: F) -> PyResult<(Vec<usize>, T)>
where
    F: FnMut(usize, usize, &PyObject) -> PyResult<T>,
    T: Num + Zero + PartialOrd + Copy,
{
    let dag = &graph.graph;

    // Create a new weight function that matches the required signature
    let edge_cost = |edge_ref: EdgeReference<'_, PyObject>| -> Result<T, PyErr> {
        let source = edge_ref.source().index();
        let target = edge_ref.target().index();
        let weight = edge_ref.weight();
        weight_fn(source, target, weight)
    };

    let (path, path_weight) = match core_longest_path(dag, edge_cost) {
        Ok(Some((path, path_weight))) => (
            path.into_iter().map(NodeIndex::index).collect(),
            path_weight,
        ),
        Ok(None) => return Err(DAGHasCycle::new_err("The graph contains a cycle")),
        Err(e) => return Err(e),
    };

    Ok((path, path_weight))
}

/// Return a pair of [`petgraph::Direction`] values corresponding to the "forwards" and "backwards"
/// direction of graph traversal, based on whether the graph is being traved forwards (following
/// the edges) or backward (reversing along edges).  The order of returns is (forwards, backwards).
#[inline(always)]
pub fn traversal_directions(reverse: bool) -> (petgraph::Direction, petgraph::Direction) {
    if reverse {
        (petgraph::Direction::Outgoing, petgraph::Direction::Incoming)
    } else {
        (petgraph::Direction::Incoming, petgraph::Direction::Outgoing)
    }
}

/// Find the longest path in a DAG
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
/// :param weight_fn: A python callable that if set will be passed the 3
///     positional arguments, the source node, the target node, and the edge
///     weight for each edge as the function traverses the graph. It is expected
///     to return an unsigned integer weight for that edge. For example,
///     ``dag_longest_path(graph, lambda: _, __, weight: weight)`` could be
///     use to just use an integer edge weight. It's also worth noting that this
///     function traverses in topological order and only checks incoming edges to
///     each node.
///
/// :returns: The node indices of the longest path on the DAG
/// :rtype: NodeIndices
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[pyo3(text_signature = "(graph, /, weight_fn=None)")]
pub fn dag_longest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
) -> PyResult<NodeIndices> {
    let edge_weight_callable =
        |source: usize, target: usize, weight: &PyObject| -> PyResult<usize> {
            match &weight_fn {
                Some(weight_fn) => {
                    let res = weight_fn.call1(py, (source, target, weight))?;
                    res.extract(py)
                }
                None => Ok(1),
            }
        };
    Ok(NodeIndices {
        nodes: longest_path(graph, edge_weight_callable)?.0,
    })
}

/// Find the length of the longest path in a DAG
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
/// :param weight_fn: A python callable that if set will be passed the 3
///     positional arguments, the source node, the target node, and the edge
///     weight for each edge as the function traverses the graph. It is expected
///     to return an unsigned integer weight for that edge. For example,
///     ``dag_longest_path(graph, lambda: _, __, weight: weight)`` could be
///     use to just use an integer edge weight. It's also worth noting that this
///     function traverses in topological order and only checks incoming edges to
///     each node.
///
/// :returns: The longest path length on the DAG
/// :rtype: int
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[pyo3(text_signature = "(graph, /, weight_fn=None)")]
pub fn dag_longest_path_length(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<PyObject>,
) -> PyResult<usize> {
    let edge_weight_callable =
        |source: usize, target: usize, weight: &PyObject| -> PyResult<usize> {
            match &weight_fn {
                Some(weight_fn) => {
                    let res = weight_fn.call1(py, (source, target, weight))?;
                    res.extract(py)
                }
                None => Ok(1),
            }
        };
    let (_, path_weight) = longest_path(graph, edge_weight_callable)?;
    Ok(path_weight)
}

/// Find the weighted longest path in a DAG
///
/// This function differs from :func:`rustworkx.dag_longest_path` in that
/// this function requires a ``weight_fn`` parameter, and the ``weight_fn`` is
/// expected to return a ``float`` not an ``int``.
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
/// :param weight_fn: A python callable that will be passed the 3
///     positional arguments, the source node, the target node, and the edge
///     weight for each edge as the function traverses the graph. It is expected
///     to return a float weight for that edge. For example,
///     ``dag_longest_path(graph, lambda: _, __, weight: weight)`` could be
///     used to just use a float edge weight. It's also worth noting that this
///     function traverses in topological order and only checks incoming edges to
///     each node.
///
/// :returns: The node indices of the longest path on the DAG
/// :rtype: NodeIndices
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[pyo3(text_signature = "(graph, weight_fn, /)")]
pub fn dag_weighted_longest_path(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: PyObject,
) -> PyResult<NodeIndices> {
    let edge_weight_callable = |source: usize, target: usize, weight: &PyObject| -> PyResult<f64> {
        let res = weight_fn.call1(py, (source, target, weight))?;
        let float_res: f64 = res.extract(py)?;
        if float_res.is_nan() {
            return Err(PyValueError::new_err("NaN is not a valid edge weight"));
        }
        Ok(float_res)
    };
    Ok(NodeIndices {
        nodes: longest_path(graph, edge_weight_callable)?.0,
    })
}

/// Find the length of the weighted longest path in a DAG
///
/// This function differs from :func:`rustworkx.dag_longest_path_length` in that
/// this function requires a ``weight_fn`` parameter, and the ``weight_fn`` is
/// expected to return a ``float`` not an ``int``.
///
/// :param PyDiGraph graph: The graph to find the longest path on. The input
///     object must be a DAG without a cycle.
/// :param weight_fn: A python callable that will be passed the 3
///     positional arguments, the source node, the target node, and the edge
///     weight for each edge as the function traverses the graph. It is expected
///     to return a float weight for that edge. For example,
///     ``dag_longest_path(graph, lambda: _, __, weight: weight)`` could be
///     used to just use a float edge weight. It's also worth noting that this
///     function traverses in topological order and only checks incoming edges to
///     each node.
///
/// :returns: The longest path length on the DAG
/// :rtype: float
///
/// :raises Exception: If an unexpected error occurs or a path can't be found
/// :raises DAGHasCycle: If the input PyDiGraph has a cycle
#[pyfunction]
#[pyo3(text_signature = "(graph, weight_fn, /)")]
pub fn dag_weighted_longest_path_length(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: PyObject,
) -> PyResult<f64> {
    let edge_weight_callable = |source: usize, target: usize, weight: &PyObject| -> PyResult<f64> {
        let res = weight_fn.call1(py, (source, target, weight))?;
        let float_res: f64 = res.extract(py)?;
        if float_res.is_nan() {
            return Err(PyValueError::new_err("NaN is not a valid edge weight"));
        }
        Ok(float_res)
    };
    let (_, path_weight) = longest_path(graph, edge_weight_callable)?;
    Ok(path_weight)
}

/// Check that the PyDiGraph or PyDAG doesn't have a cycle
///
/// :param PyDiGraph graph: The graph to check for cycles
///
/// :returns: ``True`` if there are no cycles in the input graph, ``False``
///     if there are cycles
/// :rtype: bool
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn is_directed_acyclic_graph(graph: &digraph::PyDiGraph) -> bool {
    match algo::toposort(&graph.graph, None) {
        Ok(_nodes) => true,
        Err(_err) => false,
    }
}

/// Return a list of layers
///
/// A layer is a subgraph whose nodes are disjoint, i.e.,
/// a layer has depth 1. The layers are constructed using a greedy algorithm.
///
/// :param PyDiGraph graph: The DAG to get the layers from
/// :param list first_layer: A list of node ids for the first layer. This
///     will be the first layer in the output
/// :param bool index_output: When set to to ``True`` the output layers will be
///     a list of integer node indices.
///
/// :returns: A list of layers, each layer is a list of node data, or if
///     ``index_output`` is ``True`` each layer is a list of node indices.
/// :rtype: list
///
/// :raises InvalidNode: If a node index in ``first_layer`` is not in the graph
#[pyfunction]
#[pyo3(
    signature=(dag, first_layer, index_output=false),
    text_signature = "(dag, first_layer, /, index_output=False)"
)]
pub fn layers(
    py: Python,
    dag: &digraph::PyDiGraph,
    first_layer: Vec<usize>,
    index_output: bool,
) -> PyResult<PyObject> {
    let mut output_indices: Vec<Vec<usize>> = Vec::new();
    let mut output: Vec<Vec<&PyObject>> = Vec::new();
    // Convert usize to NodeIndex
    let mut first_layer_index: Vec<NodeIndex> = Vec::new();
    for index in first_layer {
        first_layer_index.push(NodeIndex::new(index));
    }

    let mut cur_layer = first_layer_index;
    let mut next_layer: Vec<NodeIndex> = Vec::new();
    let mut predecessor_count: HashMap<NodeIndex, usize> = HashMap::new();

    let mut layer_node_data: Vec<&PyObject> = Vec::new();
    if !index_output {
        for layer_node in &cur_layer {
            let node_data = match dag.graph.node_weight(*layer_node) {
                Some(data) => data,
                None => {
                    return Err(InvalidNode::new_err(format!(
                        "An index input in 'first_layer' {} is not a valid node index in the graph",
                        layer_node.index()
                    )))
                }
            };
            layer_node_data.push(node_data);
        }
        output.push(layer_node_data);
    } else {
        for layer_node in &cur_layer {
            if !dag.graph.contains_node(*layer_node) {
                return Err(InvalidNode::new_err(format!(
                    "An index input in 'first_layer' {} is not a valid node index in the graph",
                    layer_node.index()
                )));
            }
        }
        output_indices.push(cur_layer.iter().map(|x| x.index()).collect());
    }

    // Iterate until there are no more
    while !cur_layer.is_empty() {
        for node in &cur_layer {
            let children = dag
                .graph
                .neighbors_directed(*node, petgraph::Direction::Outgoing);
            let mut used_indices: HashSet<NodeIndex> = HashSet::new();
            for succ in children {
                // Skip duplicate successors
                if used_indices.contains(&succ) {
                    continue;
                }
                used_indices.insert(succ);
                let mut multiplicity: usize = 0;
                let raw_edges = dag
                    .graph
                    .edges_directed(*node, petgraph::Direction::Outgoing);
                for edge in raw_edges {
                    if edge.target() == succ {
                        multiplicity += 1;
                    }
                }
                predecessor_count
                    .entry(succ)
                    .and_modify(|e| *e -= multiplicity)
                    .or_insert(dag.in_degree(succ.index()) - multiplicity);
                if *predecessor_count.get(&succ).unwrap() == 0 {
                    next_layer.push(succ);
                    predecessor_count.remove(&succ);
                }
            }
        }
        if !index_output {
            let mut layer_node_data: Vec<&PyObject> = Vec::new();

            for layer_node in &next_layer {
                layer_node_data.push(&dag.graph[*layer_node]);
            }
            if !layer_node_data.is_empty() {
                output.push(layer_node_data);
            }
        } else if !next_layer.is_empty() {
            output_indices.push(next_layer.iter().map(|x| x.index()).collect());
        }
        cur_layer = next_layer;
        next_layer = Vec::new();
    }
    if !index_output {
        Ok(PyList::new_bound(py, output).into())
    } else {
        Ok(PyList::new_bound(py, output_indices).into())
    }
}

/// Get the lexicographical topological sorted nodes from the provided DAG
///
/// This function returns a list of nodes data in a graph lexicographically
/// topologically sorted using the provided key function. A topological sort
/// is a linear ordering of vertices such that for every directed edge from
/// node :math:`u` to node :math:`v`, :math:`u` comes before :math:`v`
/// in the ordering.  If ``reverse`` is set to ``False``, the edges are treated
/// as if they pointed in the opposite direction.
///
/// This function differs from :func:`~rustworkx.topological_sort` because
/// when there are ties between nodes in the sort order this function will
/// use the string returned by the ``key`` argument to determine the output
/// order used.  The ``reverse`` argument does not affect the ordering of keys
/// from this function, only the edges of the graph.
///
/// :param PyDiGraph dag: The DAG to get the topological sorted nodes from
/// :param callable key: key is a python function or other callable that
///     gets passed a single argument the node data from the graph and is
///     expected to return a string which will be used for resolving ties
///     in the sorting order.
/// :param bool reverse: If ``False`` (the default), perform a regular
///     topological ordering.  If ``True``, return the lexicographical
///     topological order that would have been found if all the edges in the
///     graph were reversed.  This does not affect the comparisons from the
///     ``key``.
/// :param Iterable[int] initial: If given, the initial node indices to start the topological
///     ordering from.  If not given, the topological ordering will certainly contain every node in
///     the graph.  If given, only the ``initial`` nodes and nodes that are dominated by the
///     ``initial`` set will be in the ordering.  Notably, any node that has a natural in degree of
///     zero will not be in the output ordering if ``initial`` is given and the zero-in-degree node
///     is not in it.  It is a :exc:`ValueError` to give an `initial` set where the nodes have even
///     a partial topological order between themselves.
///
/// :returns: A list of node's data lexicographically topologically sorted.
/// :rtype: list
#[pyfunction]
#[pyo3(signature = (dag, /, key, *, reverse=false, initial=None))]
pub fn lexicographical_topological_sort(
    py: Python,
    dag: &digraph::PyDiGraph,
    key: PyObject,
    reverse: bool,
    initial: Option<&Bound<PyAny>>,
) -> PyResult<PyObject> {
    let key_callable = |a: NodeIndex| -> PyResult<String> {
        let weight = &dag.graph[a];
        let res: String = key.call1(py, (weight,))?.extract(py)?;
        Ok(res)
    };
    let initial: Option<Vec<NodeIndex>> = match initial {
        Some(initial) => {
            let mut initial_vec: Vec<NodeIndex> = Vec::new();
            for maybe_index in initial.iter()? {
                let node = NodeIndex::new(maybe_index?.extract::<usize>()?);
                initial_vec.push(node);
            }
            Some(initial_vec)
        }
        None => None,
    };
    let out_list = core_lexico_topo_sort(&dag.graph, key_callable, reverse, initial.as_deref())?;
    match out_list {
        Some(out_list) => Ok(PyList::new_bound(
            py,
            out_list
                .into_iter()
                .map(|node| dag.graph[node].clone_ref(py)),
        )
        .into()),
        None => Err(PyValueError::new_err(
            "at least one initial node is reachable from another",
        )),
    }
}

/// Return the topological generations of a DAG
///
/// A topological generation is node collection in which ancestors of a node in each
/// generation are guaranteed to be in a previous generation, and any descendants of
/// a node are guaranteed to be in a following generation. Nodes are guaranteed to
/// be in the earliest possible generation that they can belong to.
///
/// :param PyDiGraph graph: The DAG to get the topological generations from
///
/// :returns: A list of topological generations.
/// :rtype: list
///
/// :raises DAGHasCycle: if a cycle is encountered while sorting the graph
#[pyfunction]
#[pyo3(text_signature = "(dag, /)")]
pub fn topological_generations(dag: &digraph::PyDiGraph) -> PyResult<Vec<NodeIndices>> {
    let mut in_degree_map: HashMap<NodeIndex, usize> = HashMap::new();
    let mut zero_in_degree: Vec<NodeIndex> = Vec::new();
    for node in dag.graph.node_indices() {
        let in_degree = dag.in_degree(node.index());
        if in_degree == 0 {
            zero_in_degree.push(node);
        } else {
            in_degree_map.insert(node, in_degree);
        }
    }

    let mut generations: Vec<NodeIndices> = Vec::new();
    let dir = petgraph::Direction::Outgoing;
    while !zero_in_degree.is_empty() {
        let this_generation = zero_in_degree.clone();
        zero_in_degree.clear();
        for node in this_generation.iter() {
            let neighbors = dag.graph.neighbors_directed(*node, dir);
            for child in neighbors {
                let child_degree = in_degree_map.get_mut(&child).unwrap();
                *child_degree -= 1;
                if *child_degree == 0 {
                    zero_in_degree.push(child);
                    in_degree_map.remove(&child);
                }
            }
        }
        generations.push(NodeIndices {
            nodes: this_generation.iter().map(|node| node.index()).collect(),
        });
    }

    // Check for cycle
    if !in_degree_map.is_empty() {
        return Err(DAGHasCycle::new_err("Topological sort encountered a cycle"));
    }
    Ok(generations)
}

/// Return the topological sort of node indices from the provided graph
///
/// :param PyDiGraph graph: The DAG to get the topological sort on
///
/// :returns: A list of node indices topologically sorted.
/// :rtype: NodeIndices
///
/// :raises DAGHasCycle: if a cycle is encountered while sorting the graph
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn topological_sort(graph: &digraph::PyDiGraph) -> PyResult<NodeIndices> {
    let nodes = match algo::toposort(&graph.graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => return Err(DAGHasCycle::new_err("Sort encountered a cycle")),
    };
    Ok(NodeIndices {
        nodes: nodes.iter().map(|node| node.index()).collect(),
    })
}

/// Collect runs that match a filter function
///
/// A run is a path of nodes where there is only a single successor and all
/// nodes in the path match the given condition. Each node in the graph can
/// appear in only a single run.
///
/// :param PyDiGraph graph: The graph to find runs in
/// :param filter_fn: The filter function to use for matching nodes. It takes
///     in one argument, the node data payload/weight object, and will return a
///     boolean whether the node matches the conditions or not. If it returns
///     ``False`` it will skip that node.
///
/// :returns: a list of runs, where each run is a list of node data
///     payload/weight for the nodes in the run
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, filter_fn)")]
pub fn collect_runs(
    py: Python,
    graph: &digraph::PyDiGraph,
    filter_fn: PyObject,
) -> PyResult<Vec<Vec<PyObject>>> {
    let mut out_list: Vec<Vec<PyObject>> = Vec::new();
    let mut seen: HashSet<NodeIndex> = HashSet::with_capacity(graph.node_count());

    let filter_node = |node: &PyObject| -> PyResult<bool> {
        let res = filter_fn.call1(py, (node,))?;
        res.extract(py)
    };

    let nodes = match algo::toposort(&graph.graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => return Err(DAGHasCycle::new_err("Sort encountered a cycle")),
    };
    for node in nodes {
        if !filter_node(&graph.graph[node])? || seen.contains(&node) {
            continue;
        }
        seen.insert(node);
        let mut group: Vec<PyObject> = vec![graph.graph[node].clone_ref(py)];
        let mut successors: Vec<NodeIndex> = graph
            .graph
            .neighbors_directed(node, petgraph::Direction::Outgoing)
            .collect();
        successors.dedup();

        while successors.len() == 1
            && filter_node(&graph.graph[successors[0]])?
            && !seen.contains(&successors[0])
        {
            group.push(graph.graph[successors[0]].clone_ref(py));
            seen.insert(successors[0]);
            successors = graph
                .graph
                .neighbors_directed(successors[0], petgraph::Direction::Outgoing)
                .collect();
            successors.dedup();
        }
        if !group.is_empty() {
            out_list.push(group);
        }
    }
    Ok(out_list)
}

/// Collect runs that match a filter function given edge colors.
///
/// A bicolor run is a list of group of nodes connected by edges of exactly
/// two colors. In addition, all nodes in the group must match the given
/// condition. Each node in the graph can appear in only a single group
/// in the bicolor run.
///
/// :param PyDiGraph graph: The graph to find runs in
/// :param filter_fn: The filter function to use for matching nodes. It takes
///     in one argument, the node data payload/weight object, and will return a
///     boolean whether the node matches the conditions or not.
///     If it returns ``True``, it will continue the bicolor chain.
///     If it returns ``False``, it will stop the bicolor chain.
///     If it returns ``None`` it will skip that node.
/// :param color_fn: The function that gives the color of the edge. It takes
///     in one argument, the edge data payload/weight object, and will
///     return a non-negative integer, the edge color. If the color is None,
///     the edge is ignored.
///
/// :returns: a list of groups with exactly two edge colors, where each group
///     is a list of node data payload/weight for the nodes in the bicolor run
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, filter_fn, color_fn)")]
pub fn collect_bicolor_runs(
    py: Python,
    graph: &digraph::PyDiGraph,
    filter_fn: PyObject,
    color_fn: PyObject,
) -> PyResult<Vec<Vec<PyObject>>> {
    let dag = &graph.graph;

    let filter_fn_wrapper = |node_index| -> Result<Option<bool>, PyErr> {
        let node_weight = dag.node_weight(node_index).expect("Invalid NodeId");
        let res = filter_fn.call1(py, (node_weight,))?;
        res.extract(py)
    };

    let color_fn_wrapper = |edge_index| -> Result<Option<usize>, PyErr> {
        let edge_weight = dag.edge_weight(edge_index).expect("Invalid EdgeId");
        let res = color_fn.call1(py, (edge_weight,))?;
        res.extract(py)
    };

    let block_list = match core_collect_bicolor_runs(dag, filter_fn_wrapper, color_fn_wrapper) {
        Ok(Some(block_list)) => block_list
            .into_iter()
            .map(|index_list| {
                index_list
                    .into_iter()
                    .map(|node_index| {
                        let node_weight = dag.node_weight(node_index).expect("Invalid NodeId");
                        node_weight.into_py(py)
                    })
                    .collect()
            })
            .collect(),
        Ok(None) => return Err(DAGHasCycle::new_err("The graph contains a cycle")),
        Err(e) => return Err(e),
    };

    Ok(block_list)
}

/// Returns the transitive reduction of a directed acyclic graph
///
/// The transitive reduction of :math:`G = (V,E)` is a graph :math:`G\prime = (V,E\prime)`
/// such that for all :math:`v` and :math:`w` in :math:`V` there is an edge :math:`(v, w)` in
/// :math:`E\prime` if and only if :math:`(v, w)` is in :math:`E`
/// and there is no path from :math:`v` to :math:`w` in :math:`G` with length greater than 1.
///
/// :param PyDiGraph graph: A directed acyclic graph
///
/// :returns: a directed acyclic graph representing the transitive reduction, and
///     a map containing the index of a node in the original graph mapped to its
///     equivalent in the resulting graph.
/// :rtype: Tuple[PyGraph, dict]
///
/// :raises PyValueError: if ``graph`` is not a DAG

#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn transitive_reduction(
    graph: &digraph::PyDiGraph,
    py: Python,
) -> PyResult<(digraph::PyDiGraph, DictMap<usize, usize>)> {
    let g = &graph.graph;
    let mut index_map = DictMap::with_capacity(g.node_count());
    if !is_directed_acyclic_graph(graph) {
        return Err(PyValueError::new_err(
            "Directed Acyclic Graph required for transitive_reduction",
        ));
    }
    let mut tr = StablePyGraph::<Directed>::with_capacity(g.node_count(), 0);
    let mut descendants = DictMap::new();
    let mut check_count = HashMap::with_capacity(g.node_count());

    for node in g.node_indices() {
        let i = node.index();
        index_map.insert(
            node,
            tr.add_node(graph.get_node_data(i).unwrap().clone_ref(py)),
        );
        check_count.insert(node, graph.in_degree(i));
    }

    for u in g.node_indices() {
        let mut u_nbrs: IndexSet<NodeIndex> = g.neighbors(u).collect();
        for v in g.neighbors(u) {
            if u_nbrs.contains(&v) {
                if !descendants.contains_key(&v) {
                    let dfs = dfs_edges(&g, Some(v));
                    descendants.insert(v, dfs);
                }
                for desc in &descendants[&v] {
                    u_nbrs.swap_remove(&NodeIndex::new(desc.1));
                }
            }
            *check_count.get_mut(&v).unwrap() -= 1;
            if check_count[&v] == 0 {
                descendants.swap_remove(&v);
            }
        }
        for v in u_nbrs {
            tr.add_edge(
                *index_map.get(&u).unwrap(),
                *index_map.get(&v).unwrap(),
                graph
                    .get_edge_data(u.index(), v.index())
                    .unwrap()
                    .clone_ref(py),
            );
        }
    }
    return Ok((
        digraph::PyDiGraph {
            graph: tr,
            node_removed: false,
            multigraph: graph.multigraph,
            attrs: py.None(),
            cycle_state: algo::DfsSpace::default(),
            check_cycle: graph.check_cycle,
        },
        index_map
            .iter()
            .map(|(k, v)| (k.index(), v.index()))
            .collect::<DictMap<usize, usize>>(),
    ));
}
