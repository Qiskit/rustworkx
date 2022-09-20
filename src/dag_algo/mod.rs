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

mod longest_path;

use hashbrown::{HashMap, HashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::iterators::NodeIndices;
use crate::{digraph, DAGHasCycle, InvalidNode};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::NodeCount;

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
        nodes: longest_path::longest_path(graph, edge_weight_callable)?.0,
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
    let (_, path_weight) = longest_path::longest_path(graph, edge_weight_callable)?;
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
        nodes: longest_path::longest_path(graph, edge_weight_callable)?.0,
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
    let (_, path_weight) = longest_path::longest_path(graph, edge_weight_callable)?;
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
#[pyfunction(index_output = "false")]
#[pyo3(text_signature = "(dag, first_layer, /, index_output=False)")]
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
        Ok(PyList::new(py, output).into())
    } else {
        Ok(PyList::new(py, output_indices).into())
    }
}

/// Get the lexicographical topological sorted nodes from the provided DAG
///
/// This function returns a list of nodes data in a graph lexicographically
/// topologically sorted using the provided key function. A topological sort
/// is a linear ordering of vertices such that for every directed edge from
/// node :math:`u` to node :math:`v`, :math:`u` comes before :math:`v`
/// in the ordering.
///
/// This function differs from :func:`~rustworkx.topological_sort` because
/// when there are ties between nodes in the sort order this function will
/// use the string returned by the ``key`` argument to determine the output
/// order used.
///
/// :param PyDiGraph dag: The DAG to get the topological sorted nodes from
/// :param callable key: key is a python function or other callable that
///     gets passed a single argument the node data from the graph and is
///     expected to return a string which will be used for resolving ties
///     in the sorting order.
///
/// :returns: A list of node's data lexicographically topologically sorted.
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(dag, key, /)")]
pub fn lexicographical_topological_sort(
    py: Python,
    dag: &digraph::PyDiGraph,
    key: PyObject,
) -> PyResult<PyObject> {
    let key_callable = |a: &PyObject| -> PyResult<PyObject> {
        let res = key.call1(py, (a,))?;
        Ok(res.to_object(py))
    };
    // HashMap of node_index indegree
    let node_count = dag.node_count();
    let mut in_degree_map: HashMap<NodeIndex, usize> = HashMap::with_capacity(node_count);
    for node in dag.graph.node_indices() {
        in_degree_map.insert(node, dag.in_degree(node.index()));
    }

    #[derive(Clone, Eq, PartialEq)]
    struct State {
        key: String,
        node: NodeIndex,
    }

    impl Ord for State {
        fn cmp(&self, other: &State) -> Ordering {
            // Notice that the we flip the ordering on costs.
            // In case of a tie we compare positions - this step is necessary
            // to make implementations of `PartialEq` and `Ord` consistent.
            other
                .key
                .cmp(&self.key)
                .then_with(|| other.node.index().cmp(&self.node.index()))
        }
    }

    // `PartialOrd` needs to be implemented as well.
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &State) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut zero_indegree = BinaryHeap::with_capacity(node_count);
    for (node, degree) in in_degree_map.iter() {
        if *degree == 0 {
            let map_key_raw = key_callable(&dag.graph[*node])?;
            let map_key: String = map_key_raw.extract(py)?;
            zero_indegree.push(State {
                key: map_key,
                node: *node,
            });
        }
    }
    let mut out_list: Vec<&PyObject> = Vec::with_capacity(node_count);
    let dir = petgraph::Direction::Outgoing;
    while let Some(State { node, .. }) = zero_indegree.pop() {
        let neighbors = dag.graph.neighbors_directed(node, dir);
        for child in neighbors {
            let child_degree = in_degree_map.get_mut(&child).unwrap();
            *child_degree -= 1;
            if *child_degree == 0 {
                let map_key_raw = key_callable(&dag.graph[child])?;
                let map_key: String = map_key_raw.extract(py)?;
                zero_indegree.push(State {
                    key: map_key,
                    node: child,
                });
                in_degree_map.remove(&child);
            }
        }
        out_list.push(&dag.graph[node])
    }
    Ok(PyList::new(py, out_list).into())
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
#[pyo3(text_signature = "(graph, filter)")]
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

/// Collect runs that match a filter function given edge colors
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
    let mut pending_list: Vec<Vec<PyObject>> = Vec::new();
    let mut block_id: Vec<Option<usize>> = Vec::new();
    let mut block_list: Vec<Vec<PyObject>> = Vec::new();

    let filter_node = |node: &PyObject| -> PyResult<Option<bool>> {
        let res = filter_fn.call1(py, (node,))?;
        res.extract(py)
    };

    let color_edge = |edge: &PyObject| -> PyResult<Option<usize>> {
        let res = color_fn.call1(py, (edge,))?;
        res.extract(py)
    };

    let nodes = match algo::toposort(&graph.graph, None) {
        Ok(nodes) => nodes,
        Err(_err) => return Err(DAGHasCycle::new_err("Sort encountered a cycle")),
    };

    // Utility for ensuring pending_list has the color index
    macro_rules! ensure_vector_has_index {
        ($pending_list: expr, $block_id: expr, $color: expr) => {
            if $color >= $pending_list.len() {
                $pending_list.resize($color + 1, Vec::new());
                $block_id.resize($color + 1, None);
            }
        };
    }

    for node in nodes {
        if let Some(is_match) = filter_node(&graph.graph[node])? {
            let raw_edges = graph
                .graph
                .edges_directed(node, petgraph::Direction::Outgoing);

            // Remove all edges that do not yield errors from color_fn
            let colors = raw_edges
                .map(|edge| {
                    let edge_weight = edge.weight();
                    color_edge(edge_weight)
                })
                .collect::<PyResult<Vec<Option<usize>>>>()?;

            // Remove null edges from color_fn
            let colors = colors.into_iter().flatten().collect::<Vec<usize>>();

            if colors.len() <= 2 && is_match {
                if colors.len() == 1 {
                    let c0 = colors[0];
                    ensure_vector_has_index!(pending_list, block_id, c0);
                    if let Some(c0_block_id) = block_id[c0] {
                        block_list[c0_block_id].push(graph.graph[node].clone_ref(py));
                    } else {
                        pending_list[c0].push(graph.graph[node].clone_ref(py));
                    }
                } else if colors.len() == 2 {
                    let c0 = colors[0];
                    let c1 = colors[1];
                    ensure_vector_has_index!(pending_list, block_id, c0);
                    ensure_vector_has_index!(pending_list, block_id, c1);

                    if block_id[c0].is_some()
                        && block_id[c1].is_some()
                        && block_id[c0] == block_id[c1]
                    {
                        block_list[block_id[c0].unwrap_or_default()]
                            .push(graph.graph[node].clone_ref(py));
                    } else {
                        let mut new_block: Vec<PyObject> =
                            Vec::with_capacity(pending_list[c0].len() + pending_list[c1].len() + 1);

                        // Clears pending lits and add to new block
                        new_block.append(&mut pending_list[c0]);
                        new_block.append(&mut pending_list[c1]);

                        new_block.push(graph.graph[node].clone_ref(py));

                        // Create new block, assign its id to color pair
                        block_id[c0] = Some(block_list.len());
                        block_id[c1] = Some(block_list.len());
                        block_list.push(new_block);
                    }
                }
            } else {
                for color in colors {
                    let color = color;
                    ensure_vector_has_index!(pending_list, block_id, color);
                    if let Some(color_block_id) = block_id[color] {
                        block_list[color_block_id].append(&mut pending_list[color]);
                    }
                    block_id[color] = None;
                    pending_list[color].clear();
                }
            }
        }
    }

    Ok(block_list)
}
