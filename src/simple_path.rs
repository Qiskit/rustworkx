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

use super::{digraph, graph, InvalidNode};

use pyo3::prelude::*;

use petgraph::algo;
use petgraph::graph::NodeIndex;

/// Return all simple paths between 2 nodes in a PyGraph object
///
/// A simple path is a path with no repeated nodes.
///
/// :param PyGraph graph: The graph to find the path in
/// :param int from: The node index to find the paths from
/// :param int to: The node index to find the paths to
/// :param int min_depth: The minimum depth of the path to include in the output
///     list of paths. By default all paths are included regardless of depth,
///     setting to 0 will behave like the default.
/// :param int cutoff: The maximum depth of path to include in the output list
///     of paths. By default includes all paths regardless of depth, setting to
///     0 will behave like default.
///
/// :returns: A list of lists where each inner list is a path of node indices
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, from, to, /, min=None, cutoff=None)")]
fn graph_all_simple_paths(
    graph: &graph::PyGraph,
    from: usize,
    to: usize,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let from_index = NodeIndex::new(from);
    if !graph.graph.contains_node(from_index) {
        return Err(InvalidNode::new_err(
            "The input index for 'from' is not a valid node index",
        ));
    }
    let to_index = NodeIndex::new(to);
    if !graph.graph.contains_node(to_index) {
        return Err(InvalidNode::new_err(
            "The input index for 'to' is not a valid node index",
        ));
    }
    let min_intermediate_nodes: usize = match min_depth {
        Some(depth) => depth - 2,
        None => 0,
    };
    let cutoff_petgraph: Option<usize> = cutoff.map(|depth| depth - 2);
    let result: Vec<Vec<usize>> = algo::all_simple_paths(
        graph,
        from_index,
        to_index,
        min_intermediate_nodes,
        cutoff_petgraph,
    )
    .map(|v: Vec<NodeIndex>| v.into_iter().map(|i| i.index()).collect())
    .collect();
    Ok(result)
}

/// Return all simple paths between 2 nodes in a PyDiGraph object
///
/// A simple path is a path with no repeated nodes.
///
/// :param PyDiGraph graph: The graph to find the path in
/// :param int from: The node index to find the paths from
/// :param int to: The node index to find the paths to
/// :param int min_depth: The minimum depth of the path to include in the output
///     list of paths. By default all paths are included regardless of depth,
///     sett to 0 will behave like the default.
/// :param int cutoff: The maximum depth of path to include in the output list
///     of paths. By default includes all paths regardless of depth, setting to
///     0 will behave like default.
///
/// :returns: A list of lists where each inner list is a path
/// :rtype: list
#[pyfunction]
#[pyo3(text_signature = "(graph, from, to, /, min_depth=None, cutoff=None)")]
fn digraph_all_simple_paths(
    graph: &digraph::PyDiGraph,
    from: usize,
    to: usize,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let from_index = NodeIndex::new(from);
    if !graph.graph.contains_node(from_index) {
        return Err(InvalidNode::new_err(
            "The input index for 'from' is not a valid node index",
        ));
    }
    let to_index = NodeIndex::new(to);
    if !graph.graph.contains_node(to_index) {
        return Err(InvalidNode::new_err(
            "The input index for 'to' is not a valid node index",
        ));
    }
    let min_intermediate_nodes: usize = match min_depth {
        Some(depth) => depth - 2,
        None => 0,
    };
    let cutoff_petgraph: Option<usize> = cutoff.map(|depth| depth - 2);
    let result: Vec<Vec<usize>> = algo::all_simple_paths(
        graph,
        from_index,
        to_index,
        min_intermediate_nodes,
        cutoff_petgraph,
    )
    .map(|v: Vec<NodeIndex>| v.into_iter().map(|i| i.index()).collect())
    .collect();
    Ok(result)
}
