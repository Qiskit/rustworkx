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

use crate::{digraph, digraph::PyDiGraph};
use hashbrown::{HashMap, HashSet};
use petgraph::algo;
use petgraph::graph::EdgeIndex;
use pyo3::prelude::*;
use pyo3::Python;
use std::cmp::Ordering;

/// [Graph] Return a new PyDiGraph by forming a union from`a` and `b` graphs.
///
/// The algorithm has three phases:
///  - adds all nodes from `b` to `a`. operates in O(n), n being number of nodes in `b`.
///  - merges nodes from `b` over `a` given that:
///     - `merge_nodes` is `true`. operates in O(n^2), n being number of nodes in `b`.
///     - respective node in`b` and `a` share the same weight
///  - adds all edges from `b` to `a`.
///     - `merge_edges` is `true`
///     - respective edge in`b` and `a` share the same weight
///
/// with the same weight in graphs `a` and `b` and merged those nodes.
///
/// The nodes from graph `b` will replace nodes from `a`.
///
///  At this point, only `PyDiGraph` is supported.
fn _digraph_union(
    py: Python,
    a: &PyDiGraph,
    b: &PyDiGraph,
    merge_nodes: bool,
    merge_edges: bool,
) -> PyResult<PyDiGraph> {
    let first = &a.graph;
    let second = &b.graph;
    let mut combined = PyDiGraph {
        graph: first.clone(),
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
    };
    let mut node_map = HashMap::with_capacity(second.node_count());
    let mut edge_map = HashSet::with_capacity(second.edge_count());

    let compare_weights = |a: &PyAny, b: &PyAny| -> PyResult<bool> {
        let res = a.compare(b)?;
        Ok(res == Ordering::Equal)
    };

    for node in second.node_indices() {
        let node_index = combined.add_node(second[node].clone_ref(py))?;
        node_map.insert(node.index(), node_index);
    }

    for edge in b.weighted_edge_list(py).edges {
        let source = edge.0;
        let target = edge.1;
        let edge_weight = edge.2;

        let new_source = *node_map.get(&source).unwrap();
        let new_target = *node_map.get(&target).unwrap();

        let edge_index = combined.add_edge(
            new_source,
            new_target,
            edge_weight.clone_ref(py),
        )?;

        let edge_node = EdgeIndex::new(edge_index);

        if combined.has_edge(source, target) {
            let w = combined.graph.edge_weight(edge_node).unwrap();
            if compare_weights(edge_weight.as_ref(py), w.as_ref(py)).unwrap() {
                edge_map.insert(edge_node);
            }
        }
    }

    if merge_nodes {
        for node in second.node_indices() {
            let weight = &second[node].clone_ref(py);
            let index = a.find_node_by_weight(py, weight.clone_ref(py));

            if index.is_some() {
                let other_node = node_map.get(&node.index());
                combined.merge_nodes(
                    py,
                    *other_node.unwrap(),
                    index.unwrap(),
                )?;
            }
        }
    }

    if merge_edges {
        for edge in edge_map {
            combined.graph.remove_edge(edge);
        }
    }

    Ok(combined)
}

/// Return a new PyDiGraph by forming a union from two input PyDiGraph objects
///
/// The algorithm in this function operates in three phases:
///
///  1. Add all the nodes from  ``second`` into ``first``. operates in O(n),
///     with n being number of nodes in `b`.
///  2. Merge nodes from ``second`` over ``first`` given that:
///
///     - The ``merge_nodes`` is ``True``. operates in O(n^2), with n being the
///       number of nodes in ``second``.
///     - The respective node in ``second`` and ``first`` share the same
///       weight/data payload.
///
///  3. Adds all the edges from ``second`` to ``first``. If the ``merge_edges``
///     parameter is ``True`` and the respective edge in ``second`` and
///     first`` share the same weight/data payload they will be merged
///     together.
///
///  :param PyDiGraph first: The first directed graph object
///  :param PyDiGraph second: The second directed graph object
///  :param bool merge_nodes: If set to ``True`` nodes will be merged between
///     ``second`` and ``first`` if the weights are equal.
///  :param bool merge_edges: If set to ``True`` edges will be merged between
///     ``second`` and ``first`` if the weights are equal.
///
///  :returns: A new PyDiGraph object that is the union of ``second`` and
///     ``first``. It's worth noting the weight/data payload objects are
///     passed by reference from ``first`` and ``second`` to this new object.
///  :rtype: PyDiGraph
#[pyfunction]
#[pyo3(text_signature = "(first, second, merge_nodes, merge_edges, /)")]
fn digraph_union(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    merge_nodes: bool,
    merge_edges: bool,
) -> PyResult<digraph::PyDiGraph> {
    let res = _digraph_union(py, first, second, merge_nodes, merge_edges)?;
    Ok(res)
}
