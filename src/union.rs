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

use crate::digraph::PyDiGraph;
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
pub fn digraph_union(
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
    };
    let mut node_map = HashMap::new();
    let mut edge_map = HashSet::new();

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
