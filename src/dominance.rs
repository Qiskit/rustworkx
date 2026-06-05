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

use super::{InvalidNode, NullGraph, digraph};
use rustworkx_core::dictmap::DictMap;

use hashbrown::HashSet;

use petgraph::algo::dominators;
use petgraph::graph::NodeIndex;

use pyo3::prelude::*;

/// Determine the immediate dominators of all nodes in a directed graph.
///
/// The dominance computation uses the algorithm published in 2006 by
/// Cooper, Harvey, and Kennedy (https://hdl.handle.net/1911/96345).
/// The time complexity is quadratic in the number of vertices.
///
/// :param PyDiGraph graph: directed graph
/// :param int start_node: the start node for the dominance computation
///
/// :returns: a mapping of node indices to their immediate dominators
/// :rtype: dict[int, int]
///
/// :raises NullGraph: the passed graph is empty
/// :raises InvalidNode: the start node is not in the graph
#[pyfunction]
#[pyo3(text_signature = "(graph, start_node, /)")]
pub fn immediate_dominators(
    graph: &digraph::PyDiGraph,
    start_node: usize,
) -> PyResult<DictMap<usize, usize>> {
    if graph.graph.node_count() == 0 {
        return Err(NullGraph::new_err("Invalid operation on a NullGraph"));
    }

    let start_node_index = NodeIndex::new(start_node);

    if !graph.graph.contains_node(start_node_index) {
        return Err(InvalidNode::new_err("Start node is not in the graph"));
    }

    let dom = dominators::simple_fast(&graph.graph, start_node_index);

    // Include the root node to match networkx.immediate_dominators
    let root_dom = [(start_node, start_node)];
    let others_dom = graph.graph.node_indices().filter_map(|index| {
        dom.immediate_dominator(index)
            .map(|res| (index.index(), res.index()))
    });
    Ok(root_dom.into_iter().chain(others_dom).collect())
}

/// Compute the dominance frontiers of all nodes in a directed graph.
///
/// The dominance and dominance frontiers computations use the
/// algorithms published in 2006 by Cooper, Harvey, and Kennedy
/// (https://hdl.handle.net/1911/96345).
///
/// :param PyDiGraph graph: directed graph
/// :param int start_node: the start node for the dominance computation
///
/// :returns: a mapping of node indices to their dominance frontiers
/// :rtype: dict[int, set[int]]
///
/// :raises NullGraph: the passed graph is empty
/// :raises InvalidNode: the start node is not in the graph
#[pyfunction]
#[pyo3(text_signature = "(graph, start_node, /)")]
pub fn dominance_frontiers(
    graph: &digraph::PyDiGraph,
    start_node: usize,
) -> PyResult<DictMap<usize, HashSet<usize>>> {
    let idom = immediate_dominators(graph, start_node)?;

    let mut df: DictMap<_, _> = idom
        .iter()
        .map(|(&node, _)| (node, HashSet::default()))
        .collect();

    for (&node, &node_idom) in &idom {
        let preds = graph.predecessor_indices(node);
        if preds.nodes.len() >= 2 {
            for mut runner in preds.nodes {
                while runner != node_idom {
                    df.entry(runner)
                        .and_modify(|e| {
                            e.insert(node);
                        })
                        .or_insert([node].into_iter().collect());
                    if let Some(&runner_idom) = idom.get(&runner) {
                        runner = runner_idom;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    Ok(df)
}
