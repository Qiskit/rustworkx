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

use hashbrown::HashSet;

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{IntoEdgeReferences, NodeFiltered};

use pyo3::prelude::*;
use pyo3::Python;

use rustworkx_core::dense_subgraph::densest_subgraph;
use rustworkx_core::dictmap::*;

use crate::digraph;
use crate::graph;
use crate::iterators::NodeMap;
use crate::StablePyGraph;

/// Find densest subgraph in a :class:`~.PyGraph`
///
/// This method does not provide any guarantees on the approximation as it
/// does a naive search using BFS traversal.
///
/// :param PyGraph graph: The graph to find densest subgraph in.
/// :param int num_nodes: The number of nodes in the subgraph to find
/// :param func edge_weight_callback: An optional callable that if specified will be
///     passed the node indices of each edge in the graph and it is expected to
///     return a float value. If specified the lowest avg weight for edges in
///     a found subgraph will be a criteria for selection in addition to the
///     connectivity of the subgraph.
/// :param func node_weight_callback: An optional callable that if specified will be
///     passed the node indices of each node in the graph and it is expected to
///     return a float value. If specified the lowest avg weight for node of
///     a found subgraph will be a criteria for selection in addition to the
///     connectivity of the subgraph.//
/// :returns: A tuple of the subgraph found and a :class:`~.NodeMap` of the
///     mapping of node indices in the input ``graph`` to the index in the
///     output subgraph.
/// :rtype: (PyGraph, NodeMap)
#[pyfunction]
#[pyo3(
    signature=(graph, num_nodes, /, edge_weight_callback=None, node_weight_callback=None)
)]
pub fn graph_densest_subgraph_of_size(
    py: Python,
    graph: &graph::PyGraph,
    num_nodes: usize,
    edge_weight_callback: Option<PyObject>,
    node_weight_callback: Option<PyObject>,
) -> PyResult<(graph::PyGraph, NodeMap)> {
    let edge_callback = edge_weight_callback.map(|callback| {
        move |edge: <&StablePyGraph<Undirected> as IntoEdgeReferences>::EdgeRef| -> PyResult<f64> {
            let res = callback
                .bind(py)
                .call1((edge.source().index(), edge.target().index()))?;
            res.extract()
        }
    });
    let node_callback = node_weight_callback.map(|callback| {
        move |node_index: NodeIndex| -> PyResult<f64> {
            let res = callback.bind(py).call1((node_index.index(),))?;
            res.extract()
        }
    });

    let subgraph_nodes = densest_subgraph(&graph.graph, num_nodes, edge_callback, node_callback)?;
    let node_subset: HashSet<NodeIndex> = subgraph_nodes.iter().copied().collect();
    let node_filter = |node: NodeIndex| -> bool { node_subset.contains(&node) };
    let filtered = NodeFiltered(&graph.graph, node_filter);
    let mut inner_graph: StablePyGraph<Undirected> =
        StablePyGraph::with_capacity(subgraph_nodes.len(), 0);
    let node_map: DictMap<usize, usize> = subgraph_nodes
        .into_iter()
        .map(|node| {
            (
                node.index(),
                inner_graph
                    .add_node(graph.graph.node_weight(node).unwrap().clone_ref(py))
                    .index(),
            )
        })
        .collect();
    for edge in filtered.edge_references() {
        let new_source = NodeIndex::new(*node_map.get(&edge.source().index()).unwrap());
        let new_target = NodeIndex::new(*node_map.get(&edge.target().index()).unwrap());
        inner_graph.add_edge(new_source, new_target, edge.weight().clone_ref(py));
    }
    let out_graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
        multigraph: graph.multigraph,
        attrs: py.None(),
    };
    Ok((out_graph, NodeMap { node_map }))
}

/// Find densest subgraph in a :class:`~.PyDiGraph`
///
/// This method does not provide any guarantees on the approximation as it
/// does a naive search using BFS traversal.
///
/// :param PyDiGraph graph: The graph to find the densest subgraph in.
/// :param int num_nodes: The number of nodes in the subgraph to find
/// :param func edge_weight_callback: An optional callable that if specified will be
///     passed the node indices of each edge in the graph and it is expected to
///     return a float value. If specified the lowest avg weight for edges in
///     a found subgraph will be a criteria for selection in addition to the
///     connectivity of the subgraph.
/// :param func node_weight_callback: An optional callable that if specified will be
///     passed the node indices of each node in the graph and it is expected to
///     return a float value. If specified the lowest avg weight for node of
///     a found subgraph will be a criteria for selection in addition to the
///     connectivity of the subgraph.
/// :returns: A tuple of the subgraph found and a :class:`~.NodeMap` of the
///     mapping of node indices in the input ``graph`` to the index in the
///     output subgraph.
/// :rtype: (PyDiGraph, NodeMap)
#[pyfunction]
#[pyo3(
    signature = (graph, num_nodes, /, edge_weight_callback=None, node_weight_callback=None)
)]
pub fn digraph_densest_subgraph_of_size(
    py: Python,
    graph: &digraph::PyDiGraph,
    num_nodes: usize,
    edge_weight_callback: Option<PyObject>,
    node_weight_callback: Option<PyObject>,
) -> PyResult<(digraph::PyDiGraph, NodeMap)> {
    let edge_callback = edge_weight_callback.map(|callback| {
        move |edge: <&StablePyGraph<Directed> as IntoEdgeReferences>::EdgeRef| -> PyResult<f64> {
            let res = callback
                .bind(py)
                .call1((edge.source().index(), edge.target().index()))?;
            res.extract()
        }
    });
    let node_callback = node_weight_callback.map(|callback| {
        move |node_index: NodeIndex| -> PyResult<f64> {
            let res = callback.bind(py).call1((node_index.index(),))?;
            res.extract()
        }
    });

    let subgraph_nodes = densest_subgraph(&graph.graph, num_nodes, edge_callback, node_callback)?;
    let node_subset: HashSet<NodeIndex> = subgraph_nodes.iter().copied().collect();
    let node_filter = |node: NodeIndex| -> bool { node_subset.contains(&node) };
    let filtered = NodeFiltered(&graph.graph, node_filter);
    let mut inner_graph: StablePyGraph<Directed> =
        StablePyGraph::with_capacity(subgraph_nodes.len(), 0);
    let node_map: DictMap<usize, usize> = subgraph_nodes
        .into_iter()
        .map(|node| {
            (
                node.index(),
                inner_graph
                    .add_node(graph.graph.node_weight(node).unwrap().clone_ref(py))
                    .index(),
            )
        })
        .collect();
    for edge in filtered.edge_references() {
        let new_source = NodeIndex::new(*node_map.get(&edge.source().index()).unwrap());
        let new_target = NodeIndex::new(*node_map.get(&edge.target().index()).unwrap());
        inner_graph.add_edge(new_source, new_target, edge.weight().clone_ref(py));
    }
    let out_graph = digraph::PyDiGraph {
        graph: inner_graph,
        node_removed: false,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: graph.check_cycle,
        multigraph: graph.multigraph,
        attrs: py.None(),
    };
    Ok((out_graph, NodeMap { node_map }))
}
