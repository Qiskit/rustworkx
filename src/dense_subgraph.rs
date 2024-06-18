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

use hashbrown::{HashMap, HashSet};

use petgraph::algo;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::{IntoEdgeReferences, NodeFiltered};
use petgraph::EdgeType;

use rayon::prelude::*;

use pyo3::prelude::*;
use pyo3::Python;

use rustworkx_core::dictmap::*;

use crate::digraph;
use crate::graph;
use crate::iterators::NodeMap;
use crate::StablePyGraph;

struct SubsetResult {
    pub count: usize,
    pub error: f64,
    pub map: Vec<NodeIndex>,
    pub subgraph: Vec<[NodeIndex; 2]>,
}

pub fn densest_subgraph<Ty>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    num_nodes: usize,
    edge_weight_callback: Option<PyObject>,
    node_weight_callback: Option<PyObject>,
) -> PyResult<(StablePyGraph<Ty>, NodeMap)>
where
    Ty: EdgeType + Sync,
{
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    let float_callback =
        |callback: &PyObject, source_node: usize, target_node: usize| -> PyResult<f64> {
            let res = callback.bind(py).call1((source_node, target_node))?;
            res.extract()
        };
    let node_callback = |callback: &PyObject, node_index: usize| -> PyResult<f64> {
        let res = callback.bind(py).call1((node_index,))?;
        res.extract()
    };
    let mut edge_weight_map: Option<HashMap<[NodeIndex; 2], f64>> = None;
    let mut node_weight_map: Option<HashMap<NodeIndex, f64>> = None;

    if edge_weight_callback.is_some() {
        let mut inner_weight_map: HashMap<[NodeIndex; 2], f64> =
            HashMap::with_capacity(graph.edge_count());
        let callback = edge_weight_callback.as_ref().unwrap();
        for edge in graph.edge_references() {
            let source: NodeIndex = edge.source();
            let target: NodeIndex = edge.target();
            let weight = float_callback(callback, source.index(), target.index())?;
            inner_weight_map.insert([source, target], weight);
            if !graph.is_directed() {
                inner_weight_map.insert([target, source], weight);
            }
        }
        edge_weight_map = Some(inner_weight_map);
    }
    let mut avg_node_error: f64 = 0.;
    if node_weight_callback.is_some() {
        let callback = node_weight_callback.as_ref().unwrap();
        let mut inner_weight_map: HashMap<NodeIndex, f64> =
            HashMap::with_capacity(graph.node_count());
        for node in graph.node_indices() {
            let node_index = node.index();
            let weight = node_callback(callback, node_index)?;
            avg_node_error += weight;
            inner_weight_map.insert(node, weight);
        }
        avg_node_error /= graph.node_count() as f64;
        node_weight_map = Some(inner_weight_map);
    }
    let reduce_identity_fn = || -> SubsetResult {
        SubsetResult {
            count: 0,
            map: Vec::new(),
            error: f64::INFINITY,
            subgraph: Vec::new(),
        }
    };

    let reduce_fn = |best: SubsetResult, curr: SubsetResult| -> SubsetResult {
        if edge_weight_callback.is_some() || node_weight_callback.is_some() {
            if curr.count >= best.count && curr.error <= best.error {
                curr
            } else {
                best
            }
        } else if curr.count > best.count {
            curr
        } else {
            best
        }
    };

    let best_result = node_indices
        .into_par_iter()
        .filter_map(|index| {
            let mut subgraph: Vec<[NodeIndex; 2]> = Vec::with_capacity(num_nodes);
            let mut bfs = Bfs::new(&graph, index);
            let mut bfs_vec: Vec<NodeIndex> = Vec::with_capacity(num_nodes);
            let mut bfs_set: HashSet<NodeIndex> = HashSet::with_capacity(num_nodes);

            let mut count = 0;
            while let Some(node) = bfs.next(&graph) {
                bfs_vec.push(node);
                bfs_set.insert(node);
                count += 1;
                if count >= num_nodes {
                    break;
                }
            }
            if bfs_vec.len() < num_nodes {
                return None;
            }
            let mut connection_count = 0;
            for node in &bfs_vec {
                for nbr in graph.neighbors(*node).filter(|j| bfs_set.contains(j)) {
                    connection_count += 1;
                    subgraph.push([*node, nbr]);
                }
            }
            let mut error = match &edge_weight_map {
                Some(map) => {
                    subgraph.iter().map(|edge| map[edge]).sum::<f64>() / subgraph.len() as f64
                }
                None => 0.,
            };
            error *= match &node_weight_map {
                Some(map) => {
                    let subgraph_node_error_avg =
                        bfs_vec.iter().map(|node| map[node]).sum::<f64>() / num_nodes as f64;
                    let node_error_diff = subgraph_node_error_avg - avg_node_error;
                    if node_error_diff > 0. {
                        num_nodes as f64 * node_error_diff
                    } else {
                        1.
                    }
                }
                None => 1.,
            };

            Some(SubsetResult {
                count: connection_count,
                error,
                map: bfs_vec,
                subgraph,
            })
        })
        .reduce(reduce_identity_fn, reduce_fn);

    let mut subgraph = StablePyGraph::<Ty>::with_capacity(num_nodes, best_result.subgraph.len());
    let mut node_map: DictMap<usize, usize> = DictMap::with_capacity(num_nodes);
    for node in best_result.map {
        let new_index = subgraph.add_node(graph[node].clone_ref(py));
        node_map.insert(node.index(), new_index.index());
    }
    let node_filter = |node: NodeIndex| -> bool { node_map.contains_key(&node.index()) };
    let filtered = NodeFiltered(graph, node_filter);
    for edge in filtered.edge_references() {
        let new_source = NodeIndex::new(*node_map.get(&edge.source().index()).unwrap());
        let new_target = NodeIndex::new(*node_map.get(&edge.target().index()).unwrap());
        subgraph.add_edge(new_source, new_target, edge.weight().clone_ref(py));
    }
    Ok((subgraph, NodeMap { node_map }))
}

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
    text_signature = "(graph, num_nodes, /, edge_weight_callback=None, node_weight_callback=None)"
)]
pub fn graph_densest_subgraph_of_size(
    py: Python,
    graph: &graph::PyGraph,
    num_nodes: usize,
    edge_weight_callback: Option<PyObject>,
    node_weight_callback: Option<PyObject>,
) -> PyResult<(graph::PyGraph, NodeMap)> {
    let (inner_graph, node_map) = densest_subgraph(
        py,
        &graph.graph,
        num_nodes,
        edge_weight_callback,
        node_weight_callback,
    )?;
    let out_graph = graph::PyGraph {
        graph: inner_graph,
        node_removed: false,
        multigraph: graph.multigraph,
        attrs: py.None(),
    };
    Ok((out_graph, node_map))
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
    text_signature = "(graph, num_nodes, /, edge_weight_callback=None, node_weight_callback=None)"
)]
pub fn digraph_densest_subgraph_of_size(
    py: Python,
    graph: &digraph::PyDiGraph,
    num_nodes: usize,
    edge_weight_callback: Option<PyObject>,
    node_weight_callback: Option<PyObject>,
) -> PyResult<(digraph::PyDiGraph, NodeMap)> {
    let (inner_graph, node_map) = densest_subgraph(
        py,
        &graph.graph,
        num_nodes,
        edge_weight_callback,
        node_weight_callback,
    )?;
    let out_graph = digraph::PyDiGraph {
        graph: inner_graph,
        node_removed: false,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: graph.check_cycle,
        multigraph: graph.multigraph,
        attrs: py.None(),
    };
    Ok((out_graph, node_map))
}
