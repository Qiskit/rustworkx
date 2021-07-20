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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;

use crate::_all_pairs_dijkstra_shortest_paths;
use crate::graph;

#[derive(Debug)]
struct MetricClosureEdge {
    source: usize,
    target: usize,
    distance: f64,
    path: Vec<usize>,
}

/// Return the metric closure of a graph
///
/// The metric closure of a graph is the complete graph in which each edge is
/// weighted by the shortest path distance between the nodes in the graph.
///
/// :param PyGraph graph: The input graph to find the metric closure for
/// :param weight_fn: A callable object that will be passed an edge's
///     weight/data payload and expected to return a ``float``. For example,
///     you can use ``weight_fn=float`` to cast every weight as a float
///
/// :return: A metric closure graph from the input graph
/// :rtype: PyGraph
#[pyfunction]
#[pyo3(text_signature = "(graph, weight_fn, /)")]
pub fn metric_closure(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: PyObject,
) -> PyResult<graph::PyGraph> {
    let mut out_graph = graph.clone();
    out_graph.graph.clear_edges();
    let edges = _metric_closure_edges(py, graph, weight_fn)?;
    for edge in edges {
        out_graph.graph.add_edge(
            NodeIndex::new(edge.source),
            NodeIndex::new(edge.target),
            (edge.distance, edge.path).to_object(py),
        );
    }
    Ok(out_graph)
}

fn _metric_closure_edges(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: PyObject,
) -> PyResult<Vec<MetricClosureEdge>> {
    let mut out_vec = Vec::with_capacity(graph.graph.edge_count());
    let mut distances = HashMap::with_capacity(graph.graph.node_count());
    let paths = _all_pairs_dijkstra_shortest_paths(
        py,
        &graph.graph,
        weight_fn,
        Some(&mut distances),
    )?
    .paths;
    let path_keys: HashSet<usize> = paths
        .keys()
        .filter(|x| !paths[x].paths.is_empty())
        .copied()
        .collect();
    let mut nodes: HashSet<usize> =
        graph.graph.node_indices().map(|x| x.index()).collect();
    if nodes.difference(&path_keys).count() > 0 {
        return Err(PyValueError::new_err(
            "The input graph must be a connected graph. The metric closure is \
            not defined for a graph with unconnected nodes",
        ));
    }
    for node in graph.graph.node_indices().map(|x| x.index()) {
        let path_map = &paths[&node].paths;
        nodes.remove(&node);
        let distance = &distances[&node];
        for v in &nodes {
            let v_index = NodeIndex::new(*v);
            out_vec.push(MetricClosureEdge {
                source: node,
                target: *v,
                distance: distance[&v_index],
                path: path_map[v].clone(),
            });
        }
    }
    Ok(out_vec)
}
