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

use std::cmp::Ordering;

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::unionfind::UnionFind;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};

use crate::generators::pairwise;
use crate::graph;
use crate::shortest_path::all_pairs_dijkstra::all_pairs_dijkstra_shortest_paths;

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
    let node_count = graph.graph.node_count();
    if node_count == 0 {
        return Ok(Vec::new());
    }
    let mut out_vec = Vec::with_capacity(node_count * (node_count - 1) / 2);
    let mut distances = HashMap::with_capacity(graph.graph.node_count());
    let paths = all_pairs_dijkstra_shortest_paths(
        py,
        &graph.graph,
        weight_fn,
        Some(&mut distances),
    )?
    .paths;
    let mut nodes: HashSet<usize> =
        graph.graph.node_indices().map(|x| x.index()).collect();
    let first_node = graph
        .graph
        .node_indices()
        .map(|x| x.index())
        .next()
        .unwrap();
    let path_keys: HashSet<usize> =
        paths[&first_node].paths.keys().copied().collect();
    // first_node will always be missing from path_keys so if the difference
    // is > 1 with nodes that means there is another node in the graph that
    // first_node doesn't have a path to.
    if nodes.difference(&path_keys).count() > 1 {
        return Err(PyValueError::new_err(
            "The input graph must be a connected graph. The metric closure is \
            not defined for a graph with unconnected nodes",
        ));
    }
    // Iterate over node indices for a deterministic order
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

/// Return an approximation to the minimum Steiner tree of a graph.
///
/// The minimum tree of ``graph`` with regard to a set of ``terminal_nodes``
/// is a tree within ``graph`` that spans those nodes and has a minimum size
/// (measured as the sum of edge weights) amoung all such trees.
///
/// The minimum steiner tree can be approximated by computing the minimum
/// spanning tree of the subgraph of the metric closure of ``graph`` induced
/// by the terminal nodes, where the metric closure of ``graph`` is the
/// complete graph in which each edge is weighted by the shortest path distance
/// between nodes in ``graph``.
///
/// This algorithm [1]_ produces a tree whose weight is within a
/// :math:`(2 - (2 / t))` factor of the weight of the optimal Steiner tree
/// where :math:`t` is the number of terminal nodes.
///
/// :param PyGraph graph: The graph to compute the minimum Steiner tree for
/// :param list terminal_nodes: The list of node indices for which the Steiner
///     tree is to be computed for.
/// :param weight_fn: A callable object that will be passed an edge's
///     weight/data payload and expected to return a ``float``. For example,
///     you can use ``weight_fn=float`` to cast every weight as a float.
///
/// :returns: An approximation to the minimal steiner tree of ``graph`` induced
///     by ``terminal_nodes``.
/// :rtype: PyGraph
///
/// .. [1] Kou, Markowsky & Berman,
///    "A fast algorithm for Steiner trees"
///    Acta Informatica 15, 141–145 (1981).
///    https://link.springer.com/article/10.1007/BF00288961
#[pyfunction]
#[pyo3(text_signature = "(graph, terminal_nodes, weight_fn, /)")]
pub fn steiner_tree(
    py: Python,
    graph: &graph::PyGraph,
    terminal_nodes: Vec<usize>,
    weight_fn: PyObject,
) -> PyResult<graph::PyGraph> {
    let terminal_node_set: HashSet<usize> =
        terminal_nodes.into_iter().collect();
    let metric_edges =
        _metric_closure_edges(py, graph, weight_fn.clone_ref(py))?;
    // Calculate mst edges from metric closure edge list:
    let mut subgraphs = UnionFind::<usize>::new(graph.graph.node_bound());
    let mut edge_list: Vec<MetricClosureEdge> =
        Vec::with_capacity(metric_edges.len());
    for edge in metric_edges {
        if !terminal_node_set.contains(&edge.source)
            || !terminal_node_set.contains(&edge.target)
        {
            continue;
        }
        let weight = edge.distance;
        if weight.is_nan() {
            return Err(PyValueError::new_err("NaN found as an edge weight"));
        }
        edge_list.push(edge);
    }
    edge_list.par_sort_unstable_by(|a, b| {
        let weight_a = (a.distance, a.source, a.target);
        let weight_b = (b.distance, b.source, b.target);
        weight_a.partial_cmp(&weight_b).unwrap_or(Ordering::Less)
    });
    let mut mst_edges: Vec<MetricClosureEdge> = Vec::new();
    for float_edge_pair in edge_list {
        let u = float_edge_pair.source;
        let v = float_edge_pair.target;
        if subgraphs.union(u, v) {
            mst_edges.push(float_edge_pair);
        }
    }
    // Generate the output graph from the MST of the metric closure
    let out_edge_list: Vec<[usize; 2]> = mst_edges
        .iter()
        .map(|edge| pairwise(edge.path.clone()))
        .flatten()
        .filter_map(|x| x.0.map(|a| [a, x.1]))
        .collect();
    let out_edges: HashSet<(usize, usize)> =
        out_edge_list.iter().map(|x| (x[0], x[1])).collect();
    let mut out_graph = graph.clone();
    let out_nodes: HashSet<NodeIndex> = out_edge_list
        .iter()
        .map(|x| x.iter())
        .flatten()
        .copied()
        .map(NodeIndex::new)
        .collect();
    for node in graph
        .graph
        .node_indices()
        .filter(|node| !out_nodes.contains(node))
    {
        out_graph.graph.remove_node(node);
        out_graph.node_removed = true;
    }
    for edge in graph.graph.edge_references().filter(|edge| {
        let source = edge.source().index();
        let target = edge.target().index();
        !out_edges.contains(&(source, target))
            && !out_edges.contains(&(target, source))
    }) {
        out_graph.graph.remove_edge(edge.id());
    }
    // Deduplicate potential duplicate edges
    deduplicate_edges(py, &mut out_graph, &weight_fn)?;
    Ok(out_graph)
}

fn deduplicate_edges(
    py: Python,
    out_graph: &mut graph::PyGraph,
    weight_fn: &PyObject,
) -> PyResult<()> {
    if out_graph.multigraph {
        // Find all edges between nodes
        let mut duplicate_map: HashMap<
            [NodeIndex; 2],
            Vec<(EdgeIndex, PyObject)>,
        > = HashMap::new();
        for edge in out_graph.graph.edge_references() {
            if duplicate_map.contains_key(&[edge.source(), edge.target()]) {
                duplicate_map
                    .get_mut(&[edge.source(), edge.target()])
                    .unwrap()
                    .push((edge.id(), edge.weight().clone_ref(py)));
            } else if duplicate_map
                .contains_key(&[edge.target(), edge.source()])
            {
                duplicate_map
                    .get_mut(&[edge.target(), edge.source()])
                    .unwrap()
                    .push((edge.id(), edge.weight().clone_ref(py)));
            } else {
                duplicate_map.insert(
                    [edge.source(), edge.target()],
                    vec![(edge.id(), edge.weight().clone_ref(py))],
                );
            }
        }
        // For a node pair with > 1 edge find minimum edge and remove others
        for edges_raw in duplicate_map.values().filter(|x| x.len() > 1) {
            let mut edges: Vec<(EdgeIndex, f64)> =
                Vec::with_capacity(edges_raw.len());
            for edge in edges_raw {
                let res = weight_fn.call1(py, (&edge.1,))?;
                let raw = res.to_object(py);
                let weight = raw.extract(py)?;
                edges.push((edge.0, weight));
            }
            edges.par_sort_unstable_by(|a, b| {
                let weight_a = a.1;
                let weight_b = b.1;
                weight_a.partial_cmp(&weight_b).unwrap_or(Ordering::Less)
            });
            edges[1..].iter().for_each(|x| {
                out_graph.graph.remove_edge(x.0);
            });
        }
    }
    Ok(())
}
