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

use super::{graph, weight_callable};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Python;

use petgraph::prelude::*;
use petgraph::stable_graph::EdgeReference;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{IntoEdgeReferences, NodeIndexable};

use rayon::prelude::*;

use crate::iterators::WeightedEdgeList;

/// Find the edges in the minimum spanning tree or forest of an undirected graph
/// using Kruskal's algorithm.
///
/// A Minimum Spanning Tree (MST) is a subset of the edges of a connected,
/// undirected graph that connects all vertices together without cycles and
/// with the minimum possible total edge weight.
///
/// This function computes the edges that form the MST or forest of an
/// undirected graph. Kruskal's algorithm works by sorting all the edges in the
/// graph by their weights and then adding them one by one to the MST, ensuring
/// that no cycles are formed.
///
///     >>> G = rx.PyGraph()
///     >>> G.add_nodes_from(["A", "B", "C", "D"])
///     NodeIndices[0, 1, 2, 3]
///     >>> G.add_edges_from([(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)])
///     EdgeIndices[0, 1, 2, 3, 4]
///     >>> rx.minimum_spanning_edges(G, weight_fn=lambda x: x)
///     WeightedEdgeList[(2, 3, 4), (0, 3, 5), (0, 1, 10)]
///
/// In this example, the edge `(0, 2, 6)` won't become part of the MST because
/// in the moment it's considered, the two other edges connecting nodes
/// 0-3-2 are already parts of MST because of their lower weight.
///
/// To obtain the result as a graph, see :func:`~minimum_spanning_tree`.
///
/// :param PyGraph graph: An undirected graph
/// :param weight_fn: A callable object (function, lambda, etc) that takes
///     an edge object and returns a ``float``. This function is used to
///     extract the numerical weight for each edge. For example:
///
///         rx.minimum_spanning_edges(G, weight_fn=lambda x: 1)
///
///     will assign a weight of 1 to all edges and thus ignore the real weights.
///
///         rx.minimum_spanning_edges(G, weight_fn=float)
///
///     will just cast the edge object to a ``float`` to determine its weight.
/// :param float default_weight: If ``weight_fn`` isn't specified, this optional
///     float value will be used for the weight/cost of each edge.
///
/// :returns: The :math:`N - |c|` edges of the Minimum Spanning Tree (or Forest,
///     if :math:`|c| > 1`) where :math:`N` is the number of nodes and
///     :math:`|c|` is the number of connected components of the graph.
/// :rtype: WeightedEdgeList
///
/// :raises ValueError: If a NaN value is found (or computed) as an edge weight.
#[pyfunction]
#[pyo3(signature=(graph, weight_fn=None, default_weight=1.0), text_signature = "(graph, weight_fn=None, default_weight=1.0)")]
pub fn minimum_spanning_edges(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<WeightedEdgeList> {
    let mut subgraphs = UnionFind::<usize>::new(graph.graph.node_bound());

    let mut edge_list: Vec<(f64, EdgeReference<PyObject>)> =
        Vec::with_capacity(graph.graph.edge_count());
    for edge in graph.graph.edge_references() {
        let weight = weight_callable(py, &weight_fn, edge.weight(), default_weight)?;
        if weight.is_nan() {
            return Err(PyValueError::new_err("NaN found as an edge weight"));
        }
        edge_list.push((weight, edge));
    }

    edge_list.par_sort_unstable_by(|(weight_a, _), (weight_b, _)| {
        weight_a.partial_cmp(weight_b).unwrap_or(Ordering::Less)
    });

    let mut mst_edges: Vec<(usize, usize, PyObject)> = Vec::new();
    for (_, edge) in edge_list.iter() {
        let u = edge.source().index();
        let v = edge.target().index();
        if subgraphs.union(u, v) {
            mst_edges.push((u, v, edge.weight().clone_ref(py)));
        }
    }

    Ok(WeightedEdgeList { edges: mst_edges })
}

/// Find the minimum spanning tree or forest of an undirected graph using
/// Kruskal's algorithm.
///
/// A Minimum Spanning Tree (MST) is a subset of the edges of a connected,
/// undirected graph that connects all vertices together without cycles and
/// with the minimum possible total edge weight.
///
/// This function computes the edges that form the MST or forest of an
/// undirected graph. Kruskal's algorithm works by sorting all the edges in the
/// graph by their weights and then adding them one by one to the MST, ensuring
/// that no cycles are formed.
///
///     >>> G = rx.PyGraph()
///     >>> G.add_nodes_from(["A", "B", "C", "D"])
///     NodeIndices[0, 1, 2, 3]
///     >>> G.add_edges_from([(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)])
///     EdgeIndices[0, 1, 2, 3, 4]
///     >>> mst_G = rx.minimum_spanning_tree(G, weight_fn=lambda x: x)
///     >>> mst_G.weighted_edge_list()
///     WeightedEdgeList[(2, 3, 4), (0, 3, 5), (0, 1, 10)]
///
/// In this example, the edge `(0, 2, 6)` won't become part of the MST because
/// in the moment it's considered, the two other edges connecting nodes
/// 0-3-2 are already parts of MST because of their lower weight.
///
/// To obtain the result just as a list of edges, see :func:`~minimum_spanning_edges`.
///
/// :param PyGraph graph: An undirected graph
/// :param weight_fn: A callable object (function, lambda, etc) that takes
///     an edge object and returns a ``float``. This function is used to
///     extract the numerical weight for each edge. For example:
///
///         rx.minimum_spanning_tree(G, weight_fn=lambda x: 1)
///
///     will assign a weight of 1 to all edges and thus ignore the real weights.
///
///         rx.minimum_spanning_tree(G, weight_fn=float)
///
///     will just cast the edge object to a ``float`` to determine its weight.
/// :param float default_weight: If ``weight_fn`` isn't specified, this optional
///     float value will be used for the weight/cost of each edge.
///
/// :returns: A Minimum Spanning Tree (or Forest, if the graph is not connected).
/// :rtype: PyGraph
///
/// .. note::
///
///     The new graph will keep the same node indices, but edge indices might differ.
///
/// :raises ValueError: If a NaN value is found (or computed) as an edge weight.
#[pyfunction]
#[pyo3(signature=(graph, weight_fn=None, default_weight=1.0), text_signature = "(graph, weight_fn=None, default_weight=1.0)")]
pub fn minimum_spanning_tree(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<PyObject>,
    default_weight: f64,
) -> PyResult<graph::PyGraph> {
    let mut spanning_tree = (*graph).clone();
    spanning_tree.graph.clear_edges();

    for &(u, v, ref weight) in minimum_spanning_edges(py, graph, weight_fn, default_weight)?
        .edges
        .iter()
    {
        spanning_tree.add_edge(u, v, weight.clone_ref(py))?;
    }

    Ok(spanning_tree)
}
