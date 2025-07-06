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

use super::{digraph, graph};
use hashbrown::HashSet;

use crate::digraph::PyDiGraph;
use petgraph::algo::kosaraju_scc;
use petgraph::algo::DfsSpace;
use petgraph::graph::DiGraph;
use pyo3::prelude::*;

use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use petgraph::visit::NodeCount;
use petgraph::graph::NodeIndex;
use rayon::prelude::*;

use rustworkx_core::traversal::build_transitive_closure_dag;

fn _graph_triangles(graph: &graph::PyGraph, node: usize) -> (usize, usize) {
    let mut triangles: usize = 0;

    let index = NodeIndex::new(node);
    let mut neighbors: HashSet<NodeIndex> = graph.graph.neighbors(index).collect();
    neighbors.remove(&index);

    for nodev in &neighbors {
        triangles += graph
            .graph
            .neighbors(*nodev)
            .filter(|&x| (x != *nodev) && neighbors.contains(&x))
            .count();
    }

    let d: usize = neighbors.len();
    let triples: usize = match d {
        0 => 0,
        _ => (d * (d - 1)) / 2,
    };

    (triangles / 2, triples)
}

/// Compute the transitivity of an undirected graph.
///
/// The transitivity of a graph is defined as:
///
/// .. math::
///     `c=3 \times \frac{\text{number of triangles}}{\text{number of connected triples}}`
///
/// A “connected triple” means a single vertex with
/// edges running to an unordered pair of others.
///
/// This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyGraph graph: Graph to be used.
///
/// :returns: Transitivity.
/// :rtype: float
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn graph_transitivity(graph: &graph::PyGraph) -> f64 {
    let node_indices: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let (triangles, triples) = node_indices
        .par_iter()
        .map(|node| _graph_triangles(graph, node.index()))
        .reduce(
            || (0, 0),
            |(sumx, sumy), (resx, resy)| (sumx + resx, sumy + resy),
        );

    match triangles {
        0 => 0.0,
        _ => triangles as f64 / triples as f64,
    }
}

fn _digraph_triangles(graph: &digraph::PyDiGraph, node: usize) -> (usize, usize) {
    let mut triangles: usize = 0;

    let index = NodeIndex::new(node);
    let mut out_neighbors: HashSet<NodeIndex> = graph
        .graph
        .neighbors_directed(index, petgraph::Direction::Outgoing)
        .collect();
    out_neighbors.remove(&index);

    let mut in_neighbors: HashSet<NodeIndex> = graph
        .graph
        .neighbors_directed(index, petgraph::Direction::Incoming)
        .collect();
    in_neighbors.remove(&index);

    let neighbors = out_neighbors.iter().chain(in_neighbors.iter());

    for nodev in neighbors {
        triangles += graph
            .graph
            .neighbors_directed(*nodev, petgraph::Direction::Outgoing)
            .chain(
                graph
                    .graph
                    .neighbors_directed(*nodev, petgraph::Direction::Incoming),
            )
            .map(|x| {
                let mut res: usize = 0;

                if (x != *nodev) && out_neighbors.contains(&x) {
                    res += 1;
                }
                if (x != *nodev) && in_neighbors.contains(&x) {
                    res += 1;
                }
                res
            })
            .sum::<usize>();
    }

    let d_in: usize = in_neighbors.len();
    let d_out: usize = out_neighbors.len();

    let d_tot = d_out + d_in;
    let d_bil: usize = out_neighbors.intersection(&in_neighbors).count();
    let triples: usize = match d_tot {
        0 => 0,
        _ => d_tot * (d_tot - 1) - 2 * d_bil,
    };

    (triangles / 2, triples)
}

/// Compute the transitivity of a directed graph.
///
/// The transitivity of a directed graph is defined in [Fag]_, Eq.8:
///
/// .. math::
///     `c=3 \times \frac{\text{number of triangles}}{\text{number of all possible triangles}}`
///
/// A triangle is a connected triple of nodes.
/// Different edge orientations counts as different triangles.
///
/// This function is multithreaded and will run
/// launch a thread pool with threads equal to the number of CPUs by default.
/// You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would
/// limit the thread pool to 4 threads.
///
/// .. note::
///
///     The function implicitly assumes that there are no parallel edges
///     or self loops. It may produce incorrect/unexpected results if the
///     input graph has self loops or parallel edges.
///
/// :param PyDiGraph graph: Directed graph to be used.
///
/// :returns: Transitivity.
/// :rtype: float
///
/// .. [Fag] Clustering in complex directed networks by G. Fagiolo,
///    Physical Review E, 76(2), 026107 (2007)
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn digraph_transitivity(graph: &digraph::PyDiGraph) -> f64 {
    let node_indices: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let (triangles, triples) = node_indices
        .par_iter()
        .map(|node| _digraph_triangles(graph, node.index()))
        .reduce(
            || (0, 0),
            |(sumx, sumy), (resx, resy)| (sumx + resx, sumy + resy),
        );

    match triangles {
        0 => 0.0,
        _ => triangles as f64 / triples as f64,
    }
}

/// Returns the transitive closure of a graph
#[pyfunction]
#[pyo3(text_signature = "(graph, /")]
pub fn transitive_closure(py: Python, graph: &PyDiGraph) -> PyResult<PyDiGraph> {
    let sccs = kosaraju_scc(&graph.graph);

    let mut condensed_graph = DiGraph::new();
    let mut scc_nodes = Vec::new();
    let mut scc_map: Vec<NodeIndex> = vec![NodeIndex::end(); graph.node_count()];

    for scc in &sccs {
        let scc_node = condensed_graph.add_node(());
        scc_nodes.push(scc_node);
        for node in scc {
            scc_map[node.index()] = scc_node;
        }
    }
    for edge in graph.graph.edge_references() {
        let (source, target) = (edge.source(), edge.target());

        if scc_map[source.index()] != scc_map[target.index()] {
            condensed_graph.add_edge(scc_map[source.index()], scc_map[target.index()], ());
        }
    }

    let closure_graph_result = build_transitive_closure_dag(condensed_graph, None, || {});
    let out_graph = closure_graph_result.unwrap();

    let mut new_graph = graph.graph.clone();
    new_graph.clear();

    let mut result_map: Vec<NodeIndex> = vec![NodeIndex::end(); out_graph.node_count()];
    for (_index, node) in out_graph.node_indices().enumerate() {
        let result_node = new_graph.add_node(py.None());
        result_map[node.index()] = result_node;
    }
    for edge in out_graph.edge_references() {
        let (source, target) = (edge.source(), edge.target());
        new_graph.add_edge(
            result_map[source.index()],
            result_map[target.index()],
            py.None(),
        );
    }
    let out = PyDiGraph {
        graph: new_graph,
        cycle_state: DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
        attrs: py.None(),
    };

    Ok(out)
}
