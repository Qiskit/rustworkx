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

use super::{digraph, graph};
use hashbrown::HashSet;

use pyo3::prelude::*;

use petgraph::graph::NodeIndex;
use rayon::prelude::*;

fn _graph_triangles(graph: &graph::PyGraph, node: usize) -> (usize, usize) {
    let mut triangles: usize = 0;

    let index = NodeIndex::new(node);
    let mut neighbors: HashSet<NodeIndex> =
        graph.graph.neighbors(index).collect();
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
fn graph_transitivity(graph: &graph::PyGraph) -> f64 {
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

fn _digraph_triangles(
    graph: &digraph::PyDiGraph,
    node: usize,
) -> (usize, usize) {
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

    let din: usize = in_neighbors.len();
    let dout: usize = out_neighbors.len();

    let dtot = dout + din;
    let dbil: usize = out_neighbors.intersection(&in_neighbors).count();
    let triples: usize = match dtot {
        0 => 0,
        _ => dtot * (dtot - 1) - 2 * dbil,
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
fn digraph_transitivity(graph: &digraph::PyDiGraph) -> f64 {
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
