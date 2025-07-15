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

use pyo3::prelude::*;

use rustworkx_core::transitivity as core_transitivity;

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
/// This function is multithreaded and will
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
    core_transitivity::graph_transitivity(&graph.graph)
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
/// This function is multithreaded and will
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
    core_transitivity::digraph_transitivity(&graph.graph)
}
