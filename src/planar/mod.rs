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

use crate::graph::PyGraph;
use rustworkx_core::planar;

use pyo3::prelude::*;

/// Check if an undirected graph is planar.
///
/// A graph is planar iff it can be drawn in a plane without any edge
/// intersections. The planarity check algorithm is based on the
/// Left-Right Planarity Test [Brandes]_.
///
/// :param PyGraph graph: The graph to be used.
///
/// :returns: Whether the provided graph is planar.
/// :rtype: bool
///
/// .. [Brandes] Ulrik Brandes:
///    The Left-Right Planarity Test
///    2009
///    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.217.9208
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
pub fn is_planar(graph: &PyGraph) -> bool {
    planar::is_planar(&graph.graph)
}
