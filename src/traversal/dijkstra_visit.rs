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

use pyo3::prelude::*;

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::Control;

use rustworkx_core::traversal::DijkstraEvent;

#[derive(FromPyObject)]
pub struct PyDijkstraVisitor {
    discover_vertex: Py<PyAny>,
    finish_vertex: Py<PyAny>,
    examine_edge: Py<PyAny>,
    edge_relaxed: Py<PyAny>,
    edge_not_relaxed: Py<PyAny>,
}

pub fn dijkstra_handler(
    py: Python,
    vis: &PyDijkstraVisitor,
    event: DijkstraEvent<NodeIndex, &Py<PyAny>, f64>,
) -> PyResult<Control<()>> {
    let res = match event {
        DijkstraEvent::Discover(u, score) => vis.discover_vertex.call1(py, (u.index(), score)),
        DijkstraEvent::ExamineEdge(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.examine_edge.call1(py, (edge,))
        }
        DijkstraEvent::EdgeRelaxed(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.edge_relaxed.call1(py, (edge,))
        }
        DijkstraEvent::EdgeNotRelaxed(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.edge_not_relaxed.call1(py, (edge,))
        }
        DijkstraEvent::Finish(u) => vis.finish_vertex.call1(py, (u.index(),)),
    };

    super::handle_visitor_result(py, res, || matches!(event, DijkstraEvent::Finish(..)), true)
}
