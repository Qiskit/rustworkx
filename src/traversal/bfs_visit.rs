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

use crate::PruneSearch;
use retworkx_core::traversal::BfsEvent;

#[derive(FromPyObject)]
pub struct PyBfsVisitor {
    discover_vertex: PyObject,
    finish_vertex: PyObject,
    tree_edge: PyObject,
    non_tree_edge: PyObject,
    gray_target_edge: PyObject,
    black_target_edge: PyObject,
}

pub fn bfs_handler(
    py: Python,
    vis: &PyBfsVisitor,
    event: BfsEvent<NodeIndex, &PyObject>,
) -> PyResult<Control<()>> {
    let res = match event {
        BfsEvent::Discover(u) => vis.discover_vertex.call1(py, (u.index(),)),
        BfsEvent::TreeEdge(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.tree_edge.call1(py, (edge,))
        }
        BfsEvent::NonTreeEdge(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.non_tree_edge.call1(py, (edge,))
        }
        BfsEvent::GrayTargetEdge(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.gray_target_edge.call1(py, (edge,))
        }
        BfsEvent::BlackTargetEdge(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.black_target_edge.call1(py, (edge,))
        }
        BfsEvent::Finish(u) => vis.finish_vertex.call1(py, (u.index(),)),
    };

    match res {
        Err(e) => {
            if e.is_instance::<PruneSearch>(py) {
                Ok(Control::Prune)
            } else {
                Err(e)
            }
        }
        Ok(_) => Ok(Control::Continue),
    }
}
