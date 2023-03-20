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
use petgraph::visit::{Control, Time};

use crate::PruneSearch;
use rustworkx_core::traversal::DfsEvent;

#[derive(FromPyObject)]
pub struct PyDfsVisitor {
    discover_vertex: PyObject,
    finish_vertex: PyObject,
    tree_edge: PyObject,
    back_edge: PyObject,
    forward_or_cross_edge: PyObject,
}

pub fn dfs_handler(
    py: Python,
    vis: &PyDfsVisitor,
    event: DfsEvent<NodeIndex, &PyObject>,
) -> PyResult<Control<()>> {
    let res = match event {
        DfsEvent::Discover(u, Time(t)) => vis.discover_vertex.call1(py, (u.index(), t)),
        DfsEvent::TreeEdge(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.tree_edge.call1(py, (edge,))
        }
        DfsEvent::BackEdge(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.back_edge.call1(py, (edge,))
        }
        DfsEvent::CrossForwardEdge(u, v, weight) => {
            let edge = (u.index(), v.index(), weight);
            vis.forward_or_cross_edge.call1(py, (edge,))
        }
        DfsEvent::Finish(u, Time(t)) => vis.finish_vertex.call1(py, (u.index(), t)),
    };

    match res {
        Err(e) => {
            if e.is_instance_of::<PruneSearch>(py) {
                Ok(Control::Prune)
            } else {
                Err(e)
            }
        }
        Ok(_) => Ok(Control::Continue),
    }
}
