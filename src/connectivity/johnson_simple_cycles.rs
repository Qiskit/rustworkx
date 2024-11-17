// Licensed under the apache license, version 2.0 (the "license"); you may
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

use crate::digraph::PyDiGraph;
use petgraph::graph::NodeIndex;
use std::ops::Deref;

use pyo3::prelude::*;

use crate::iterators::NodeIndices;
use rustworkx_core::connectivity::{johnson_simple_cycles, SimpleCycleIter};

#[pyclass(module = "rustworkx")]
pub struct PySimpleCycleIter {
    graph_clone: Py<PyDiGraph>,
    iter: SimpleCycleIter,
}

impl PySimpleCycleIter {
    pub fn new(py: Python, graph: Bound<PyDiGraph>) -> PyResult<Self> {
        // For compatibility with networkx manually insert self cycles and filter
        // from Johnson's algorithm
        let self_cycles_vec: Vec<NodeIndex> = graph
            .borrow()
            .graph
            .node_indices()
            .filter(|n| graph.borrow().graph.neighbors(*n).any(|x| x == *n))
            .collect();
        if self_cycles_vec.is_empty() {
            let iter = johnson_simple_cycles(&graph.borrow().graph, None);
            let out_graph = graph.unbind();
            Ok(PySimpleCycleIter {
                graph_clone: out_graph,
                iter,
            })
        } else {
            // Copy graph to remove self edges before running johnson's algorithm
            let mut graph_clone = graph.borrow().copy();
            for node in &self_cycles_vec {
                while let Some(edge_index) = graph_clone.graph.find_edge(*node, *node) {
                    graph_clone.graph.remove_edge(edge_index);
                }
            }
            let self_cycles = if self_cycles_vec.is_empty() {
                None
            } else {
                Some(self_cycles_vec)
            };
            let iter = johnson_simple_cycles(&graph_clone.graph, self_cycles);
            let out_graph = Py::new(py, graph_clone)?;
            Ok(PySimpleCycleIter {
                graph_clone: out_graph,
                iter,
            })
        }
    }
}

#[pymethods]
impl PySimpleCycleIter {
    fn __iter__(slf: PyRef<Self>) -> Py<PySimpleCycleIter> {
        slf.into()
    }

    fn __next__(mut slf: PyRefMut<Self>, py: Python) -> PyResult<Option<NodeIndices>> {
        let py_clone = slf.graph_clone.clone_ref(py);
        let binding = py_clone.borrow(py);
        let graph = binding.deref();
        let res: Option<Vec<NodeIndex>> = slf.iter.next(&graph.graph);
        Ok(res.map(|cycle| NodeIndices {
            nodes: cycle.into_iter().map(|x| x.index()).collect(),
        }))
    }
}
