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

#![allow(clippy::too_many_arguments)]
// This module was originally forked from petgraph's isomorphism module @ v0.5.0
// to handle PyDiGraph inputs instead of petgraph's generic Graph. However it has
// since diverged significantly from the original petgraph implementation.

use std::cmp::Ordering;
use std::iter::Iterator;

use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::PyTraverseError;

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::EdgeType;
use petgraph::{Directed, Outgoing, Undirected};

use crate::iterators::NodeMap;
use crate::StablePyGraph;

use rustworkx_core::isomorphism::vf2;

struct PyMatcher(Option<PyObject>);

impl<Ty: EdgeType> vf2::NodeMatcher<StablePyGraph<Ty>, StablePyGraph<Ty>> for PyMatcher {
    type Error = PyErr;

    fn enabled(&self) -> bool {
        self.0.is_some()
    }

    fn eq(
        &mut self,
        g0: &StablePyGraph<Ty>,
        g1: &StablePyGraph<Ty>,
        n0: NodeIndex,
        n1: NodeIndex,
    ) -> Result<bool, Self::Error> {
        if let (Some(a), Some(b)) = (g0.node_weight(n0), g1.node_weight(n1)) {
            unsafe {
                // Note: we can assume this since we'll have the GIL whenever we're
                // accessing the (Di|)GraphVF2Mapping pyclass methods.
                let py = Python::assume_gil_acquired();
                let res = self.0.as_ref().unwrap().call1(py, (a, b))?;
                res.is_truthy(py)
            }
        } else {
            Ok(false)
        }
    }
}

impl<Ty: EdgeType> vf2::EdgeMatcher<StablePyGraph<Ty>, StablePyGraph<Ty>> for PyMatcher {
    type Error = PyErr;

    fn enabled(&self) -> bool {
        self.0.is_some()
    }

    fn eq(
        &mut self,
        g0: &StablePyGraph<Ty>,
        g1: &StablePyGraph<Ty>,
        e0: (NodeIndex, NodeIndex),
        e1: (NodeIndex, NodeIndex),
    ) -> Result<bool, Self::Error> {
        let w0 = g0
            .edges_directed(e0.0, Outgoing)
            .find(|edge| edge.target() == e0.1)
            .and_then(|edge| g0.edge_weight(edge.id()));
        let w1 = g1
            .edges_directed(e1.0, Outgoing)
            .find(|edge| edge.target() == e1.1)
            .and_then(|edge| g1.edge_weight(edge.id()));
        if let (Some(a), Some(b)) = (w0, w1) {
            unsafe {
                // Note: we can assume this since we'll have the GIL whenever we're
                // accessing the (Di|)GraphVF2Mapping pyclass methods.
                let py = Python::assume_gil_acquired();
                let res = self.0.as_ref().unwrap().call1(py, (a, b))?;
                res.is_truthy(py)
            }
        } else {
            Ok(false)
        }
    }
}

macro_rules! vf2_mapping_impl {
    ($name:ident, $Ty:ty) => {
        #[pyclass(module = "rustworkx")]
        pub struct $name {
            vf2: vf2::Vf2Algorithm<StablePyGraph<$Ty>, StablePyGraph<$Ty>, PyMatcher, PyMatcher>,
        }

        impl $name {
            pub fn new(
                _py: Python,
                g0: &StablePyGraph<$Ty>,
                g1: &StablePyGraph<$Ty>,
                node_match: Option<PyObject>,
                edge_match: Option<PyObject>,
                id_order: bool,
                ordering: Ordering,
                induced: bool,
                call_limit: Option<usize>,
            ) -> Self {
                let vf2 = vf2::Vf2Algorithm::new(
                    g0,
                    g1,
                    PyMatcher(node_match),
                    PyMatcher(edge_match),
                    id_order,
                    ordering,
                    induced,
                    call_limit,
                );
                $name { vf2 }
            }
        }

        #[pymethods]
        impl $name {
            fn __iter__(slf: PyRef<Self>) -> Py<$name> {
                slf.into()
            }

            fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<NodeMap>> {
                Python::with_gil(|_py| match slf.vf2.next() {
                    Some(mapping) => Ok(Some(NodeMap {
                        node_map: mapping.map_err(|e| match e {
                            vf2::IsIsomorphicError::NodeMatcherErr(e) => e,
                            vf2::IsIsomorphicError::EdgeMatcherErr(e) => e,
                        })?,
                    })),
                    None => Ok(None),
                })
            }

            fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
                for node in self.vf2.st.0.graph.node_weights() {
                    visit.call(node)?;
                }
                for edge in self.vf2.st.0.graph.edge_weights() {
                    visit.call(edge)?;
                }
                for node in self.vf2.st.1.graph.node_weights() {
                    visit.call(node)?;
                }
                for edge in self.vf2.st.1.graph.edge_weights() {
                    visit.call(edge)?;
                }
                if let Some(ref obj) = self.vf2.node_match.0 {
                    visit.call(obj)?;
                }
                if let Some(ref obj) = self.vf2.edge_match.0 {
                    visit.call(obj)?;
                }
                Ok(())
            }

            fn __clear__(&mut self) {
                self.vf2.st.0.graph = StablePyGraph::<$Ty>::default();
                self.vf2.st.0.graph = StablePyGraph::<$Ty>::default();
                self.vf2.node_match.0 = None;
                self.vf2.edge_match.0 = None;
            }
        }
    };
}

vf2_mapping_impl!(DiGraphVf2Mapping, Directed);
vf2_mapping_impl!(GraphVf2Mapping, Undirected);
