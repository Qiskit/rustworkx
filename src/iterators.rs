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

#![allow(clippy::upper_case_acronyms)]

use std::collections::hash_map::DefaultHasher;
use std::convert::TryInto;
use std::hash::Hasher;

use pyo3::class::{PyObjectProtocol, PySequenceProtocol};
use pyo3::exceptions::{PyIndexError, PyNotImplementedError};
use pyo3::gc::{PyGCProtocol, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PySequence;
use pyo3::PyTraverseError;

/// A custom class for the return from :func:`retworkx.bfs_successors`
///
/// This class is a container class for the results of the
/// :func:`retworkx.bfs_successors` function. It implements the Python
/// sequence protocol. So you can treat the return as read-only
/// sequence/list that is integer indexed. If you want to use it as an
/// iterator you can by wrapping it in an ``iter()`` that will yield the
/// results in order.
///
/// For example::
///
///     import retworkx
///
///     graph = retworkx.generators.directed_path_graph(5)
///     bfs_succ = retworkx.bfs_successors(0)
///     # Index based access
///     third_element = bfs_succ[2]
///     # Use as iterator
///     bfs_iter = iter(bfs_succ)
///     first_element = next(bfs_iter)
///     second_element = nex(bfs_iter)
///
#[pyclass(module = "retworkx", gc)]
pub struct BFSSuccessors {
    pub bfs_successors: Vec<(PyObject, Vec<PyObject>)>,
}

#[pymethods]
impl BFSSuccessors {
    #[new]
    fn new() -> Self {
        BFSSuccessors {
            bfs_successors: Vec::new(),
        }
    }

    fn __getstate__(&self) -> Vec<(PyObject, Vec<PyObject>)> {
        self.bfs_successors.clone()
    }

    fn __setstate__(&mut self, state: Vec<(PyObject, Vec<PyObject>)>) {
        self.bfs_successors = state;
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for BFSSuccessors {
    fn __richcmp__(
        &self,
        other: &'p PySequence,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<bool> {
        let compare = |other: &PySequence| -> PyResult<bool> {
            if other.len()? as usize != self.bfs_successors.len() {
                return Ok(false);
            }
            let gil = Python::acquire_gil();
            let py = gil.python();
            for i in 0..self.bfs_successors.len() {
                let other_raw = other.get_item(i.try_into().unwrap())?;
                let other_value: (PyObject, Vec<PyObject>) =
                    other_raw.extract()?;
                if self.bfs_successors[i].0.as_ref(py).compare(other_value.0)?
                    != std::cmp::Ordering::Equal
                {
                    return Ok(false);
                }
                for (index, obj) in self.bfs_successors[i].1.iter().enumerate()
                {
                    if obj.as_ref(py).compare(&other_value.1[index])?
                        != std::cmp::Ordering::Equal
                    {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        };
        match op {
            pyo3::basic::CompareOp::Eq => compare(other),
            pyo3::basic::CompareOp::Ne => match compare(other) {
                Ok(res) => Ok(!res),
                Err(err) => Err(err),
            },
            _ => Err(PyNotImplementedError::new_err(
                "Comparison not implemented",
            )),
        }
    }

    fn __str__(&self) -> PyResult<String> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let mut str_vec: Vec<String> =
            Vec::with_capacity(self.bfs_successors.len());
        for node in &self.bfs_successors {
            let mut successor_list: Vec<String> =
                Vec::with_capacity(node.1.len());
            for succ in &node.1 {
                successor_list.push(format!("{}", succ.as_ref(py).str()?));
            }
            str_vec.push(format!(
                "({}, [{}])",
                node.0.as_ref(py).str()?,
                successor_list.join(", ")
            ));
        }
        Ok(format!("BFSSuccessors[{}]", str_vec.join(", ")))
    }

    fn __hash__(&self) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        let gil = Python::acquire_gil();
        let py = gil.python();
        for index in &self.bfs_successors {
            hasher.write_isize(index.0.as_ref(py).hash()?);
            for succ in &index.1 {
                hasher.write_isize(succ.as_ref(py).hash()?);
            }
        }
        Ok(hasher.finish())
    }
}

#[pyproto]
impl PySequenceProtocol for BFSSuccessors {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.bfs_successors.len())
    }

    fn __getitem__(
        &'p self,
        idx: isize,
    ) -> PyResult<(PyObject, Vec<PyObject>)> {
        if idx >= self.bfs_successors.len().try_into().unwrap() {
            Err(PyIndexError::new_err(format!("Invalid index, {}", idx)))
        } else {
            Ok(self.bfs_successors[idx as usize].clone())
        }
    }
}

#[pyproto]
impl PyGCProtocol for BFSSuccessors {
    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        for node in &self.bfs_successors {
            visit.call(&node.0)?;
            for succ in &node.1 {
                visit.call(succ)?;
            }
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.bfs_successors = Vec::new();
    }
}

/// A custom class for the return of node indices
///
/// This class is a container class for the results of functions that
/// return a list of node indices. It implements the Python sequence
/// protocol. So you can treat the return as a read-only sequence/list
/// that is integer indexed. If you want to use it as an iterator you
/// can by wrapping it in an ``iter()`` that will yield the results in
/// order.
///
/// For example::
///
///     import retworkx
///
///     graph = retworkx.generators.directed_path_graph(5)
///     nodes = retworkx.node_indexes(0)
///     # Index based access
///     third_element = nodes[2]
///     # Use as iterator
///     nodes_iter = iter(node)
///     first_element = next(nodes_iter)
///     second_element = next(nodes_iter)
///
#[pyclass(module = "retworkx", gc)]
pub struct NodeIndices {
    pub nodes: Vec<usize>,
}

#[pymethods]
impl NodeIndices {
    #[new]
    fn new() -> NodeIndices {
        NodeIndices { nodes: Vec::new() }
    }

    fn __getstate__(&self) -> Vec<usize> {
        self.nodes.clone()
    }

    fn __setstate__(&mut self, state: Vec<usize>) {
        self.nodes = state;
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for NodeIndices {
    fn __richcmp__(
        &self,
        other: &'p PySequence,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<bool> {
        let compare = |other: &PySequence| -> PyResult<bool> {
            if other.len()? as usize != self.nodes.len() {
                return Ok(false);
            }
            for i in 0..self.nodes.len() {
                let other_raw = other.get_item(i.try_into().unwrap())?;
                let other_value: usize = other_raw.extract()?;
                if other_value != self.nodes[i] {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        match op {
            pyo3::basic::CompareOp::Eq => compare(other),
            pyo3::basic::CompareOp::Ne => match compare(other) {
                Ok(res) => Ok(!res),
                Err(err) => Err(err),
            },
            _ => Err(PyNotImplementedError::new_err(
                "Comparison not implemented",
            )),
        }
    }

    fn __str__(&self) -> PyResult<String> {
        let str_vec: Vec<String> =
            self.nodes.iter().map(|n| format!("{}", n)).collect();
        Ok(format!("NodeIndices[{}]", str_vec.join(", ")))
    }

    fn __hash__(&self) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        for index in &self.nodes {
            hasher.write_usize(*index);
        }
        Ok(hasher.finish())
    }
}

#[pyproto]
impl PySequenceProtocol for NodeIndices {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.nodes.len())
    }

    fn __getitem__(&'p self, idx: isize) -> PyResult<usize> {
        if idx >= self.nodes.len().try_into().unwrap() {
            Err(PyIndexError::new_err(format!("Invalid index, {}", idx)))
        } else {
            Ok(self.nodes[idx as usize])
        }
    }
}

#[pyproto]
impl PyGCProtocol for NodeIndices {
    fn __traverse__(&self, _visit: PyVisit) -> Result<(), PyTraverseError> {
        Ok(())
    }

    fn __clear__(&mut self) {}
}

/// A custom class for the return of edge lists
///
/// This class is a container class for the results of functions that
/// return a list of edges. It implements the Python sequence
/// protocol. So you can treat the return as a read-only sequence/list
/// that is integer indexed. If you want to use it as an iterator you
/// can by wrapping it in an ``iter()`` that will yield the results in
/// order.
///
/// For example::
///
///     import retworkx
///
///     graph = retworkx.generators.directed_path_graph(5)
///     edges = graph.edge_list()
///     # Index based access
///     third_element = edges[2]
///     # Use as iterator
///     edges_iter = iter(edges)
///     first_element = next(edges_iter)
///     second_element = next(edges_iter)
///
#[pyclass(module = "retworkx", gc)]
pub struct EdgeList {
    pub edges: Vec<(usize, usize)>,
}

#[pymethods]
impl EdgeList {
    #[new]
    fn new() -> EdgeList {
        EdgeList { edges: Vec::new() }
    }

    fn __getstate__(&self) -> Vec<(usize, usize)> {
        self.edges.clone()
    }

    fn __setstate__(&mut self, state: Vec<(usize, usize)>) {
        self.edges = state;
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for EdgeList {
    fn __richcmp__(
        &self,
        other: &'p PySequence,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<bool> {
        let compare = |other: &PySequence| -> PyResult<bool> {
            if other.len()? as usize != self.edges.len() {
                return Ok(false);
            }
            for i in 0..self.edges.len() {
                let other_raw = other.get_item(i.try_into().unwrap())?;
                let other_value: (usize, usize) = other_raw.extract()?;
                if other_value != self.edges[i] {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        match op {
            pyo3::basic::CompareOp::Eq => compare(other),
            pyo3::basic::CompareOp::Ne => match compare(other) {
                Ok(res) => Ok(!res),
                Err(err) => Err(err),
            },
            _ => Err(PyNotImplementedError::new_err(
                "Comparison not implemented",
            )),
        }
    }

    fn __str__(&self) -> PyResult<String> {
        let str_vec: Vec<String> = self
            .edges
            .iter()
            .map(|e| format!("({}, {})", e.0, e.1))
            .collect();
        Ok(format!("EdgeList[{}]", str_vec.join(", ")))
    }

    fn __hash__(&self) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        for edge in &self.edges {
            hasher.write_usize(edge.0);
            hasher.write_usize(edge.1);
        }
        Ok(hasher.finish())
    }
}

#[pyproto]
impl PySequenceProtocol for EdgeList {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.edges.len())
    }

    fn __getitem__(&'p self, idx: isize) -> PyResult<(usize, usize)> {
        if idx >= self.edges.len().try_into().unwrap() {
            Err(PyIndexError::new_err(format!("Invalid index, {}", idx)))
        } else {
            Ok(self.edges[idx as usize])
        }
    }
}

#[pyproto]
impl PyGCProtocol for EdgeList {
    fn __traverse__(&self, _visit: PyVisit) -> Result<(), PyTraverseError> {
        Ok(())
    }

    fn __clear__(&mut self) {}
}

/// A custom class for the return of edge lists with weights
///
/// This class is a container class for the results of functions that
/// return a list of edges with weights. It implements the Python sequence
/// protocol. So you can treat the return as a read-only sequence/list
/// that is integer indexed. If you want to use it as an iterator you
/// can by wrapping it in an ``iter()`` that will yield the results in
/// order.
///
/// For example::
///
///     import retworkx
///
///     graph = retworkx.generators.directed_path_graph(5)
///     edges = graph.weighted_edge_list()
///     # Index based access
///     third_element = edges[2]
///     # Use as iterator
///     edges_iter = iter(edges)
///     first_element = next(edges_iter)
///     second_element = next(edges_iter)
///
#[pyclass(module = "retworkx", gc)]
pub struct WeightedEdgeList {
    pub edges: Vec<(usize, usize, PyObject)>,
}

#[pymethods]
impl WeightedEdgeList {
    #[new]
    fn new() -> WeightedEdgeList {
        WeightedEdgeList { edges: Vec::new() }
    }

    fn __getstate__(&self) -> Vec<(usize, usize, PyObject)> {
        self.edges.clone()
    }

    fn __setstate__(&mut self, state: Vec<(usize, usize, PyObject)>) {
        self.edges = state;
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for WeightedEdgeList {
    fn __richcmp__(
        &self,
        other: &'p PySequence,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<bool> {
        let compare = |other: &PySequence| -> PyResult<bool> {
            if other.len()? as usize != self.edges.len() {
                return Ok(false);
            }
            let gil = Python::acquire_gil();
            let py = gil.python();
            for i in 0..self.edges.len() {
                let other_raw = other.get_item(i.try_into().unwrap())?;
                let other_value: (usize, usize, PyObject) =
                    other_raw.extract()?;
                if other_value.0 != self.edges[i].0
                    || other_value.1 != self.edges[i].1
                    || self.edges[i].2.as_ref(py).compare(other_value.2)?
                        != std::cmp::Ordering::Equal
                {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        match op {
            pyo3::basic::CompareOp::Eq => compare(other),
            pyo3::basic::CompareOp::Ne => match compare(other) {
                Ok(res) => Ok(!res),
                Err(err) => Err(err),
            },
            _ => Err(PyNotImplementedError::new_err(
                "Comparison not implemented",
            )),
        }
    }

    fn __str__(&self) -> PyResult<String> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let mut str_vec: Vec<String> = Vec::with_capacity(self.edges.len());
        for edge in &self.edges {
            str_vec.push(format!(
                "({}, {}, {})",
                edge.0,
                edge.1,
                edge.2.as_ref(py).str()?
            ));
        }
        Ok(format!("WeightedEdgeList[{}]", str_vec.join(", ")))
    }

    fn __hash__(&self) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        let gil = Python::acquire_gil();
        let py = gil.python();
        for index in &self.edges {
            hasher.write_usize(index.0);
            hasher.write_usize(index.1);
            hasher.write_isize(index.2.as_ref(py).hash()?);
        }
        Ok(hasher.finish())
    }
}

#[pyproto]
impl PySequenceProtocol for WeightedEdgeList {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.edges.len())
    }

    fn __getitem__(&'p self, idx: isize) -> PyResult<(usize, usize, PyObject)> {
        if idx >= self.edges.len().try_into().unwrap() {
            Err(PyIndexError::new_err(format!("Invalid index, {}", idx)))
        } else {
            Ok(self.edges[idx as usize].clone())
        }
    }
}

#[pyproto]
impl PyGCProtocol for WeightedEdgeList {
    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        for edge in &self.edges {
            visit.call(&edge.2)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.edges = Vec::new();
    }
}
