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

#![allow(clippy::float_cmp, clippy::upper_case_acronyms)]

use std::collections::hash_map::DefaultHasher;
use std::convert::TryInto;
use std::hash::Hasher;

use hashbrown::HashMap;

use pyo3::class::iter::{IterNextOutput, PyIterProtocol};
use pyo3::class::{PyMappingProtocol, PyObjectProtocol, PySequenceProtocol};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyNotImplementedError};
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
#[derive(Clone)]
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

/// A class representing a mapping of node indices to node indices
///
/// This class is equivalent to having a dict of the form::
///
///     {1: 0, 3: 1}
///
#[pyclass(module = "retworkx", gc)]
pub struct NodeMap {
    pub node_map: HashMap<usize, usize>,
}

#[pymethods]
impl NodeMap {
    #[new]
    fn new() -> NodeMap {
        NodeMap {
            node_map: HashMap::new(),
        }
    }

    fn __getstate__(&self) -> HashMap<usize, usize> {
        self.node_map.clone()
    }

    fn __setstate__(&mut self, state: HashMap<usize, usize>) {
        self.node_map = state;
    }

    fn keys(&self) -> NodeMapKeys {
        NodeMapKeys {
            node_map_keys: self.node_map.keys().copied().collect(),
            iter_pos: 0,
        }
    }

    fn values(&self) -> NodeMapValues {
        NodeMapValues {
            node_map_values: self.node_map.values().copied().collect(),
            iter_pos: 0,
        }
    }

    fn items(&self) -> NodeMapItems {
        let items: Vec<(usize, usize)> =
            self.node_map.iter().map(|(k, v)| (*k, *v)).collect();
        NodeMapItems {
            node_map_items: items,
            iter_pos: 0,
        }
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for NodeMap {
    fn __richcmp__(
        &self,
        other: PyObject,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<bool> {
        let compare = |other: PyObject| -> PyResult<bool> {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let other_ref = other.as_ref(py);
            if other_ref.len()? != self.node_map.len() {
                return Ok(false);
            }
            for (key, value) in &self.node_map {
                match other_ref.get_item(key) {
                    Ok(other_raw) => {
                        let other_value: usize = other_raw.extract()?;
                        if other_value != *value {
                            return Ok(false);
                        }
                    }
                    Err(ref err)
                        if Python::with_gil(|py| {
                            err.is_instance::<PyKeyError>(py)
                        }) =>
                    {
                        return Ok(false);
                    }
                    Err(err) => return Err(err),
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
        let mut str_vec: Vec<String> = Vec::with_capacity(self.node_map.len());
        for path in &self.node_map {
            str_vec.push(format!("{}: {}", path.0, path.1));
        }
        Ok(format!("NodeMap{{{}}}", str_vec.join(", ")))
    }

    fn __hash__(&self) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        for index in &self.node_map {
            hasher.write_usize(*index.0);
            hasher.write_usize(*index.1);
        }
        Ok(hasher.finish())
    }
}

#[pyproto]
impl PySequenceProtocol for NodeMap {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.node_map.len())
    }

    fn __contains__(&self, index: usize) -> PyResult<bool> {
        Ok(self.node_map.contains_key(&index))
    }
}

#[pyproto]
impl PyMappingProtocol for NodeMap {
    /// Return the number of nodes in the graph
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.node_map.len())
    }
    fn __getitem__(&'p self, idx: usize) -> PyResult<usize> {
        match self.node_map.get(&idx) {
            Some(data) => Ok(*data),
            None => Err(PyIndexError::new_err("No node found for index")),
        }
    }
}

#[pyproto]
impl PyIterProtocol for NodeMap {
    fn __iter__(slf: PyRef<Self>) -> NodeMapKeys {
        NodeMapKeys {
            node_map_keys: slf.node_map.keys().copied().collect(),
            iter_pos: 0,
        }
    }
}

#[pyproto]
impl PyGCProtocol for NodeMap {
    fn __traverse__(&self, _visit: PyVisit) -> Result<(), PyTraverseError> {
        Ok(())
    }

    fn __clear__(&mut self) {}
}

#[pyclass(module = "retworkx")]
pub struct NodeMapKeys {
    pub node_map_keys: Vec<usize>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for NodeMapKeys {
    fn __iter__(slf: PyRef<Self>) -> Py<NodeMapKeys> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<usize, &'static str> {
        if slf.iter_pos < slf.node_map_keys.len() {
            let res = IterNextOutput::Yield(slf.node_map_keys[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

#[pyclass(module = "retworkx")]
pub struct NodeMapValues {
    pub node_map_values: Vec<usize>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for NodeMapValues {
    fn __iter__(slf: PyRef<Self>) -> Py<NodeMapValues> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<usize, &'static str> {
        if slf.iter_pos < slf.node_map_values.len() {
            let res = IterNextOutput::Yield(slf.node_map_values[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

#[pyclass(module = "retworkx")]
pub struct NodeMapItems {
    pub node_map_items: Vec<(usize, usize)>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for NodeMapItems {
    fn __iter__(slf: PyRef<Self>) -> Py<NodeMapItems> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<(usize, usize), &'static str> {
        if slf.iter_pos < slf.node_map_items.len() {
            let res = IterNextOutput::Yield(slf.node_map_items[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

/// A class representing a mapping of node indices to 2D positions
///
/// This class is equivalent to having a dict of the form::
///
///     {1: [0, 1], 3: [0.5, 1.2]}
///
/// It is used to efficiently represent a retworkx generated 2D layout for a
/// graph. It behaves as a drop in replacement for a readonly ``dict``.
#[pyclass(module = "retworkx", gc)]
pub struct Pos2DMapping {
    pub pos_map: HashMap<usize, [f64; 2]>,
}

#[pymethods]
impl Pos2DMapping {
    #[new]
    fn new() -> Pos2DMapping {
        Pos2DMapping {
            pos_map: HashMap::new(),
        }
    }

    fn __getstate__(&self) -> HashMap<usize, [f64; 2]> {
        self.pos_map.clone()
    }

    fn __setstate__(&mut self, state: HashMap<usize, [f64; 2]>) {
        self.pos_map = state;
    }

    fn keys(&self) -> Pos2DMappingKeys {
        Pos2DMappingKeys {
            pos_keys: self.pos_map.keys().copied().collect(),
            iter_pos: 0,
        }
    }

    fn values(&self) -> Pos2DMappingValues {
        Pos2DMappingValues {
            pos_values: self.pos_map.values().copied().collect(),
            iter_pos: 0,
        }
    }

    fn items(&self) -> Pos2DMappingItems {
        let items: Vec<(usize, [f64; 2])> =
            self.pos_map.iter().map(|(k, v)| (*k, *v)).collect();
        Pos2DMappingItems {
            pos_items: items,
            iter_pos: 0,
        }
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for Pos2DMapping {
    fn __richcmp__(
        &self,
        other: PyObject,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<bool> {
        let compare = |other: PyObject| -> PyResult<bool> {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let other_ref = other.as_ref(py);
            if other_ref.len()? != self.pos_map.len() {
                return Ok(false);
            }
            for (key, value) in &self.pos_map {
                match other_ref.get_item(key) {
                    Ok(other_raw) => {
                        let other_value: [f64; 2] = other_raw.extract()?;
                        if other_value != *value {
                            return Ok(false);
                        }
                    }
                    Err(ref err)
                        if Python::with_gil(|py| {
                            err.is_instance::<PyKeyError>(py)
                        }) =>
                    {
                        return Ok(false);
                    }
                    Err(err) => return Err(err),
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
        let mut str_vec: Vec<String> = Vec::with_capacity(self.pos_map.len());
        for path in &self.pos_map {
            str_vec.push(format!("{}: ({}, {})", path.0, path.1[0], path.1[1]));
        }
        Ok(format!("Pos2DMapping{{{}}}", str_vec.join(", ")))
    }

    fn __hash__(&self) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        for index in &self.pos_map {
            hasher.write_usize(*index.0);
            hasher.write(&index.1[0].to_be_bytes());
            hasher.write(&index.1[1].to_be_bytes());
        }
        Ok(hasher.finish())
    }
}

#[pyproto]
impl PySequenceProtocol for Pos2DMapping {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.pos_map.len())
    }

    fn __contains__(&self, index: usize) -> PyResult<bool> {
        Ok(self.pos_map.contains_key(&index))
    }
}

#[pyproto]
impl PyMappingProtocol for Pos2DMapping {
    /// Return the number of nodes in the graph
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.pos_map.len())
    }
    fn __getitem__(&'p self, idx: usize) -> PyResult<[f64; 2]> {
        match self.pos_map.get(&idx) {
            Some(data) => Ok(*data),
            None => Err(PyIndexError::new_err("No node found for index")),
        }
    }
}

#[pyproto]
impl PyIterProtocol for Pos2DMapping {
    fn __iter__(slf: PyRef<Self>) -> Pos2DMappingKeys {
        Pos2DMappingKeys {
            pos_keys: slf.pos_map.keys().copied().collect(),
            iter_pos: 0,
        }
    }
}

#[pyproto]
impl PyGCProtocol for Pos2DMapping {
    fn __traverse__(&self, _visit: PyVisit) -> Result<(), PyTraverseError> {
        Ok(())
    }

    fn __clear__(&mut self) {}
}

#[pyclass(module = "retworkx")]
pub struct Pos2DMappingKeys {
    pub pos_keys: Vec<usize>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for Pos2DMappingKeys {
    fn __iter__(slf: PyRef<Self>) -> Py<Pos2DMappingKeys> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<usize, &'static str> {
        if slf.iter_pos < slf.pos_keys.len() {
            let res = IterNextOutput::Yield(slf.pos_keys[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

#[pyclass(module = "retworkx")]
pub struct Pos2DMappingValues {
    pub pos_values: Vec<[f64; 2]>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for Pos2DMappingValues {
    fn __iter__(slf: PyRef<Self>) -> Py<Pos2DMappingValues> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<[f64; 2], &'static str> {
        if slf.iter_pos < slf.pos_values.len() {
            let res = IterNextOutput::Yield(slf.pos_values[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

#[pyclass(module = "retworkx")]
pub struct Pos2DMappingItems {
    pub pos_items: Vec<(usize, [f64; 2])>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for Pos2DMappingItems {
    fn __iter__(slf: PyRef<Self>) -> Py<Pos2DMappingItems> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<(usize, [f64; 2]), &'static str> {
        if slf.iter_pos < slf.pos_items.len() {
            let res = IterNextOutput::Yield(slf.pos_items[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

/// A custom class for the return of paths to target nodes
///
/// This class is a container class for the results of functions that
/// return a mapping of target nodes and paths. It implements the Python
/// mapping protocol. So you can treat the return as a read-only
/// mapping/dict. If you want to use it as an iterator you can by
/// wrapping it in an ``iter()`` that will yield the results in
/// order.
///
/// For example::
///
///     import retworkx
///
///     graph = retworkx.generators.directed_path_graph(5)
///     edges = retworkx.dijkstra_shortest_paths(0)
///     # Target node access
///     third_element = edges[2]
///     # Use as iterator
///     edges_iter = iter(edges)
///     first_target = next(edges_iter)
///     first_path = edges[first_target]
///     second_target = next(edges_iter)
///     second_path = edges[second_target]
///
#[pyclass(module = "retworkx", gc)]
pub struct PathMapping {
    pub paths: HashMap<usize, Vec<usize>>,
}

#[pymethods]
impl PathMapping {
    #[new]
    fn new() -> PathMapping {
        PathMapping {
            paths: HashMap::new(),
        }
    }

    fn __getstate__(&self) -> HashMap<usize, Vec<usize>> {
        self.paths.clone()
    }

    fn __setstate__(&mut self, state: HashMap<usize, Vec<usize>>) {
        self.paths = state;
    }

    fn keys(&self) -> PathMappingKeys {
        PathMappingKeys {
            path_keys: self.paths.keys().copied().collect(),
            iter_pos: 0,
        }
    }

    fn values(&self) -> PathMappingValues {
        PathMappingValues {
            path_values: self
                .paths
                .values()
                .map(|v| NodeIndices { nodes: v.to_vec() })
                .collect(),
            iter_pos: 0,
        }
    }

    fn items(&self) -> PathMappingItems {
        let items: Vec<(usize, NodeIndices)> = self
            .paths
            .iter()
            .map(|(k, v)| (*k, NodeIndices { nodes: v.to_vec() }))
            .collect();
        PathMappingItems {
            path_items: items,
            iter_pos: 0,
        }
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for PathMapping {
    fn __richcmp__(
        &self,
        other: PyObject,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<bool> {
        let compare = |other: PyObject| -> PyResult<bool> {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let other_ref = other.as_ref(py);
            if other_ref.len()? != self.paths.len() {
                return Ok(false);
            }
            for (key, value) in &self.paths {
                match other_ref.get_item(key) {
                    Ok(other_raw) => {
                        let other_value: &PySequence =
                            other_raw.downcast::<PySequence>()?;
                        if value.len() as isize != other_value.len()? {
                            return Ok(false);
                        }
                        for (i, item) in value.iter().enumerate() {
                            let other_item_raw =
                                other_value.get_item(i as isize)?;
                            let other_item_value: usize =
                                other_item_raw.extract()?;
                            if other_item_value != *item {
                                return Ok(false);
                            }
                        }
                    }
                    Err(ref err)
                        if Python::with_gil(|py| {
                            err.is_instance::<PyKeyError>(py)
                        }) =>
                    {
                        return Ok(false);
                    }
                    Err(err) => return Err(err),
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
        let mut str_vec: Vec<String> = Vec::with_capacity(self.paths.len());
        for path in &self.paths {
            str_vec.push(format!(
                "{}: {}",
                path.0,
                format!(
                    "[{}]",
                    path.1
                        .iter()
                        .map(|n| format!("{}", n))
                        .collect::<Vec<String>>()
                        .join(", ")
                ),
            ));
        }
        Ok(format!("PathMapping{{{}}}", str_vec.join(", ")))
    }

    fn __hash__(&self) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        for index in &self.paths {
            hasher.write_usize(*index.0);
            for node in index.1 {
                hasher.write_usize(*node);
            }
        }
        Ok(hasher.finish())
    }
}

#[pyproto]
impl PyMappingProtocol for PathMapping {
    /// Return the number of nodes in the graph
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.paths.len())
    }
    fn __getitem__(&'p self, idx: usize) -> PyResult<NodeIndices> {
        match self.paths.get(&idx) {
            Some(data) => Ok(NodeIndices {
                nodes: data.clone(),
            }),
            None => Err(PyIndexError::new_err("No node found for index")),
        }
    }
}

#[pyproto]
impl PySequenceProtocol for PathMapping {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.paths.len())
    }

    fn __contains__(&self, index: usize) -> PyResult<bool> {
        Ok(self.paths.contains_key(&index))
    }
}

#[pyproto]
impl PyIterProtocol for PathMapping {
    fn __iter__(slf: PyRef<Self>) -> PathMappingKeys {
        PathMappingKeys {
            path_keys: slf.paths.keys().copied().collect(),
            iter_pos: 0,
        }
    }
}

#[pyproto]
impl PyGCProtocol for PathMapping {
    fn __traverse__(&self, _visit: PyVisit) -> Result<(), PyTraverseError> {
        Ok(())
    }

    fn __clear__(&mut self) {}
}

#[pyclass(module = "retworkx")]
pub struct PathMappingKeys {
    pub path_keys: Vec<usize>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for PathMappingKeys {
    fn __iter__(slf: PyRef<Self>) -> Py<PathMappingKeys> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<usize, &'static str> {
        if slf.iter_pos < slf.path_keys.len() {
            let res = IterNextOutput::Yield(slf.path_keys[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

#[pyclass(module = "retworkx")]
pub struct PathMappingValues {
    pub path_values: Vec<NodeIndices>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for PathMappingValues {
    fn __iter__(slf: PyRef<Self>) -> Py<PathMappingValues> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<NodeIndices, &'static str> {
        if slf.iter_pos < slf.path_values.len() {
            let res =
                IterNextOutput::Yield(slf.path_values[slf.iter_pos].clone());
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

#[pyclass(module = "retworkx")]
pub struct PathMappingItems {
    pub path_items: Vec<(usize, NodeIndices)>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for PathMappingItems {
    fn __iter__(slf: PyRef<Self>) -> Py<PathMappingItems> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<(usize, NodeIndices), &'static str> {
        if slf.iter_pos < slf.path_items.len() {
            let res =
                IterNextOutput::Yield(slf.path_items[slf.iter_pos].clone());
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

/// A custom class for the return of path lengths to target nodes
///
/// This class is a container class for the results of functions that
/// return a mapping of target nodes and paths. It implements the Python
/// mapping protocol. So you can treat the return as a read-only
/// mapping/dict. If you want to use it as an iterator you can by
/// wrapping it in an ``iter()`` that will yield the results in
/// order.
///
/// For example::
///
///     import retworkx
///
///     graph = retworkx.generators.directed_path_graph(5)
///     edges = retworkx.dijkstra_shortest_path_lengths(0)
///     # Target node access
///     third_element = edges[2]
///     # Use as iterator
///     edges_iter = iter(edges)
///     first_target = next(edges_iter)
///     first_path = edges[first_target]
///     second_target = next(edges_iter)
///     second_path = edges[second_target]
///
#[pyclass(module = "retworkx", gc)]
pub struct PathLengthMapping {
    pub path_lengths: HashMap<usize, f64>,
}

#[pymethods]
impl PathLengthMapping {
    #[new]
    fn new() -> PathLengthMapping {
        PathLengthMapping {
            path_lengths: HashMap::new(),
        }
    }

    fn __getstate__(&self) -> HashMap<usize, f64> {
        self.path_lengths.clone()
    }

    fn __setstate__(&mut self, state: HashMap<usize, f64>) {
        self.path_lengths = state;
    }

    fn keys(&self) -> PathLengthMappingKeys {
        PathLengthMappingKeys {
            path_length_keys: self.path_lengths.keys().copied().collect(),
            iter_pos: 0,
        }
    }

    fn values(&self) -> PathLengthMappingValues {
        PathLengthMappingValues {
            path_length_values: self.path_lengths.values().copied().collect(),
            iter_pos: 0,
        }
    }

    fn items(&self) -> PathLengthMappingItems {
        let items: Vec<(usize, f64)> =
            self.path_lengths.iter().map(|(k, v)| (*k, *v)).collect();
        PathLengthMappingItems {
            path_length_items: items,
            iter_pos: 0,
        }
    }
}

#[pyproto]
impl<'p> PyObjectProtocol<'p> for PathLengthMapping {
    fn __richcmp__(
        &self,
        other: PyObject,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<bool> {
        let compare = |other: PyObject| -> PyResult<bool> {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let other_ref = other.as_ref(py);
            if other_ref.len()? != self.path_lengths.len() {
                return Ok(false);
            }
            for (key, value) in &self.path_lengths {
                match other_ref.get_item(key) {
                    Ok(other_raw) => {
                        let other_value: f64 = other_raw.extract()?;
                        if other_value != *value {
                            return Ok(false);
                        }
                    }
                    Err(ref err)
                        if Python::with_gil(|py| {
                            err.is_instance::<PyKeyError>(py)
                        }) =>
                    {
                        return Ok(false);
                    }
                    Err(err) => return Err(err),
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
        let mut str_vec: Vec<String> =
            Vec::with_capacity(self.path_lengths.len());
        for path in &self.path_lengths {
            str_vec.push(format!("{}: {}", path.0, path.1,));
        }
        Ok(format!("PathLengthMapping{{{}}}", str_vec.join(", ")))
    }

    fn __hash__(&self) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        for index in &self.path_lengths {
            hasher.write_usize(*index.0);
            hasher.write(&index.1.to_be_bytes());
        }
        Ok(hasher.finish())
    }
}

#[pyproto]
impl PySequenceProtocol for PathLengthMapping {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.path_lengths.len())
    }

    fn __contains__(&self, index: usize) -> PyResult<bool> {
        Ok(self.path_lengths.contains_key(&index))
    }
}

#[pyproto]
impl PyMappingProtocol for PathLengthMapping {
    /// Return the number of nodes in the graph
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.path_lengths.len())
    }
    fn __getitem__(&'p self, idx: usize) -> PyResult<f64> {
        match self.path_lengths.get(&idx) {
            Some(data) => Ok(*data),
            None => Err(PyIndexError::new_err("No node found for index")),
        }
    }
}

#[pyproto]
impl PyIterProtocol for PathLengthMapping {
    fn __iter__(slf: PyRef<Self>) -> PathLengthMappingKeys {
        PathLengthMappingKeys {
            path_length_keys: slf.path_lengths.keys().copied().collect(),
            iter_pos: 0,
        }
    }
}

#[pyproto]
impl PyGCProtocol for PathLengthMapping {
    fn __traverse__(&self, _visit: PyVisit) -> Result<(), PyTraverseError> {
        Ok(())
    }

    fn __clear__(&mut self) {}
}

#[pyclass(module = "retworkx")]
pub struct PathLengthMappingKeys {
    pub path_length_keys: Vec<usize>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for PathLengthMappingKeys {
    fn __iter__(slf: PyRef<Self>) -> Py<PathLengthMappingKeys> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<usize, &'static str> {
        if slf.iter_pos < slf.path_length_keys.len() {
            let res = IterNextOutput::Yield(slf.path_length_keys[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

#[pyclass(module = "retworkx")]
pub struct PathLengthMappingValues {
    pub path_length_values: Vec<f64>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for PathLengthMappingValues {
    fn __iter__(slf: PyRef<Self>) -> Py<PathLengthMappingValues> {
        slf.into()
    }
    fn __next__(mut slf: PyRefMut<Self>) -> IterNextOutput<f64, &'static str> {
        if slf.iter_pos < slf.path_length_values.len() {
            let res =
                IterNextOutput::Yield(slf.path_length_values[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}

#[pyclass(module = "retworkx")]
pub struct PathLengthMappingItems {
    pub path_length_items: Vec<(usize, f64)>,
    iter_pos: usize,
}

#[pyproto]
impl PyIterProtocol for PathLengthMappingItems {
    fn __iter__(slf: PyRef<Self>) -> Py<PathLengthMappingItems> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<(usize, f64), &'static str> {
        if slf.iter_pos < slf.path_length_items.len() {
            let res =
                IterNextOutput::Yield(slf.path_length_items[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}
