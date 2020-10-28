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

use std::convert::TryInto;

use pyo3::class::iter::{IterNextOutput, PyIterProtocol};
use pyo3::class::PySequenceProtocol;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

/// A custom iterator class for the return from :func:`retworkx.bfs_successors`
///
/// This class is a container class for the results of the
/// :func:`retworkx.bfs_successors` function. It implements both the Python
/// iterator protocol and sequence protocol. So you can treat the return as
/// either a read-only sequence/list that is integer indexed or use it as an
/// iterator that will yield the results in order.
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
///     first_element = next(bfs_succ)
///     second_element = nex(bfs_succ)
///
#[pyclass(module = "retworkx")]
pub struct BFSSuccessors {
    pub bfs_successors: Vec<(PyObject, Vec<PyObject>)>,
    pub index: usize,
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
impl PyIterProtocol for BFSSuccessors {
    fn __iter__(slf: PyRef<Self>) -> Py<BFSSuccessors> {
        slf.into()
    }
    fn __next__(
        mut slf: PyRefMut<Self>,
    ) -> IterNextOutput<(PyObject, Vec<PyObject>), &'static str> {
        if slf.index < slf.bfs_successors.len() {
            let res =
                IterNextOutput::Yield(slf.bfs_successors[slf.index].clone());
            slf.index += 1;
            res
        } else {
            IterNextOutput::Return("Ended")
        }
    }
}
