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

// There are two useful macros to quickly define a new custom return type:
//
// :`custom_vec_iter_impl` holds a `Vec<T>` and can be used as a
//  read-only sequence/list. To use it, you should specify the name of the new type,
//  the name of the vector that holds the data, the type `T` and a docstring.
//
//  e.g `custom_vec_iter_impl!(MyReadOnlyType, data, (usize, f64), "Docs");`
//      defines a new type named `MyReadOnlyType` that holds a vector called `data`
//      of values `(usize, f64)`.
//
// :`custom_hash_map_iter_impl` holds a `HashMap<K, V>` and can be used as
//  a read-only mapping/dict. To use it, you should specify the name of the new type,
//  the name of the hash map that holds the data, the type of the keys `K`,
//  the type of the values `V` and a docstring.
//
//  e.g `custom_hash_map_iter_impl!(MyReadOnlyType, data, usize, f64, "Docs");`
//      defines a new type named `MyReadOnlyType` that holds a mapping called `data`
//      from `usize` to `f64`.
//
// You should always implement `PyGCProtocol` for the new custom return type. If you
// don't store any python object, just use the macro `default_pygc_protocol_impl`.
//
// e.g `default_pygc_protocol_impl!(MyReadOnlyType);`
//
// Types `T, K, V` above should implement `PyHash`, `PyEq`, `PyDisplay` traits.
// These are arleady implemented for many primitive rust types and `PyObject`.

#![allow(clippy::float_cmp, clippy::upper_case_acronyms)]

use std::collections::hash_map::DefaultHasher;
use std::convert::TryInto;
use std::hash::Hasher;

use hashbrown::HashMap;
use num_bigint::BigUint;

use pyo3::class::iter::{IterNextOutput, PyIterProtocol};
use pyo3::class::{PyMappingProtocol, PyObjectProtocol, PySequenceProtocol};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyNotImplementedError};
use pyo3::gc::{PyGCProtocol, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PySequence;
use pyo3::PyTraverseError;

macro_rules! last_type {
     ($a:ident,) => { $a };
     ($a:ident, $($rest_a:ident,)+) => { last_type!($($rest_a,)+) };
 }

// similar to `std::hash::Hash` trait.
trait PyHash {
    fn hash<H: Hasher>(&self, py: Python, state: &mut H) -> PyResult<()>;
}

impl PyHash for PyObject {
    #[inline]
    fn hash<H: Hasher>(&self, py: Python, state: &mut H) -> PyResult<()> {
        state.write_isize(self.as_ref(py).hash()?);
        Ok(())
    }
}

// see https://doc.rust-lang.org/src/core/hash/mod.rs.html#553
macro_rules! pyhash_impl {
    ($(($ty:ident, $meth:ident),)*) => ($(
        impl PyHash for $ty {
            #[inline]
            fn hash<H: Hasher>(&self, _: Python, state: &mut H) -> PyResult<()>{
                state.$meth(*self);
                Ok(())
            }
        }
    )*)
}

pyhash_impl! {
    (u8, write_u8),
    (u16, write_u16),
    (u32, write_u32),
    (u64, write_u64),
    (usize, write_usize),
    (i8, write_i8),
    (i16, write_i16),
    (i32, write_i32),
    (i64, write_i64),
    (isize, write_isize),
    (u128, write_u128),
    (i128, write_i128),
}

impl PyHash for f64 {
    #[inline]
    fn hash<H: Hasher>(&self, _: Python, state: &mut H) -> PyResult<()> {
        state.write(&self.to_be_bytes());
        Ok(())
    }
}

impl PyHash for BigUint {
    #[inline]
    fn hash<H: Hasher>(&self, _: Python, state: &mut H) -> PyResult<()> {
        self.iter_u64_digits().for_each(|i| state.write_u64(i));
        Ok(())
    }
}

// see https://doc.rust-lang.org/src/core/hash/mod.rs.html#624
macro_rules! pyhash_tuple_impls {
     ( $($name:ident)+) => (
         impl<$($name: PyHash),+> PyHash for ($($name,)+) where last_type!($($name,)+): ?Sized {
             #[allow(non_snake_case)]
             #[inline]
             fn hash<S: Hasher>(&self, py: Python, state: &mut S) -> PyResult<()> {
                 let ($(ref $name,)+) = *self;
                 $($name.hash(py, state)?;)+
                 Ok(())
             }
         }
     );
 }

pyhash_tuple_impls! { A }
pyhash_tuple_impls! { A B }
pyhash_tuple_impls! { A B C }

impl<T: PyHash> PyHash for [T] {
    #[inline]
    fn hash<H: Hasher>(&self, py: Python, state: &mut H) -> PyResult<()> {
        // self.len().hash(py, state)?;
        for elem in self {
            elem.hash(py, state)?;
        }
        Ok(())
    }
}

macro_rules! pyhash_array_impls {
    ($($N:expr)+) =>  {$(
        impl<T: PyHash> PyHash for [T; $N] {
            fn hash<H: Hasher>(&self, py: Python, state: &mut H) -> PyResult<()> {
                PyHash::hash(&self[..], py, state)?;
                Ok(())
            }
        }
    )+}
}

pyhash_array_impls! {2 3}

impl<T: PyHash> PyHash for Vec<T> {
    #[inline]
    fn hash<H: Hasher>(&self, py: Python, state: &mut H) -> PyResult<()> {
        PyHash::hash(&self[..], py, state)?;
        Ok(())
    }
}

impl<K: PyHash, V: PyHash> PyHash for HashMap<K, V> {
    #[inline]
    fn hash<H: Hasher>(&self, py: Python, state: &mut H) -> PyResult<()> {
        for (key, value) in self {
            key.hash(py, state)?;
            value.hash(py, state)?;
        }
        Ok(())
    }
}

// similar to `std::cmp::PartialEq` trait.
trait PyEq<Rhs: ?Sized = Self> {
    fn eq(&self, other: &Rhs, py: Python) -> PyResult<bool>;
}

impl PyEq for PyObject {
    #[inline]
    fn eq(&self, other: &Self, py: Python) -> PyResult<bool> {
        Ok(self.as_ref(py).compare(other)? == std::cmp::Ordering::Equal)
    }
}

macro_rules! pyeq_impl {
    ($($t:ty)*) => ($(
        impl PyEq for $t {
            #[inline]
            fn eq(&self, other: &Self, _: Python) -> PyResult<bool> {
                Ok((*self) == (*other))
            }
        }
    )*)
}

pyeq_impl! {bool char usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 BigUint}

// see https://doc.rust-lang.org/src/core/tuple.rs.html#7
macro_rules! pyeq_tuple_impls {
    ($(
        $Tuple:ident {
            $(($idx:tt) -> $T:ident)+
        }
    )+) => {
        $(
            impl<$($T:PyEq),+> PyEq for ($($T,)+) where last_type!($($T,)+): ?Sized {
                #[allow(clippy::needless_question_mark)]
                #[inline]
                fn eq(&self, other: &($($T,)+), py: Python) -> PyResult<bool> {
                    Ok($(self.$idx.eq(&other.$idx, py)?)&&+)
                }
            }
        )+
    }
}

pyeq_tuple_impls! {
    Tuple1 {
        (0) -> A
    }
    Tuple2 {
        (0) -> A
        (1) -> B
    }
    Tuple3 {
        (0) -> A
        (1) -> B
        (2) -> C
    }
}

impl<A, B> PyEq<[B]> for [A]
where
    A: PyEq<B>,
{
    #[inline]
    fn eq(&self, other: &[B], py: Python) -> PyResult<bool> {
        if self.len() != other.len() {
            return Ok(false);
        }

        for (x, y) in self.iter().zip(other.iter()) {
            if !PyEq::eq(x, y, py)? {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

macro_rules! pyeq_array_impls {
    ($($N:expr)+) =>  {$(
        impl<A, B> PyEq<[B; $N]> for [A; $N]
        where
            A: PyEq<B>,
        {
            #[inline]
            fn eq(&self, other: &[B; $N], py: Python) -> PyResult<bool> {
                PyEq::eq(&self[..], &other[..], py)
            }
        }
    )+}
}

pyeq_array_impls! {2 3}

impl<A, B> PyEq<Vec<B>> for Vec<A>
where
    A: PyEq<B>,
{
    #[inline]
    fn eq(&self, other: &Vec<B>, py: Python) -> PyResult<bool> {
        PyEq::eq(&self[..], &other[..], py)
    }
}

impl<T> PyEq<PyAny> for T
where
    for<'p> T: PyEq<T> + Clone + FromPyObject<'p>,
{
    #[inline]
    fn eq(&self, other: &PyAny, py: Python) -> PyResult<bool> {
        let other_value: T = other.extract()?;
        PyEq::eq(self, &other_value, py)
    }
}

impl<A> PyEq<PySequence> for Vec<A>
where
    A: PyEq<PyAny>,
{
    #[inline]
    fn eq(&self, other: &PySequence, py: Python) -> PyResult<bool> {
        if other.len()? as usize != self.len() {
            return Ok(false);
        }

        for (i, item) in self.iter().enumerate() {
            let other_raw = other.get_item(i.try_into().unwrap())?;
            if !PyEq::eq(item, other_raw, py)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl<K, V> PyEq<PyObject> for HashMap<K, V>
where
    for<'p> K: PyEq<K> + Clone + pyo3::ToBorrowedObject,
    for<'p> V: PyEq<PyAny>,
{
    #[inline]
    fn eq(&self, other: &PyObject, py: Python) -> PyResult<bool> {
        let other_ref = other.as_ref(py);
        if other_ref.len()? != self.len() {
            return Ok(false);
        }
        for (key, value) in self {
            match other_ref.get_item(key) {
                Ok(other_raw) => {
                    if !PyEq::eq(value, other_raw, py)? {
                        return Ok(false);
                    }
                }
                Err(ref err) if err.is_instance::<PyKeyError>(py) => {
                    return Ok(false);
                }
                Err(err) => return Err(err),
            }
        }
        Ok(true)
    }
}

trait PyDisplay {
    fn str(&self, py: Python) -> PyResult<String>;
}

impl PyDisplay for PyObject {
    fn str(&self, py: Python) -> PyResult<String> {
        Ok(format!("{}", self.as_ref(py).str()?))
    }
}

macro_rules! py_display_impl {
    ($($t:ty)*) => ($(
        impl PyDisplay for $t {
            fn str(&self, _: Python) -> PyResult<String> {
                Ok(format!{"{}", self})
            }
        }
    )*)
}

py_display_impl! {bool char usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64 BigUint}

macro_rules! py_display_tuple_impls {
     ( $($name:ident)+) => (
         impl<$($name: PyDisplay),+> PyDisplay for ($($name,)+) where last_type!($($name,)+): ?Sized {
             #[allow(non_snake_case)]
             fn str(&self, py: Python) -> PyResult<String> {
                 let ($(ref $name,)+) = *self;
                 let mut str_vec: Vec<String> = Vec::new();
                 $(str_vec.push($name.str(py)?);)+

                 Ok(format!("({})", str_vec.join(", ")))
             }
         }
     );
 }

py_display_tuple_impls! { A }
py_display_tuple_impls! { A B }
py_display_tuple_impls! { A B C }

impl<A: PyDisplay> PyDisplay for [A] {
    fn str(&self, py: Python) -> PyResult<String> {
        let mut str_vec: Vec<String> = Vec::with_capacity(self.len());
        for elem in self {
            str_vec.push(elem.str(py)?);
        }

        Ok(format!("[{}]", str_vec.join(", ")))
    }
}

macro_rules! py_display_array_impls {
    ($($N:expr)+) =>  {$(
        impl<A: PyDisplay> PyDisplay for [A; $N] {
            fn str(&self, py: Python) -> PyResult<String> {
                self[..].str(py)
            }
        }
    )+}
}

py_display_array_impls! {2 3}

impl<A: PyDisplay> PyDisplay for Vec<A> {
    fn str(&self, py: Python) -> PyResult<String> {
        self[..].str(py)
    }
}

impl<K: PyDisplay, V: PyDisplay> PyDisplay for HashMap<K, V> {
    fn str(&self, py: Python) -> PyResult<String> {
        let mut str_vec: Vec<String> = Vec::with_capacity(self.len());
        for elem in self {
            str_vec.push(format!("{}: {}", elem.0.str(py)?, elem.1.str(py)?));
        }

        Ok(format!("{{{}}}", str_vec.join(", ")))
    }
}

macro_rules! default_pygc_protocol_impl {
    ($name:ident) => {
        #[pyproto]
        impl PyGCProtocol for $name {
            fn __traverse__(&self, _: PyVisit) -> Result<(), PyTraverseError> {
                Ok(())
            }

            fn __clear__(&mut self) {}
        }
    };
}

macro_rules! custom_vec_iter_impl {
    ($name:ident, $data:ident, $T:ty, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(module = "retworkx", gc)]
        #[derive(Clone)]
        pub struct $name {
            pub $data: Vec<$T>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self {
                $name { $data: Vec::new() }
            }

            fn __getstate__(&self) -> Vec<$T> {
                self.$data.clone()
            }

            fn __setstate__(&mut self, state: Vec<$T>) {
                self.$data = state;
            }
        }

        #[pyproto]
        impl<'p> PyObjectProtocol<'p> for $name {
            fn __richcmp__(
                &self,
                other: &'p PySequence,
                op: pyo3::basic::CompareOp,
            ) -> PyResult<bool> {
                let compare = |other: &PySequence| -> PyResult<bool> {
                    Python::with_gil(|py| PyEq::eq(&self.$data, other, py))
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
                Python::with_gil(|py| {
                    Ok(format!("{}{}", stringify!($name), self.$data.str(py)?))
                })
            }

            fn __hash__(&self) -> PyResult<u64> {
                let mut hasher = DefaultHasher::new();
                Python::with_gil(|py| {
                    PyHash::hash(&self.$data, py, &mut hasher)
                })?;

                Ok(hasher.finish())
            }
        }

        #[pyproto]
        impl PySequenceProtocol for $name {
            fn __len__(&self) -> PyResult<usize> {
                Ok(self.$data.len())
            }

            fn __getitem__(&'p self, idx: isize) -> PyResult<$T> {
                if idx >= self.$data.len().try_into().unwrap() {
                    Err(PyIndexError::new_err(format!(
                        "Invalid index, {}",
                        idx
                    )))
                } else {
                    Ok(self.$data[idx as usize].clone())
                }
            }
        }
    };
}

custom_vec_iter_impl!(
    BFSSuccessors,
    bfs_successors,
    (PyObject, Vec<PyObject>),
    "A custom class for the return from :func:`retworkx.bfs_successors`
    
    This class is a container class for the results of the
    :func:`retworkx.bfs_successors` function. It implements the Python
    sequence protocol. So you can treat the return as read-only
    sequence/list that is integer indexed. If you want to use it as an
    iterator you can by wrapping it in an ``iter()`` that will yield the
    results in order.

    For example::

        import retworkx

        graph = retworkx.generators.directed_path_graph(5)
        bfs_succ = retworkx.bfs_successors(0)
        # Index based access
        third_element = bfs_succ[2]
        # Use as iterator
        bfs_iter = iter(bfs_succ)
        first_element = next(bfs_iter)
        second_element = nex(bfs_iter)

    "
);

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

custom_vec_iter_impl!(
    NodeIndices,
    nodes,
    usize,
    "A custom class for the return of node indices

    This class is a container class for the results of functions that
    return a list of node indices. It implements the Python sequence
    protocol. So you can treat the return as a read-only sequence/list
    that is integer indexed. If you want to use it as an iterator you
    can by wrapping it in an ``iter()`` that will yield the results in
    order.

    For example::

        import retworkx

        graph = retworkx.generators.directed_path_graph(5)
        nodes = retworkx.node_indexes(0)
        # Index based access
        third_element = nodes[2]
        # Use as iterator
        nodes_iter = iter(node)
        first_element = next(nodes_iter)
        second_element = next(nodes_iter)

    "
);
default_pygc_protocol_impl!(NodeIndices);

custom_vec_iter_impl!(
    EdgeList,
    edges,
    (usize, usize),
    "A custom class for the return of edge lists

    This class is a container class for the results of functions that
    return a list of edges. It implements the Python sequence
    protocol. So you can treat the return as a read-only sequence/list
    that is integer indexed. If you want to use it as an iterator you
    can by wrapping it in an ``iter()`` that will yield the results in
    order.

    For example::

        import retworkx

        graph = retworkx.generators.directed_path_graph(5)
        edges = graph.edge_list()
        # Index based access
        third_element = edges[2]
        # Use as iterator
        edges_iter = iter(edges)
        first_element = next(edges_iter)
        second_element = next(edges_iter)

    "
);
default_pygc_protocol_impl!(EdgeList);

custom_vec_iter_impl!(
    WeightedEdgeList,
    edges,
    (usize, usize, PyObject),
    "A custom class for the return of edge lists with weights

    This class is a container class for the results of functions that
    return a list of edges with weights. It implements the Python sequence
    protocol. So you can treat the return as a read-only sequence/list
    that is integer indexed. If you want to use it as an iterator you
    can by wrapping it in an ``iter()`` that will yield the results in
    order.

    For example::

        import retworkx

        graph = retworkx.generators.directed_path_graph(5)
        edges = graph.weighted_edge_list()
        # Index based access
        third_element = edges[2]
        # Use as iterator
        edges_iter = iter(edges)
        first_element = next(edges_iter)
        second_element = next(edges_iter)

    "
);

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

custom_vec_iter_impl!(
    EdgeIndices,
    edges,
    usize,
    "A custom class for the return of edge indices
    
    This class is a container class for the results of functions that
    return a list of edge indices. It implements the Python sequence
    protocol. So you can treat the return as a read-only sequence/list
    that is integer indexed. If you want to use it as an iterator you
    can by wrapping it in an ``iter()`` that will yield the results in
    order.

    For example::

        import retworkx

        graph = retworkx.generators.directed_path_graph(5)
        edges = retworkx.edge_indices()
        # Index based access
        third_element = edges[2]
        # Use as iterator
        edges_iter = iter(edges)
        first_element = next(edges_iter)
        second_element = next(edges_iter)

    "
);
default_pygc_protocol_impl!(EdgeIndices);

macro_rules! py_object_protocol_impl {
    ($name:ident, $data:ident) => {
        #[pyproto]
        impl<'p> PyObjectProtocol<'p> for $name {
            fn __richcmp__(
                &self,
                other: PyObject,
                op: pyo3::basic::CompareOp,
            ) -> PyResult<bool> {
                let compare = |other: PyObject| -> PyResult<bool> {
                    Python::with_gil(|py| PyEq::eq(&self.$data, &other, py))
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
                Python::with_gil(|py| {
                    Ok(format!("{}{}", stringify!($name), self.$data.str(py)?))
                })
            }

            fn __hash__(&self) -> PyResult<u64> {
                let mut hasher = DefaultHasher::new();
                Python::with_gil(|py| {
                    PyHash::hash(&self.$data, py, &mut hasher)
                })?;

                Ok(hasher.finish())
            }
        }
    };
}

macro_rules! py_iter_protocol_impl {
    ($name:ident, $data:ident, $T:ty) => {
        #[pyclass(module = "retworkx")]
        pub struct $name {
            pub $data: Vec<$T>,
            iter_pos: usize,
        }

        #[pyproto]
        impl PyIterProtocol for $name {
            fn __iter__(slf: PyRef<Self>) -> Py<$name> {
                slf.into()
            }
            fn __next__(
                mut slf: PyRefMut<Self>,
            ) -> IterNextOutput<$T, &'static str> {
                if slf.iter_pos < slf.$data.len() {
                    let res =
                        IterNextOutput::Yield(slf.$data[slf.iter_pos].clone());
                    slf.iter_pos += 1;
                    res
                } else {
                    IterNextOutput::Return("Ended")
                }
            }
        }
    };
}

macro_rules! custom_hash_map_iter_impl {
    (
        $name:ident, $nameKeys:ident, $nameValues:ident, $nameItems:ident,
        $data:ident, $keys:ident, $values:ident, $items:ident,
        $K:ty, $V:ty, $doc:literal
    ) => {
        #[doc = $doc]
        #[pyclass(module = "retworkx", gc)]
        #[derive(Clone)]
        pub struct $name {
            pub $data: HashMap<$K, $V>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> $name {
                $name {
                    $data: HashMap::new(),
                }
            }

            fn __getstate__(&self) -> HashMap<$K, $V> {
                self.$data.clone()
            }

            fn __setstate__(&mut self, state: HashMap<$K, $V>) {
                self.$data = state;
            }

            fn keys(&self) -> $nameKeys {
                $nameKeys {
                    $keys: self.$data.keys().copied().collect(),
                    iter_pos: 0,
                }
            }

            fn values(&self) -> $nameValues {
                $nameValues {
                    $values: self.$data.values().cloned().collect(),
                    iter_pos: 0,
                }
            }

            fn items(&self) -> $nameItems {
                let items: Vec<($K, $V)> =
                    self.$data.iter().map(|(k, v)| (*k, v.clone())).collect();
                $nameItems {
                    $items: items,
                    iter_pos: 0,
                }
            }
        }

        py_object_protocol_impl!($name, $data);

        #[pyproto]
        impl PySequenceProtocol for $name {
            fn __len__(&self) -> PyResult<usize> {
                Ok(self.$data.len())
            }

            fn __contains__(&self, index: usize) -> PyResult<bool> {
                Ok(self.$data.contains_key(&index))
            }
        }

        #[pyproto]
        impl PyMappingProtocol for $name {
            fn __len__(&self) -> PyResult<usize> {
                Ok(self.$data.len())
            }

            fn __getitem__(&'p self, idx: usize) -> PyResult<$V> {
                match self.$data.get(&idx) {
                    Some(data) => Ok(data.clone()),
                    None => {
                        Err(PyIndexError::new_err("No node found for index"))
                    }
                }
            }
        }

        #[pyproto]
        impl PyIterProtocol for $name {
            fn __iter__(slf: PyRef<Self>) -> $nameKeys {
                $nameKeys {
                    $keys: slf.$data.keys().copied().collect(),
                    iter_pos: 0,
                }
            }
        }

        py_iter_protocol_impl!($nameKeys, $keys, $K);
        py_iter_protocol_impl!($nameValues, $values, $V);
        py_iter_protocol_impl!($nameItems, $items, ($K, $V));
    };
}

custom_hash_map_iter_impl!(
    Pos2DMapping,
    Pos2DMappingKeys,
    Pos2DMappingValues,
    Pos2DMappingItems,
    pos_map,
    pos_keys,
    pos_values,
    pos_items,
    usize,
    [f64; 2],
    "A class representing a mapping of node indices to 2D positions

    This class is equivalent to having a dict of the form::

        {1: [0, 1], 3: [0.5, 1.2]}

    It is used to efficiently represent a retworkx generated 2D layout for a
    graph. It behaves as a drop in replacement for a readonly ``dict``.
    "
);
default_pygc_protocol_impl!(Pos2DMapping);

custom_hash_map_iter_impl!(
    EdgeIndexMap,
    EdgeIndexMapKeys,
    EdgeIndexMapValues,
    EdgeIndexMapItems,
    edge_map,
    edge_map_keys,
    edge_map_values,
    edge_map_items,
    usize,
    (usize, usize, PyObject),
    "A class representing a mapping of edge indices to a tuple of node indices
    and weight/data payload

    This class is equivalent to having a read only dict of the form::

        {1: (0, 1, 'weight'), 3: (2, 3, 1.2)}
    
    It is used to efficiently represent an edge index map for a retworkx
    graph. It behaves as a drop in replacement for a readonly ``dict``.
    "
);

#[pyproto]
impl PyGCProtocol for EdgeIndexMap {
    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        for edge in &self.edge_map {
            visit.call(&edge.1 .2)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.edge_map = HashMap::new();
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
#[derive(Clone)]
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

py_object_protocol_impl!(PathMapping, paths);

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

default_pygc_protocol_impl!(PathMapping);

py_iter_protocol_impl!(PathMappingKeys, path_keys, usize);
py_iter_protocol_impl!(PathMappingValues, path_values, NodeIndices);
py_iter_protocol_impl!(PathMappingItems, path_items, (usize, NodeIndices));

impl PyHash for PathMapping {
    fn hash<H: Hasher>(&self, py: Python, state: &mut H) -> PyResult<()> {
        PyHash::hash(&self.paths, py, state)?;
        Ok(())
    }
}

impl PyEq<PyAny> for PathMapping {
    #[inline]
    fn eq(&self, other: &PyAny, py: Python) -> PyResult<bool> {
        PyEq::eq(&self.paths, &other.to_object(py), py)
    }
}

impl PyDisplay for PathMapping {
    fn str(&self, py: Python) -> PyResult<String> {
        Ok(format!("PathMapping{}", self.paths.str(py)?))
    }
}

custom_hash_map_iter_impl!(
    PathLengthMapping,
    PathLengthMappingKeys,
    PathLengthMappingValues,
    PathLengthMappingItems,
    path_lengths,
    path_lengths_keys,
    path_lengths_values,
    path_lengths_items,
    usize,
    f64,
    "A custom class for the return of path lengths to target nodes

    This class is a container class for the results of functions that
    return a mapping of target nodes and paths. It implements the Python
    mapping protocol. So you can treat the return as a read-only
    mapping/dict. If you want to use it as an iterator you can by
    wrapping it in an ``iter()`` that will yield the results in
    order.

    For example::

        import retworkx

        graph = retworkx.generators.directed_path_graph(5)
        edges = retworkx.dijkstra_shortest_path_lengths(0)
        # Target node access
        third_element = edges[2]
        # Use as iterator
        edges_iter = iter(edges)
        first_target = next(edges_iter)
        first_path = edges[first_target]
        second_target = next(edges_iter)
        second_path = edges[second_target]

    "
);
default_pygc_protocol_impl!(PathLengthMapping);

impl PyHash for PathLengthMapping {
    fn hash<H: Hasher>(&self, py: Python, state: &mut H) -> PyResult<()> {
        PyHash::hash(&self.path_lengths, py, state)?;
        Ok(())
    }
}

impl PyEq<PyAny> for PathLengthMapping {
    #[inline]
    fn eq(&self, other: &PyAny, py: Python) -> PyResult<bool> {
        PyEq::eq(&self.path_lengths, &other.to_object(py), py)
    }
}

impl PyDisplay for PathLengthMapping {
    fn str(&self, py: Python) -> PyResult<String> {
        Ok(format!("PathLengthMapping{}", self.path_lengths.str(py)?))
    }
}

custom_hash_map_iter_impl!(
    CentralityMapping,
    CentralityMappingKeys,
    CentralityMappingValues,
    CentralityMappingItems,
    centralities,
    centralities_keys,
    centralities_values,
    centralities_items,
    usize,
    f64,
    "A custom class for the return of centralities at target nodes

    This class is a container class for the results of functions that
    return a mapping of node index to the betweenness score for that node.
    It implements the Python mapping protocol so you can treat the return
    as a read-only mapping/dict.
    "
);
default_pygc_protocol_impl!(CentralityMapping);

custom_hash_map_iter_impl!(
    NodesCountMapping,
    NodesCountMappingKeys,
    NodesCountMappingValues,
    NodesCountMappingItems,
    map,
    map_keys,
    map_values,
    map_items,
    usize,
    BigUint,
    "A custom class for the return of number path lengths to target nodes

    This class is a container class for the results of functions that
    return a mapping of target nodes and counts. It implements the Python
    mapping protocol. So you can treat the return as a read-only
    mapping/dict. If you want to use it as an iterator you can by
    wrapping it in an ``iter()`` that will yield the results in
    order.

    For example::
    
        import retworkx
    
        graph = retworkx.generators.directed_path_graph(5)
        edges = retworkx.num_shortest_paths_unweighted(0)
        # Target node access
        third_element = edges[2]
        # Use as iterator
        edges_iter = iter(edges)
        first_target = next(edges_iter)
        first_path = edges[first_target]
        second_target = next(edges_iter)
        second_path = edges[second_target]
    "
);
default_pygc_protocol_impl!(NodesCountMapping);

custom_hash_map_iter_impl!(
    AllPairsPathLengthMapping,
    AllPairsPathLengthMappingKeys,
    AllPairsPathLengthMappingValues,
    AllPairsPathLengthMappingItems,
    path_lengths,
    path_lengths_keys,
    path_lengths_values,
    path_lengths_items,
    usize,
    PathLengthMapping,
    "A custom class for the return of path lengths to target nodes from all nodes

    This class is a container class for the results of functions that
    return a mapping of target nodes and paths from all nodes. It implements
    the Python mapping protocol. So you can treat the return as a read-only
    mapping/dict.

    For example::

        import retworkx

        graph = retworkx.generators.directed_path_graph(5)
        edges = retworkx.all_pairs_dijkstra_shortest_path_lengths(graph)
        # Target node access
        third_node_shortest_path_lengths = edges[2]

    "
);
default_pygc_protocol_impl!(AllPairsPathLengthMapping);

custom_hash_map_iter_impl!(
    AllPairsPathMapping,
    AllPairsPathMappingKeys,
    AllPairsPathMappingValues,
    AllPairsPathMappingItems,
    paths,
    path_keys,
    path_values,
    path_items,
    usize,
    PathMapping,
    "A custom class for the return of paths to target nodes from all nodes

    This class is a container class for the results of functions that
    return a mapping of target nodes and paths from all nodes. It implements
    the Python mapping protocol. So you can treat the return as a read-only
    mapping/dict.

    For example::

        import retworkx

        graph = retworkx.generators.directed_path_graph(5)
        edges = retworkx.all_pairs_dijkstra_shortest_paths(graph)
        # Target node access
        third_node_shortest_paths = edges[2]

    "
);
default_pygc_protocol_impl!(AllPairsPathMapping);

custom_hash_map_iter_impl!(
    NodeMap,
    NodeMapKeys,
    NodeMapValues,
    NodeMapItems,
    node_map,
    node_map_keys,
    node_map_values,
    node_map_items,
    usize,
    usize,
    "A class representing a mapping of node indices to node indices

     This class is equivalent to having a dict of the form::

         {1: 0, 3: 1}

    Unlike a dict though this class is unordered and multiple NodeMap
    objects with the same contents might yield a different order when
    iterated over. If a consistent order is required you should sort
    the object.
    "
);
default_pygc_protocol_impl!(NodeMap);
