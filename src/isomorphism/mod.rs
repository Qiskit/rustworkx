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

#![allow(clippy::float_cmp)]
#![allow(clippy::module_inception)]

mod isomorphism;

use crate::{digraph, graph};

use std::cmp::Ordering;

use pyo3::prelude::*;
use pyo3::Python;

/// Determine if 2 directed graphs are isomorphic
///
/// This checks if 2 graphs are isomorphic both structurally and also
/// comparing the node data and edge data using the provided matcher functions.
/// The matcher function takes in 2 data objects and will compare them. A simple
/// example that checks if they're just equal would be::
///
///     graph_a = retworkx.PyDiGraph()
///     graph_b = retworkx.PyDiGraph()
///     retworkx.is_isomorphic(graph_a, graph_b,
///                            lambda x, y: x == y)
///
/// .. note::
///
///     For better performance on large graphs, consider setting `id_order=False`.
///
/// :param PyDiGraph first: The first graph to compare
/// :param PyDiGraph second: The second graph to compare
/// :param callable node_matcher: A python callable object that takes 2 positional
///     one for each node data object. If the return of this
///     function evaluates to True then the nodes passed to it are vieded
///     as matching.
/// :param callable edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are vieded
///     as matching.
/// :param bool id_order: If set to ``False`` this function will use a
///     heuristic matching order based on [VF2]_ paper. Otherwise it will
///     default to matching the nodes in order specified by their ids.
///
/// :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
///     not.
/// :rtype: bool
#[pyfunction(id_order = "true")]
#[pyo3(
    text_signature = "(first, second, node_matcher=None, edge_matcher=None, id_order=True, /)"
)]
fn digraph_is_isomorphic(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
) -> PyResult<bool> {
    let compare_nodes = node_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let compare_edges = edge_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let res = isomorphism::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        compare_nodes,
        compare_edges,
        id_order,
        Ordering::Equal,
        true,
    )?;
    Ok(res)
}

/// Determine if 2 undirected graphs are isomorphic
///
/// This checks if 2 graphs are isomorphic both structurally and also
/// comparing the node data and edge data using the provided matcher functions.
/// The matcher function takes in 2 data objects and will compare them. A simple
/// example that checks if they're just equal would be::
///
///     graph_a = retworkx.PyGraph()
///     graph_b = retworkx.PyGraph()
///     retworkx.is_isomorphic(graph_a, graph_b,
///                            lambda x, y: x == y)
///
/// .. note::
///
///     For better performance on large graphs, consider setting `id_order=False`.
///
/// :param PyGraph first: The first graph to compare
/// :param PyGraph second: The second graph to compare
/// :param callable node_matcher: A python callable object that takes 2 positional
///     one for each node data object. If the return of this
///     function evaluates to True then the nodes passed to it are vieded
///     as matching.
/// :param callable edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are vieded
///     as matching.
/// :param bool (default=True) id_order:  If set to true, the algorithm matches the
///     nodes in order specified by their ids. Otherwise, it uses a heuristic
///     matching order based in [VF2]_ paper.
///
/// :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
///     not.
/// :rtype: bool
#[pyfunction(id_order = "true")]
#[pyo3(
    text_signature = "(first, second, node_matcher=None, edge_matcher=None, id_order=True, /)"
)]
fn graph_is_isomorphic(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
) -> PyResult<bool> {
    let compare_nodes = node_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let compare_edges = edge_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let res = isomorphism::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        compare_nodes,
        compare_edges,
        id_order,
        Ordering::Equal,
        true,
    )?;
    Ok(res)
}

/// Determine if 2 directed graphs are subgraph - isomorphic
///
/// This checks if 2 graphs are subgraph isomorphic both structurally and also
/// comparing the node data and edge data using the provided matcher functions.
/// The matcher function takes in 2 data objects and will compare them.
/// Since there is an ambiguity in the term 'subgraph', do note that we check
/// for an node-induced subgraph if argument `induced` is set to `True`. If it is
/// set to `False`, we check for a non induced subgraph, meaning the second graph
/// can have fewer edges than the subgraph of the first. By default it's `True`. A
/// simple example that checks if they're just equal would be::
///
///     graph_a = retworkx.PyDiGraph()
///     graph_b = retworkx.PyDiGraph()
///     retworkx.is_subgraph_isomorphic(graph_a, graph_b,
///                                     lambda x, y: x == y)
///
/// .. note::
///
///     For better performance on large graphs, consider setting `id_order=False`.
///
/// :param PyDiGraph first: The first graph to compare
/// :param PyDiGraph second: The second graph to compare
/// :param callable node_matcher: A python callable object that takes 2 positional
///     one for each node data object. If the return of this
///     function evaluates to True then the nodes passed to it are vieded
///     as matching.
/// :param callable edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are vieded
///     as matching.
/// :param bool id_order: If set to ``True`` this function will match the nodes
///     in order specified by their ids. Otherwise it will default to a heuristic
///     matching order based on [VF2]_ paper.
/// :param bool induced: If set to ``True`` this function will check the existence
///     of a node-induced subgraph of first isomorphic to second graph.
///     Default: ``True``.
///
/// :returns: ``True`` if there is a subgraph of `first` isomorphic to `second`,
///     ``False`` if there is not.
/// :rtype: bool
#[pyfunction(id_order = "false", induced = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None, id_order=False, induced=True)"
)]
fn digraph_is_subgraph_isomorphic(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    induced: bool,
) -> PyResult<bool> {
    let compare_nodes = node_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let compare_edges = edge_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let res = isomorphism::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        compare_nodes,
        compare_edges,
        id_order,
        Ordering::Greater,
        induced,
    )?;
    Ok(res)
}

/// Determine if 2 undirected graphs are subgraph - isomorphic
///
/// This checks if 2 graphs are subgraph isomorphic both structurally and also
/// comparing the node data and edge data using the provided matcher functions.
/// The matcher function takes in 2 data objects and will compare them.
/// Since there is an ambiguity in the term 'subgraph', do note that we check
/// for an node-induced subgraph if argument `induced` is set to `True`. If it is
/// set to `False`, we check for a non induced subgraph, meaning the second graph
/// can have fewer edges than the subgraph of the first. By default it's `True`. A
/// simple example that checks if they're just equal would be::
///
///     graph_a = retworkx.PyGraph()
///     graph_b = retworkx.PyGraph()
///     retworkx.is_subgraph_isomorphic(graph_a, graph_b,
///                                     lambda x, y: x == y)
///
/// .. note::
///
///     For better performance on large graphs, consider setting `id_order=False`.
///
/// :param PyGraph first: The first graph to compare
/// :param PyGraph second: The second graph to compare
/// :param callable node_matcher: A python callable object that takes 2 positional
///     one for each node data object. If the return of this
///     function evaluates to True then the nodes passed to it are vieded
///     as matching.
/// :param callable edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are vieded
///     as matching.
/// :param bool id_order: If set to ``True`` this function will match the nodes
///     in order specified by their ids. Otherwise it will default to a heuristic
///     matching order based on [VF2]_ paper.
/// :param bool induced: If set to ``True`` this function will check the existence
///     of a node-induced subgraph of first isomorphic to second graph.
///     Default: ``True``.
///
/// :returns: ``True`` if there is a subgraph of `first` isomorphic to `second`,
///     ``False`` if there is not.
/// :rtype: bool
#[pyfunction(id_order = "false", induced = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None, id_order=False, induced=True)"
)]
fn graph_is_subgraph_isomorphic(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    induced: bool,
) -> PyResult<bool> {
    let compare_nodes = node_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let compare_edges = edge_matcher.map(|f| {
        move |a: &PyObject, b: &PyObject| -> PyResult<bool> {
            let res = f.call1(py, (a, b))?;
            Ok(res.is_true(py).unwrap())
        }
    });

    let res = isomorphism::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        compare_nodes,
        compare_edges,
        id_order,
        Ordering::Greater,
        induced,
    )?;
    Ok(res)
}
