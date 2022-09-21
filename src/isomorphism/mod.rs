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

mod vf2;

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
///     graph_a = rustworkx.PyDiGraph()
///     graph_b = rustworkx.PyDiGraph()
///     rustworkx.is_isomorphic(graph_a, graph_b,
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
/// :param int call_limit: An optional bound on the number of states that VF2 algorithm
///     visits while searching for a solution. If it exceeds this limit, the algorithm
///     will stop and return ``False``.
///
/// :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
///     not.
/// :rtype: bool
#[pyfunction(id_order = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None,
                    id_order=True, call_limit=None)"
)]
pub fn digraph_is_isomorphic(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    call_limit: Option<usize>,
) -> PyResult<bool> {
    vf2::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        node_matcher,
        edge_matcher,
        id_order,
        Ordering::Equal,
        true,
        call_limit,
    )
}

/// Determine if 2 undirected graphs are isomorphic
///
/// This checks if 2 graphs are isomorphic both structurally and also
/// comparing the node data and edge data using the provided matcher functions.
/// The matcher function takes in 2 data objects and will compare them. A simple
/// example that checks if they're just equal would be::
///
///     graph_a = rustworkx.PyGraph()
///     graph_b = rustworkx.PyGraph()
///     rustworkx.is_isomorphic(graph_a, graph_b,
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
/// :param int call_limit: An optional bound on the number of states that VF2 algorithm
///     visits while searching for a solution. If it exceeds this limit, the algorithm
///     will stop and return ``False``.
///
/// :returns: ``True`` if the 2 graphs are isomorphic ``False`` if they are
///     not.
/// :rtype: bool
#[pyfunction(id_order = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None,
                    id_order=True, call_limit=None)"
)]
pub fn graph_is_isomorphic(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    call_limit: Option<usize>,
) -> PyResult<bool> {
    vf2::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        node_matcher,
        edge_matcher,
        id_order,
        Ordering::Equal,
        true,
        call_limit,
    )
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
///     graph_a = rustworkx.PyDiGraph()
///     graph_b = rustworkx.PyDiGraph()
///     rustworkx.is_subgraph_isomorphic(graph_a, graph_b,
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
/// :param int call_limit: An optional bound on the number of states that VF2 algorithm
///     visits while searching for a solution. If it exceeds this limit, the algorithm
///     will stop and return ``False``.
///
/// :returns: ``True`` if there is a subgraph of `first` isomorphic to `second`,
///     ``False`` if there is not.
/// :rtype: bool
#[pyfunction(id_order = "false", induced = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None,
                    id_order=False, induced=True, call_limit=None)"
)]
pub fn digraph_is_subgraph_isomorphic(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    induced: bool,
    call_limit: Option<usize>,
) -> PyResult<bool> {
    vf2::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        node_matcher,
        edge_matcher,
        id_order,
        Ordering::Greater,
        induced,
        call_limit,
    )
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
///     graph_a = rustworkx.PyGraph()
///     graph_b = rustworkx.PyGraph()
///     rustworkx.is_subgraph_isomorphic(graph_a, graph_b,
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
/// :param int call_limit: An optional bound on the number of states that VF2 algorithm
///     visits while searching for a solution. If it exceeds this limit, the algorithm
///     will stop and return ``False``.
///
/// :returns: ``True`` if there is a subgraph of `first` isomorphic to `second`,
///     ``False`` if there is not.
/// :rtype: bool
#[pyfunction(id_order = "false", induced = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None,
                    id_order=False, induced=True, call_limit=None)"
)]
pub fn graph_is_subgraph_isomorphic(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    induced: bool,
    call_limit: Option<usize>,
) -> PyResult<bool> {
    vf2::is_isomorphic(
        py,
        &first.graph,
        &second.graph,
        node_matcher,
        edge_matcher,
        id_order,
        Ordering::Greater,
        induced,
        call_limit,
    )
}

/// Return an iterator over all vf2 mappings between two :class:`~rustworkx.PyDiGraph` objects
///
/// This funcion will run the vf2 algorithm used from
/// :func:`~rustworkx.is_isomorphic` and :func:`~rustworkx.is_subgraph_isomorphic`
/// but instead of returning a boolean it will return an iterator over all possible
/// mapping of node ids found from ``first`` to ``second``. If the graphs are not
/// isomorphic then the iterator will be empty. A simple example that retrieves
/// one mapping would be::
///
///         graph_a = rustworkx.generators.directed_path_graph(3)
///         graph_b = rustworkx.generators.direccted_path_graph(2)
///         vf2 = rustworkx.digraph_vf2_mapping(graph_a, graph_b, subgraph=True)
///         try:
///             mapping = next(vf2)
///         except StopIteration:
///             pass
///
///
/// :param PyDiGraph first: The first graph to find the mapping for
/// :param PyDiGraph second: The second graph to find the mapping for
/// :param node_matcher: An optional python callable object that takes 2
///     positional arguments, one for each node data object in either graph.
///     If the return of this function evaluates to True then the nodes
///     passed to it are viewed as matching.
/// :param edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are viewed
///     as matching.
/// :param bool id_order: If set to ``False`` this function will use a
///     heuristic matching order based on [VF2]_ paper. Otherwise it will
///     default to matching the nodes in order specified by their ids.
/// :param bool subgraph: If set to ``True`` the function will return the
///     subgraph isomorphic found between the graphs.
/// :param bool induced: If set to ``True`` this function will check the existence
///     of a node-induced subgraph of first isomorphic to second graph.
///     Default: ``True``.
/// :param int call_limit: An optional bound on the number of states that VF2 algorithm
///     visits while searching for a solution. If it exceeds this limit, the algorithm
///     will stop.
///
/// :returns: An iterator over dicitonaries of node indices from ``first`` to node
///     indices in ``second`` representing the mapping found.
/// :rtype: Iterable[NodeMap]
#[pyfunction(id_order = "true", subgraph = "false", induced = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None,
                    id_order=True, subgraph=False, induced=True, call_limit=None)"
)]
pub fn digraph_vf2_mapping(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    subgraph: bool,
    induced: bool,
    call_limit: Option<usize>,
) -> vf2::DiGraphVf2Mapping {
    let ordering = if subgraph {
        Ordering::Greater
    } else {
        Ordering::Equal
    };

    vf2::DiGraphVf2Mapping::new(
        py,
        &first.graph,
        &second.graph,
        node_matcher,
        edge_matcher,
        id_order,
        ordering,
        induced,
        call_limit,
    )
}

/// Return an iterator over all vf2 mappings between two :class:`~rustworkx.PyGraph` objects
///
/// This funcion will run the vf2 algorithm used from
/// :func:`~rustworkx.is_isomorphic` and :func:`~rustworkx.is_subgraph_isomorphic`
/// but instead of returning a boolean it will return an iterator over all possible
/// mapping of node ids found from ``first`` to ``second``. If the graphs are not
/// isomorphic then the iterator will be empty. A simple example that retrieves
/// one mapping would be::
///
///         graph_a = rustworkx.generators.path_graph(3)
///         graph_b = rustworkx.generators.path_graph(2)
///         vf2 = rustworkx.graph_vf2_mapping(graph_a, graph_b, subgraph=True)
///         try:
///             mapping = next(vf2)
///         except StopIteration:
///             pass
///
/// :param PyGraph first: The first graph to find the mapping for
/// :param PyGraph second: The second graph to find the mapping for
/// :param node_matcher: An optional python callable object that takes 2
///     positional arguments, one for each node data object in either graph.
///     If the return of this function evaluates to True then the nodes
///     passed to it are viewed as matching.
/// :param edge_matcher: A python callable object that takes 2 positional
///     one for each edge data object. If the return of this
///     function evaluates to True then the edges passed to it are viewed
///     as matching.
/// :param bool id_order: If set to ``False`` this function will use a
///     heuristic matching order based on [VF2]_ paper. Otherwise it will
///     default to matching the nodes in order specified by their ids.
/// :param bool subgraph: If set to ``True`` the function will return the
///     subgraph isomorphic found between the graphs.
/// :param bool induced: If set to ``True`` this function will check the existence
///     of a node-induced subgraph of first isomorphic to second graph.
///     Default: ``True``.
/// :param int call_limit: An optional bound on the number of states that VF2 algorithm
///     visits while searching for a solution. If it exceeds this limit, the algorithm
///     will stop. Default: ``None``.
///
/// :returns: An iterator over dicitonaries of node indices from ``first`` to node
///     indices in ``second`` representing the mapping found.
/// :rtype: Iterable[NodeMap]
#[pyfunction(id_order = "true", subgraph = "false", induced = "true")]
#[pyo3(
    text_signature = "(first, second, /, node_matcher=None, edge_matcher=None,
                    id_order=True, subgraph=False, induced=True, call_limit=None)"
)]
pub fn graph_vf2_mapping(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
    node_matcher: Option<PyObject>,
    edge_matcher: Option<PyObject>,
    id_order: bool,
    subgraph: bool,
    induced: bool,
    call_limit: Option<usize>,
) -> vf2::GraphVf2Mapping {
    let ordering = if subgraph {
        Ordering::Greater
    } else {
        Ordering::Equal
    };

    vf2::GraphVf2Mapping::new(
        py,
        &first.graph,
        &second.graph,
        node_matcher,
        edge_matcher,
        id_order,
        ordering,
        induced,
        call_limit,
    )
}
