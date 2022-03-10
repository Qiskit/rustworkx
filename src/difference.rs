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

use crate::{digraph, graph, StablePyGraph};

use hashbrown::HashSet;

use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::{algo, EdgeType};

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::Python;

fn difference<Ty: EdgeType>(
    py: Python,
    first: &StablePyGraph<Ty>,
    second: &StablePyGraph<Ty>,
) -> PyResult<StablePyGraph<Ty>> {
    let indexes_first = first.node_indices().collect::<HashSet<_>>();
    let indexes_second = second.node_indices().collect::<HashSet<_>>();

    if indexes_first != indexes_second {
        return Err(PyIndexError::new_err(
            "Node sets of the graphs should be equal",
        ));
    }

    let mut final_graph = StablePyGraph::<Ty>::with_capacity(
        first.node_count(),
        first.edge_count() - second.edge_count(),
    );

    for node in first.node_indices() {
        let weight = &first[node];
        final_graph.add_node(weight.clone_ref(py));
    }

    for e in first.edge_references() {
        let has_edge = second.find_edge(e.source(), e.target());

        match has_edge {
            Some(_x) => continue,
            None => final_graph.add_edge(e.source(), e.target(), e.weight().clone_ref(py)),
        };
    }

    Ok(final_graph)
}

/// Return a new PyGraph that is the difference from two input
/// PyGraph objects
///
/// :param PyGraph first: The first undirected graph object
/// :param PyGraph second: The second undirected graph object
///
/// :returns: A new PyGraph object that is the difference of ``first``
///     and ``second``. It's worth noting the weight/data payload objects are
///     passed by reference from ``first`` graph to this new object.
///
/// :rtype: :class:`~retworkx.PyGraph`
#[pyfunction()]
#[pyo3(text_signature = "(first, second, /)")]
fn graph_difference(py: Python, first: &graph::PyGraph, second: &graph::PyGraph) -> PyResult<graph::PyGraph> {
    let out_graph = difference(py, &first.graph, &second.graph)?;

    Ok(graph::PyGraph {
        graph: out_graph,
        multigraph: true,
        node_removed: false,
    })
}

/// Return a new PyDiGraph that is the difference from two input
/// PyGraph objects
///
/// :param PyGraph first: The first undirected graph object
/// :param PyGraph second: The second undirected graph object
///
/// :returns: A new PyDiGraph object that is the difference of ``first``
///     and ``second``. It's worth noting the weight/data payload objects are
///     passed by reference from ``first`` graph to this new object.
///
/// :rtype: :class:`~retworkx.PyDiGraph`
#[pyfunction()]
#[pyo3(text_signature = "(first, second, /)")]
fn digraph_difference(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
) -> PyResult<digraph::PyDiGraph> {
    let out_graph = difference(py, &first.graph, &second.graph)?;

    Ok(digraph::PyDiGraph {
        graph: out_graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
    })
}
