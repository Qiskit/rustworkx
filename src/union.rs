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

use crate::{digraph, find_node_by_weight, graph, StablePyGraph};

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};
use petgraph::{algo, EdgeType};

use pyo3::prelude::*;
use pyo3::Python;

#[derive(Copy, Clone)]
enum Entry<T> {
    Merged(T),
    Added(T),
    None,
}

fn extract<T>(x: Entry<T>) -> T {
    match x {
        Entry::Merged(val) => val,
        Entry::Added(val) => val,
        Entry::None => panic!("Unexpected internal error: called `Entry::extract()` on a `None` value. Please file an issue at https://github.com/Qiskit/retworkx/issues/new/choose with the details on how you encountered this."),
    }
}

fn union<Ty: EdgeType>(
    py: Python,
    first: &StablePyGraph<Ty>,
    second: &StablePyGraph<Ty>,
    merge_nodes: bool,
    merge_edges: bool,
) -> PyResult<StablePyGraph<Ty>> {
    let mut out_graph = first.clone();

    let mut node_map: Vec<Entry<NodeIndex>> =
        vec![Entry::None; second.node_bound()];
    for node in second.node_indices() {
        let weight = &second[node];
        if merge_nodes {
            if let Some(index) = find_node_by_weight(py, first, weight)? {
                node_map[node.index()] = Entry::Merged(index);
                continue;
            }
        }

        let index = out_graph.add_node(weight.clone_ref(py));
        node_map[node.index()] = Entry::Added(index);
    }

    let weights_equal = |a: &PyObject, b: &PyObject| -> PyResult<bool> {
        a.as_ref(py)
            .rich_compare(b, pyo3::basic::CompareOp::Eq)?
            .is_true()
    };

    for edge in second.edge_references() {
        let source = edge.source().index();
        let target = edge.target().index();
        let new_weight = edge.weight();

        let mut found = false;
        if merge_edges {
            // if both endpoints were merged,
            // check if need to skip the edge as well.
            if let (Entry::Merged(new_source), Entry::Merged(new_target)) =
                (node_map[source], node_map[target])
            {
                for edge in first.edges(new_source) {
                    if edge.target() == new_target
                        && weights_equal(new_weight, edge.weight())?
                    {
                        found = true;
                        break;
                    }
                }
            }
        }

        if !found {
            let new_source = extract(node_map[source]);
            let new_target = extract(node_map[target]);
            out_graph.add_edge(
                new_source,
                new_target,
                new_weight.clone_ref(py),
            );
        }
    }

    Ok(out_graph)
}

/// Return a new PyGraph by forming a union from two input PyGraph objects
///
/// The algorithm in this function operates in three phases:
///
/// 1. Add all the nodes from  ``second`` into ``first``. operates in
///    :math:`\mathcal{O}(n_2)`, with :math:`n_2` being number of nodes in
///    ``second``.
/// 2. Merge nodes from ``second`` over ``first`` given that:
///
///    - The ``merge_nodes`` is ``True``. operates in :math:`\mathcal{O}(n_1 n_2)`,
///      with :math:`n_1` being the number of nodes in ``first`` and :math:`n_2`
///      the number of nodes in ``second``
///    - The respective node in ``second`` and ``first`` share the same
///      weight/data payload.
///
/// 3. Adds all the edges from ``second`` to ``first``. If the ``merge_edges``
///    parameter is ``True`` and the respective edge in ``second`` and
///    ``first`` share the same weight/data payload they will be merged together.
///
/// :param PyGraph first: The first undirected graph object
/// :param PyGraph second: The second undirected graph object
/// :param bool merge_nodes: If set to ``True`` nodes will be merged between
///     ``second`` and ``first`` if the weights are equal. Default: ``False``.
/// :param bool merge_edges: If set to ``True`` edges will be merged between
///     ``second`` and ``first`` if the weights are equal. Default: ``False``.
///
/// :returns: A new PyGraph object that is the union of ``second`` and
///     ``first``. It's worth noting the weight/data payload objects are
///     passed by reference from ``first`` and ``second`` to this new object.
/// :rtype: PyGraph
#[pyfunction(merge_nodes = false, merge_edges = false)]
#[pyo3(
    text_signature = "(first, second, /, merge_nodes=False, merge_edges=False)"
)]
fn graph_union(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
    merge_nodes: bool,
    merge_edges: bool,
) -> PyResult<graph::PyGraph> {
    let out_graph =
        union(py, &first.graph, &second.graph, merge_nodes, merge_edges)?;

    Ok(graph::PyGraph {
        graph: out_graph,
        node_removed: first.node_removed,
        multigraph: true,
    })
}

/// Return a new PyDiGraph by forming a union from two input PyDiGraph objects
///
/// The algorithm in this function operates in three phases:
///
/// 1. Add all the nodes from  ``second`` into ``first``. operates in
///    :math:`\mathcal{O}(n_2)`, with :math:`n_2` being number of nodes in
///    ``second``.
/// 2. Merge nodes from ``second`` over ``first`` given that:
///
///    - The ``merge_nodes`` is ``True``. operates in :math:`\mathcal{O}(n_1 n_2)`,
///      with :math:`n_1` being the number of nodes in ``first`` and :math:`n_2`
///      the number of nodes in ``second``
///    - The respective node in ``second`` and ``first`` share the same
///      weight/data payload.
///
/// 3. Adds all the edges from ``second`` to ``first``. If the ``merge_edges``
///    parameter is ``True`` and the respective edge in ``second`` and
///    ``first`` share the same weight/data payload they will be merged together.
///
/// :param PyDiGraph first: The first directed graph object
/// :param PyDiGraph second: The second directed graph object
/// :param bool merge_nodes: If set to ``True`` nodes will be merged between
///     ``second`` and ``first`` if the weights are equal. Default: ``False``.
/// :param bool merge_edges: If set to ``True`` edges will be merged between
///     ``second`` and ``first`` if the weights are equal. Default: ``False``.
///
/// :returns: A new PyDiGraph object that is the union of ``second`` and
///     ``first``. It's worth noting the weight/data payload objects are
///     passed by reference from ``first`` and ``second`` to this new object.
/// :rtype: PyDiGraph
#[pyfunction(merge_nodes = false, merge_edges = false)]
#[pyo3(
    text_signature = "(first, second, /, merge_nodes=False, merge_edges=False)"
)]
fn digraph_union(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
    merge_nodes: bool,
    merge_edges: bool,
) -> PyResult<digraph::PyDiGraph> {
    let out_graph =
        union(py, &first.graph, &second.graph, merge_nodes, merge_edges)?;

    Ok(digraph::PyDiGraph {
        graph: out_graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: first.node_removed,
        multigraph: true,
    })
}
