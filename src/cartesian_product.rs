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

use hashbrown::HashMap;

use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::{algo, EdgeType};

use pyo3::prelude::*;
use pyo3::Python;

fn cartesian_product<Ty: EdgeType>(
    py: Python,
    first: &StablePyGraph<Ty>,
    second: &StablePyGraph<Ty>,
) -> PyResult<StablePyGraph<Ty>> {
    let mut final_graph = StablePyGraph::<Ty>::with_capacity(
        first.node_count() * second.node_count(),
        first.node_count() * second.edge_count()
            + first.edge_count() * second.node_count(),
    );

    let mut hash_nodes =
        HashMap::with_capacity(first.node_count() * second.node_count());

    let nodes_first = first.node_indices();
    let nodes_second = second.node_indices();

    let cross =
        nodes_first.flat_map(|x| nodes_second.clone().map(move |y| (x, y)));

    for (x, y) in cross {
        let weight_x = &first[x];
        let weight_y = &second[y];
        let n0 = final_graph.add_node((weight_x, weight_y).into_py(py));
        hash_nodes.insert((x, y), n0);
    }

    for edge_first in first.edge_references() {
        for node_second in second.node_indices() {
            let source =
                hash_nodes.get(&(edge_first.source(), node_second)).unwrap();
            let target =
                hash_nodes.get(&(edge_first.target(), node_second)).unwrap();

            final_graph.add_edge(
                *source,
                *target,
                edge_first.weight().clone_ref(py),
            );
        }
    }

    for node_first in first.node_indices() {
        for edge_second in second.edge_references() {
            let source =
                hash_nodes.get(&(node_first, edge_second.source())).unwrap();
            let target =
                hash_nodes.get(&(node_first, edge_second.target())).unwrap();

            final_graph.add_edge(
                *source,
                *target,
                edge_second.weight().clone_ref(py),
            );
        }
    }

    Ok(final_graph)
}

/// Return a new PyGraph by forming the cartesian product from two input
/// PyGraph objects
///
/// :param PyGraph first: The first undirected graph object
/// :param PyGraph second: The second undirected graph object
///
/// :returns: A new PyGraph object that is the cartesian product of ``first``
///     and ``second``.
///     It's worth noting the weight/data payload objects are
///     passed by reference from ``first`` and ``second`` to this new object.
/// :rtype: PyGraph
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph_1 = retworkx.generators.path_graph(2)
///   graph_2 = retworkx.generators.path_graph(3)
///   graph_product = retworkx.graph_cartesian_product(graph_1, graph_2)
///   mpl_draw(graph_product)
#[pyfunction()]
#[pyo3(text_signature = "(first, second, /)")]
fn graph_cartesian_product(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
) -> PyResult<graph::PyGraph> {
    let out_graph = cartesian_product(py, &first.graph, &second.graph)?;

    Ok(graph::PyGraph {
        graph: out_graph,
        multigraph: true,
        node_removed: false,
    })
}

/// Return a new PyDiGraph by forming the cartesian product from two input
/// PyDiGraph objects
///
/// :param PyDiGraph first: The first undirected graph object
/// :param PyDiGraph second: The second undirected graph object
///
/// :returns: A new PyDiGraph object that is the cartesian product of ``first``
///     and ``second``.
///     It's worth noting the weight/data payload objects are
///     passed by reference from ``first`` and ``second`` to this new object.
/// :rtype: PyDiGraph
///
/// .. jupyter-execute::
///
///   import retworkx.generators
///   from retworkx.visualization import mpl_draw
///
///   graph_1 = retworkx.generators.directed_path_graph(2)
///   graph_2 = retworkx.generators.directed_path_graph(3)
///   graph_product = retworkx.digraph_cartesian_product(graph_1, graph_2)
///   mpl_draw(graph_product)
#[pyfunction()]
#[pyo3(text_signature = "(first, second, /)")]
fn digraph_cartesian_product(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
) -> PyResult<digraph::PyDiGraph> {
    let out_graph = cartesian_product(py, &first.graph, &second.graph)?;

    Ok(digraph::PyDiGraph {
        graph: out_graph,
        cycle_state: algo::DfsSpace::default(),
        check_cycle: false,
        node_removed: false,
        multigraph: true,
    })
}
