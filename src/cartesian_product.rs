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

use crate::iterators::ProductNodeMap;
use crate::{digraph, graph, StablePyGraph};

use hashbrown::HashMap;

use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences};
use petgraph::{algo, EdgeType};

use pyo3::prelude::*;
use pyo3::Python;

fn cartesian_product<Ty: EdgeType>(
    py: Python,
    first: &StablePyGraph<Ty>,
    second: &StablePyGraph<Ty>,
) -> (StablePyGraph<Ty>, ProductNodeMap) {
    let mut final_graph = StablePyGraph::<Ty>::with_capacity(
        first.node_count() * second.node_count(),
        first.node_count() * second.edge_count()
            + first.edge_count() * second.node_count(),
    );

    let mut hash_nodes =
        HashMap::with_capacity(first.node_count() * second.node_count());

    for (x, weight_x) in first.node_references() {
        for (y, weight_y) in second.node_references() {
            let n0 = final_graph.add_node((weight_x, weight_y).into_py(py));
            hash_nodes.insert((x, y), n0);
        }
    }

    for edge_first in first.edge_references() {
        for node_second in second.node_indices() {
            let source = hash_nodes[&(edge_first.source(), node_second)];
            let target = hash_nodes[&(edge_first.target(), node_second)];

            final_graph.add_edge(
                source,
                target,
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

    let out_node_map = ProductNodeMap {
        node_map: hash_nodes
            .into_iter()
            .map(|((x, y), n)| ((x.index(), y.index()), n.index()))
            .collect(),
    };

    (final_graph, out_node_map)
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
///   graph_product, _ = retworkx.graph_cartesian_product(graph_1, graph_2)
///   mpl_draw(graph_product)
#[pyfunction()]
#[pyo3(text_signature = "(first, second, /)")]
fn graph_cartesian_product(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
) -> (graph::PyGraph, ProductNodeMap) {
    let (out_graph, out_node_map) =
        cartesian_product(py, &first.graph, &second.graph);

    (
        graph::PyGraph {
            graph: out_graph,
            multigraph: true,
            node_removed: false,
        },
        out_node_map,
    )
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
///   graph_product, _ = retworkx.digraph_cartesian_product(graph_1, graph_2)
///   mpl_draw(graph_product)
#[pyfunction()]
#[pyo3(text_signature = "(first, second, /)")]
fn digraph_cartesian_product(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
) -> (digraph::PyDiGraph, ProductNodeMap) {
    let (out_graph, out_node_map) =
        cartesian_product(py, &first.graph, &second.graph);

    (
        digraph::PyDiGraph {
            graph: out_graph,
            cycle_state: algo::DfsSpace::default(),
            check_cycle: false,
            node_removed: false,
            multigraph: true,
        },
        out_node_map,
    )
}
