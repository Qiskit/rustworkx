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

use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::{algo, EdgeType};

use pyo3::prelude::*;
use pyo3::Python;

fn tensor_product<Ty: EdgeType>(
    py: Python,
    first: &StablePyGraph<Ty>,
    second: &StablePyGraph<Ty>,
    undirected: bool,
) -> (StablePyGraph<Ty>, ProductNodeMap) {
    let num_nodes = first.node_count() * second.node_count();
    let mut num_edges = first.edge_count() * second.edge_count();

    if undirected {
        num_edges *= 2;
    }

    let mut final_graph = StablePyGraph::<Ty>::with_capacity(num_nodes, num_edges);
    let mut hash_nodes = HashMap::with_capacity(num_nodes);

    for x in first.node_indices() {
        for y in second.node_indices() {
            let weight_x = &first[x];
            let weight_y = &second[y];
            let n0 = final_graph.add_node((weight_x, weight_y).into_py(py));
            hash_nodes.insert((x, y), n0);
        }
    }

    for edge_first in first.edge_references() {
        for edge_second in second.edge_references() {
            let source = hash_nodes
                .get(&(edge_first.source(), edge_second.source()))
                .unwrap();

            let target = hash_nodes
                .get(&(edge_first.target(), edge_second.target()))
                .unwrap();

            final_graph.add_edge(
                *source,
                *target,
                (
                    edge_first.weight().clone_ref(py),
                    edge_second.weight().clone_ref(py),
                )
                    .into_py(py),
            );
        }
    }
    if undirected {
        for edge_first in first.edge_references() {
            for edge_second in second.edge_references() {
                if edge_first.source() == edge_first.target()
                    || edge_second.source() == edge_second.target()
                {
                    continue;
                }

                let source = hash_nodes
                    .get(&(edge_first.source(), edge_second.target()))
                    .unwrap();

                let target = hash_nodes
                    .get(&(edge_first.target(), edge_second.source()))
                    .unwrap();

                final_graph.add_edge(
                    *source,
                    *target,
                    (
                        edge_first.weight().clone_ref(py),
                        edge_second.weight().clone_ref(py),
                    )
                        .into_py(py),
                );
            }
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

/// Return a new PyGraph by forming the tensor product from two input
/// PyGraph objects
///
/// :param PyGraph first: The first undirected graph object
/// :param PyGraph second: The second undirected graph object
///
/// :returns: A new PyGraph object that is the tensor product of ``first``
///     and ``second``.
///     A read-only dictionary of the product of nodes is also returned. The keys
///     are a tuple where the first element is a node of the first graph and the
///     second element is a node of the second graph, and the values are the map
///     of those elements to node indices in the product graph. For example::
///     
///         {
///             (0, 0): 0,
///             (0, 1): 1,
///         }
///
/// :rtype: Tuple[:class:`~rustworkx.PyGraph`, :class:`~rustworkx.ProductNodeMap`]
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph_1 = rustworkx.generators.path_graph(2)
///   graph_2 = rustworkx.generators.path_graph(3)
///   graph_product, _ = rustworkx.graph_tensor_product(graph_1, graph_2)
///   mpl_draw(graph_product)
#[pyfunction()]
#[pyo3(text_signature = "(first, second, /)")]
pub fn graph_tensor_product(
    py: Python,
    first: &graph::PyGraph,
    second: &graph::PyGraph,
) -> (graph::PyGraph, ProductNodeMap) {
    let (out_graph, out_node_map) = tensor_product(py, &first.graph, &second.graph, true);

    (
        graph::PyGraph {
            graph: out_graph,
            multigraph: true,
            node_removed: false,
            attrs: py.None(),
        },
        out_node_map,
    )
}

/// Return a new PyDiGraph by forming the tensor product from two input
/// PyGraph objects
///
/// :param PyDiGraph first: The first undirected graph object
/// :param PyDiGraph second: The second undirected graph object
///
/// :returns: A new PyDiGraph object that is the tensor product of ``first``
///     and ``second``.
///     A read-only dictionary of the product of nodes is also returned. The keys
///     are a tuple where the first element is a node of the first graph and the
///     second element is a node of the second graph, and the values are the map
///     of those elements to node indices in the product graph. For example::
///     
///         {
///             (0, 0): 0,
///             (0, 1): 1,
///         }
///
/// :rtype: Tuple[:class:`~rustworkx.PyDiGraph`, :class:`~rustworkx.ProductNodeMap`]
///
/// .. jupyter-execute::
///
///   import rustworkx.generators
///   from rustworkx.visualization import mpl_draw
///
///   graph_1 = rustworkx.generators.directed_path_graph(2)
///   graph_2 = rustworkx.generators.directed_path_graph(3)
///   graph_product, _ = rustworkx.digraph_tensor_product(graph_1, graph_2)
///   mpl_draw(graph_product)
#[pyfunction()]
#[pyo3(text_signature = "(first, second, /)")]
pub fn digraph_tensor_product(
    py: Python,
    first: &digraph::PyDiGraph,
    second: &digraph::PyDiGraph,
) -> (digraph::PyDiGraph, ProductNodeMap) {
    let (out_graph, out_node_map) = tensor_product(py, &first.graph, &second.graph, false);

    (
        digraph::PyDiGraph {
            graph: out_graph,
            cycle_state: algo::DfsSpace::default(),
            check_cycle: false,
            node_removed: false,
            multigraph: true,
            attrs: py.None(),
        },
        out_node_map,
    )
}
