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

use std::hash::Hash;

use petgraph::data::{Build, Create};
use petgraph::visit::{Data, GraphProp, NodeIndexable};

use super::utils::pairwise;
use super::InvalidInputError;
use crate::generators::complete_graph;

/// Generate a lollipop graph
///
/// Arguments:
///
/// Generate an undirected lollipop graph where a mesh graph is connected to a
/// path.
///
/// If neither `num_path_nodes` nor `path_weights` (both described
/// below) are specified then this is equivalent to
/// :func:`~rustworkx.generators.mesh_graph`
///
/// * `num_mesh_nodes` - The number of nodes to generate the mesh graph
///     with. Node weights will be None if this is specified. If both
///     `num_mesh_nodes` and ``mesh_weights`` are set this will be ignored and
///     `mesh_weights` will be used.
/// * `num_path_nodes` - The number of nodes to generate the path
///     with. Node weights will be None if this is specified. If both
///     `num_path_nodes` and `path_weights` are set this will be ignored and
///     `path_weights` will be used.
/// * `mesh_weights` - A list of node weights for the mesh graph. If both
///     `num_mesh_nodes` and `mesh_weights` are set `num_mesh_nodes` will
///     be ignored and `mesh_weights` will be used.
/// * `path_weights` - A list of node weights for the path. If both
///     `num_path_nodes` and `path_weights` are set `num_path_nodes` will
///     be ignored and `path_weights` will be used.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes. This is ignored if `weights` is specified,
///     as the weights from that argument will be used instead.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::lollipop_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let expected_edge_list = vec![
///     (0, 1),
///     (0, 2),
///     (0, 3),
///     (1, 2),
///     (1, 3),
///     (2, 3),
///     (3, 4),
///     (4, 5),
///     (5, 6),
/// ];
/// let g: petgraph::graph::UnGraph<(), ()> = lollipop_graph(
///     Some(4),
///     Some(3),
///     None,
///     None,
///     || {()},
///     || {()},
/// ).unwrap();
/// assert_eq!(
///     expected_edge_list,
///     g.edge_references()
///         .map(|edge| (edge.source().index(), edge.target().index()))
///         .collect::<Vec<(usize, usize)>>(),
/// )
/// ```
pub fn lollipop_graph<G, T, F, H, M>(
    num_mesh_nodes: Option<usize>,
    num_path_nodes: Option<usize>,
    mesh_weights: Option<Vec<T>>,
    path_weights: Option<Vec<T>>,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable + GraphProp,
    F: FnMut() -> T,
    H: FnMut() -> M,
    G::NodeId: Eq + Hash,
{
    if num_mesh_nodes.is_none() {
        return Err(PyIndexError::new_err("num_mesh_nodes not specified"));
    }

    let mut left_mesh = StableUnGraph::<PyObject, PyObject>::default();
    let mesh_nodes: Vec<NodeIndex> = (0..num_mesh_nodes.unwrap())
        .map(|_| left_mesh.add_node(py.None()))
        .collect();
    let mut nodelen = mesh_nodes.len();
    for i in 0..nodelen - 1 {
        for j in i + 1..nodelen {
            left_mesh.add_edge(mesh_nodes[i], mesh_nodes[j], py.None());
        }
    }

    let right_mesh = left_mesh.clone();

    if let Some(num_nodes) = num_path_nodes {
        let path_nodes: Vec<NodeIndex> = (0..num_nodes)
            .map(|_| left_mesh.add_node(py.None()))
            .collect();
        left_mesh.add_edge(
            NodeIndex::new(nodelen - 1),
            NodeIndex::new(nodelen),
            py.None(),
        );

        nodelen += path_nodes.len();

        for (node_a, node_b) in pairwise(path_nodes) {
            match node_a {
                Some(node_a) => left_mesh.add_edge(node_a, node_b, py.None()),
                None => continue,
            };
        }
    }

    for node in right_mesh.node_indices() {
        let new_node = &right_mesh[node];
        left_mesh.add_node(new_node.clone_ref(py));
    }
    left_mesh.add_edge(
        NodeIndex::new(nodelen - 1),
        NodeIndex::new(nodelen),
        py.None(),
    );
    for edge in right_mesh.edge_references() {
        let new_source = NodeIndex::new(nodelen + edge.source().index());
        let new_target = NodeIndex::new(nodelen + edge.target().index());
        let weight = edge.weight();
        left_mesh.add_edge(new_source, new_target, weight.clone_ref(py));
    }





    if num_path_nodes.is_none() && path_weights.is_none() {
        return Ok(graph);
    }
    let meshlen = graph.node_count();

    let path_nodes: Vec<G::NodeId> = match path_weights {
        Some(path_weights) => path_weights
            .into_iter()
            .map(|weight| graph.add_node(weight))
            .collect(),
        None => (0..num_path_nodes.unwrap())
            .map(|_| graph.add_node(default_node_weight()))
            .collect(),
    };

    let pathlen = path_nodes.len();
    if pathlen > 0 {
        graph.add_edge(
            graph.from_index(meshlen - 1),
            graph.from_index(meshlen),
            default_edge_weight(),
        );
        for (node_a, node_b) in pairwise(path_nodes) {
            match node_a {
                Some(node_a) => graph.add_edge(node_a, node_b, default_edge_weight()),
                None => continue,
            };
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::lollipop_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph::visit::EdgeRef;
    #[test]
    fn test_lollipop_mesh_path() {
        let expected_edge_list = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
        ];
        let g: petgraph::graph::UnGraph<(), ()> =
            lollipop_graph(Some(4), Some(3), None, None, || (), || ()).unwrap();
        assert_eq!(
            expected_edge_list,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_lollipop_0_mesh_none_path() {
        let g: petgraph::graph::UnGraph<(), ()> =
            lollipop_graph(Some(0), None, None, None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }
}
