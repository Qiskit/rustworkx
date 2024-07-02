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
use petgraph::visit::{Data, NodeIndexable, NodeRef};

use super::utils::get_num_nodes;
use super::InvalidInputError;
use crate::utils::pairwise;

/// Generate a barbell graph where two identical complete graphs are
/// connected by a path.
///
///   If neither `num_path_nodes` nor `path_weights` (described below) are
///   specified, then this is equivalent to two complete graphs joined together.
///
/// Arguments:
///
/// * `num_mesh_nodes` - The number of nodes to generate the mesh graph
///     with. Node weights will be None if this is specified. If both
///     `num_mesh_nodes` and `mesh_weights` are set, this will be ignored and
///     `mesh_weights` will be used.
/// * `num_path_nodes` - The number of nodes to generate the path
///     with. Node weights will be None if this is specified. If both
///     `num_path_nodes` and `path_weights` are set, this will be ignored and
///     `path_weights` will be used.
/// * `mesh_weights` - A list of node weights for the mesh graph. If both
///     `num_mesh_nodes` and `mesh_weights` are set, `num_mesh_nodes` will
///     be ignored and `mesh_weights` will be used.
/// * `path_weights` - A list of node weights for the path. If both
///     `num_path_nodes` and `path_weights` are set, `num_path_nodes` will
///     be ignored and `path_weights` will be used.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes. This is ignored if node weights are specified.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::barbell_graph;
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
///     (6, 7),
///     (7, 8),
///     (7, 9),
///     (7, 10),
///     (8, 9),
///     (8, 10),
///     (9, 10),
/// ];
/// let g: petgraph::graph::UnGraph<(), ()> = barbell_graph(
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
pub fn barbell_graph<G, T, F, H, M>(
    num_mesh_nodes: Option<usize>,
    num_path_nodes: Option<usize>,
    mesh_weights: Option<Vec<T>>,
    path_weights: Option<Vec<T>>,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut() -> T,
    H: FnMut() -> M,
    G::NodeId: Eq + Hash + NodeRef,
    T: Clone,
{
    if num_mesh_nodes.is_none() && mesh_weights.is_none() {
        return Err(InvalidInputError {});
    }
    let num_nodes = get_num_nodes(&num_mesh_nodes, &mesh_weights);
    let num_edges = (num_nodes * (num_nodes - 1)) / 2;
    let mesh_weights_2 = mesh_weights.clone();

    let mut graph = G::with_capacity(num_nodes, num_edges);

    // Create left mesh graph
    let mesh_nodes: Vec<G::NodeId> = match mesh_weights {
        Some(mesh_weights) => mesh_weights
            .into_iter()
            .map(|weight| graph.add_node(weight))
            .collect(),
        None => (0..num_mesh_nodes.unwrap())
            .map(|_| graph.add_node(default_node_weight()))
            .collect(),
    };

    let meshlen = mesh_nodes.len();
    for i in 0..meshlen - 1 {
        for j in i + 1..meshlen {
            graph.add_edge(mesh_nodes[i], mesh_nodes[j], default_edge_weight());
        }
    }

    // Add path nodes and edges
    let path_nodes: Vec<G::NodeId> = match path_weights {
        Some(path_weights) => path_weights
            .into_iter()
            .map(|weight| graph.add_node(weight))
            .collect(),
        None => {
            if let Some(num_path) = num_path_nodes {
                (0..num_path)
                    .map(|_| graph.add_node(default_node_weight()))
                    .collect()
            } else {
                vec![]
            }
        }
    };
    let pathlen = path_nodes.len();
    if !path_nodes.is_empty() {
        graph.add_edge(
            graph.from_index(meshlen - 1),
            graph.from_index(meshlen),
            default_edge_weight(),
        );
        for (node_a, node_b) in pairwise(path_nodes) {
            if let Some(node_a) = node_a{
                graph.add_edge(node_a, node_b, default_edge_weight());
            }
        }
    }

    // Add right mesh graph
    let mesh_nodes: Vec<G::NodeId> = match mesh_weights_2 {
        Some(mesh_weights_2) => mesh_weights_2
            .into_iter()
            .map(|weight| graph.add_node(weight))
            .collect(),
        None => (0..num_mesh_nodes.unwrap())
            .map(|_| graph.add_node(default_node_weight()))
            .collect(),
    };
    let meshlen = mesh_nodes.len();

    graph.add_edge(
        graph.from_index(meshlen + pathlen - 1),
        graph.from_index(meshlen + pathlen),
        default_edge_weight(),
    );
    for i in 0..meshlen - 1 {
        for j in i + 1..meshlen {
            graph.add_edge(mesh_nodes[i], mesh_nodes[j], default_edge_weight());
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::barbell_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_barbell_mesh_path() {
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
            (6, 7),
            (7, 8),
            (7, 9),
            (7, 10),
            (8, 9),
            (8, 10),
            (9, 10),
        ];
        let g: petgraph::graph::UnGraph<(), ()> =
            barbell_graph(Some(4), Some(3), None, None, || (), || ()).unwrap();
        assert_eq!(
            expected_edge_list,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_barbell_none_mesh() {
        match barbell_graph::<petgraph::graph::UnGraph<(), ()>, (), _, _, ()>(
            None,
            None,
            None,
            None,
            || (),
            || (),
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }
}
