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
use petgraph::visit::{Data, NodeIndexable};

use super::utils::pairwise;
use super::InvalidInputError;

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
/// let g: petgraph::graph::UnGraph<(), ()> = lollipop_graph(
///     Some(3),
///     Some(3),
///     None,
///     || {()},
///     || {()},
///     false
/// ).unwrap();
/// assert_eq!(
///     vec![(0, 3), (0, 1), (1, 4), (1, 2), (2, 5),
///          (3, 6), (3, 4), (4, 7), (4, 5), (5, 8), (6, 7), (7, 8)],
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
    bidirectional: bool,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut() -> T,
    H: FnMut() -> M,
    G::NodeId: Eq + Hash,
{
    let mut graph = complete_graph(num_mesh_nodes, mesh_weights);
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
    fn test_directed_lollipop_simple_row_col() {
        let g: petgraph::graph::DiGraph<(), ()> =
            lollipop_graph(Some(3), Some(3), None, || (), || (), false).unwrap();
        assert_eq!(
            vec![
                (0, 3),
                (0, 1),
                (1, 4),
                (1, 2),
                (2, 5),
                (3, 6),
                (3, 4),
                (4, 7),
                (4, 5),
                (5, 8),
                (6, 7),
                (7, 8)
            ],
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
        assert_eq!(g.edge_count(), 12);
    }

    #[test]
    fn test_lollipop_simple_row_col() {
        let g: petgraph::graph::UnGraph<(), ()> =
            lollipop_graph(Some(3), Some(3), None, || (), || (), false).unwrap();
        assert_eq!(
            vec![
                (0, 3),
                (0, 1),
                (1, 4),
                (1, 2),
                (2, 5),
                (3, 6),
                (3, 4),
                (4, 7),
                (4, 5),
                (5, 8),
                (6, 7),
                (7, 8)
            ],
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
        assert_eq!(g.edge_count(), 12);
    }

    #[test]
    fn test_directed_lollipop_weights() {
        let g: petgraph::graph::DiGraph<usize, ()> = lollipop_graph(
            Some(2),
            Some(3),
            Some(vec![0, 1, 2, 3, 4, 5]),
            || 4,
            || (),
            false,
        )
        .unwrap();
        assert_eq!(
            vec![(0, 3), (0, 1), (1, 4), (1, 2), (2, 5), (3, 4), (4, 5),],
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
        assert_eq!(g.edge_count(), 7);
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5],
            g.node_weights().copied().collect::<Vec<usize>>(),
        );
    }

    #[test]
    fn test_directed_lollipop_more_weights() {
        let g: petgraph::graph::DiGraph<usize, ()> = lollipop_graph(
            Some(2),
            Some(3),
            Some(vec![0, 1, 2, 3, 4, 5, 6, 7]),
            || 4,
            || (),
            false,
        )
        .unwrap();
        assert_eq!(
            vec![(0, 3), (0, 1), (1, 4), (1, 2), (2, 5), (3, 4), (4, 5),],
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
        assert_eq!(g.edge_count(), 7);
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5],
            g.node_weights().copied().collect::<Vec<usize>>(),
        );
    }

    #[test]
    fn test_directed_lollipop_less_weights() {
        let g: petgraph::graph::DiGraph<usize, ()> =
            lollipop_graph(Some(2), Some(3), Some(vec![0, 1, 2, 3]), || 6, || (), false).unwrap();
        assert_eq!(
            vec![(0, 3), (0, 1), (1, 4), (1, 2), (2, 5), (3, 4), (4, 5),],
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
        assert_eq!(g.edge_count(), 7);
        assert_eq!(
            vec![0, 1, 2, 3, 6, 6],
            g.node_weights().copied().collect::<Vec<usize>>(),
        );
    }

    #[test]
    fn test_directed_lollipop_bidirectional() {
        let g: petgraph::graph::DiGraph<(), ()> =
            lollipop_graph(Some(2), Some(3), None, || (), || (), true).unwrap();
        assert_eq!(
            vec![
                (0, 3),
                (3, 0),
                (0, 1),
                (1, 0),
                (1, 4),
                (4, 1),
                (1, 2),
                (2, 1),
                (2, 5),
                (5, 2),
                (3, 4),
                (4, 3),
                (4, 5),
                (5, 4),
            ],
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
        assert_eq!(g.edge_count(), 14);
    }

    #[test]
    fn test_lollipop_error() {
        match lollipop_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
            None,
            None,
            None,
            || (),
            || (),
            false,
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }
}
