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

use super::InvalidInputError;

/// Generate a generalized Petersen graph `G(n, k)` with `2n`
/// nodes and `3n` edges.
///
///   The Petersen graph itself is denoted `G(5, 2)`
///
/// * `n` - Number of nodes in the internal star and external regular polygon.
///     n > 2.
/// * `k` - Shift that changes the internal star graph. k > 0 and 2 * k < n.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes. This is ignored if `weights` is specified.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
/// * `bidirectional` - Whether edges are added bidirectionally. If set to
///     `true` then for any edge `(u, v)` an edge `(v, u)` will also be added.
///     If the graph is undirected this will result in a parallel edge.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::petersen_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let expected_edge_list = vec![
///     (0, 2),
///     (1, 3),
///     (2, 4),
///     (3, 0),
///     (4, 1),
///     (5, 6),
///     (6, 7),
///     (7, 8),
///     (8, 9),
///     (9, 5),
///     (5, 0),
///     (6, 1),
///     (7, 2),
///     (8, 3),
///     (9, 4),
/// ];
/// let g: petgraph::graph::UnGraph<(), ()> = petersen_graph(
///     5,
///     2,
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
pub fn petersen_graph<G, T, F, H, M>(
    n: usize,
    k: usize,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut() -> T,
    H: FnMut() -> M,
    G::NodeId: Eq + Hash,
{
    if n < 3 {
        return Err(InvalidInputError {});
    }
    if k == 0 || 2 * k >= n {
        return Err(InvalidInputError {});
    }

    let mut graph = G::with_capacity(2 * n, 3 * n);

    let star_nodes: Vec<G::NodeId> = (0..n)
        .map(|_| graph.add_node(default_node_weight()))
        .collect();

    let polygon_nodes: Vec<G::NodeId> = (0..n)
        .map(|_| graph.add_node(default_node_weight()))
        .collect();

    for i in 0..n {
        graph.add_edge(
            star_nodes[i],
            star_nodes[(i + k) % n],
            default_edge_weight(),
        );
    }

    for i in 0..n {
        graph.add_edge(
            polygon_nodes[i],
            polygon_nodes[(i + 1) % n],
            default_edge_weight(),
        );
    }

    for i in 0..n {
        graph.add_edge(polygon_nodes[i], star_nodes[i], default_edge_weight());
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::petersen_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_petersen_graph() {
        let expected_edge_list = vec![
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 0),
            (4, 1),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 5),
            (5, 0),
            (6, 1),
            (7, 2),
            (8, 3),
            (9, 4),
        ];
        let g: petgraph::graph::UnGraph<(), ()> = petersen_graph(5, 2, || (), || ()).unwrap();
        assert_eq!(
            expected_edge_list,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_petersen_error() {
        match petersen_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(2, 3, || (), || ()) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }
}
