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

/// Generate a hexagonal lattice graph
///
/// Arguments:
///
/// * `rows` - The number of rows to generate the graph with.
/// * `cols` - The number of columns to generate the graph with.
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
/// use rustworkx_core::generators::hexagonal_lattice_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let g: petgraph::graph::UnGraph<(), ()> = hexagonal_lattice_graph(
///     2,
///     2,
///     || {()},
///     || {()},
///     false
/// ).unwrap();
/// let expected_edges = vec![
///      (0, 1),
///      (1, 2),
///      (2, 3),
///      (3, 4),
///      (5, 6),
///      (6, 7),
///      (7, 8),
///      (8, 9),
///      (9, 10),
///      (11, 12),
///      (12, 13),
///      (13, 14),
///      (14, 15),
///      (0, 5),
///      (2, 7),
///      (4, 9),
///      (6, 11),
///      (8, 13),
///      (10, 15),
/// ];
/// assert_eq!(
///     expected_edges,
///     g.edge_references()
///         .map(|edge| (edge.source().index(), edge.target().index()))
///         .collect::<Vec<(usize, usize)>>(),
/// )
/// ```
pub fn hexagonal_lattice_graph<G, T, F, H, M>(
    rows: usize,
    cols: usize,
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
    if rows == 0 || cols == 0 {
        return Ok(G::with_capacity(0, 0));
    }
    let mut rowlen = rows;
    let mut collen = cols;

    // Needs two times the number of nodes vertically
    rowlen = 2 * rowlen + 2;
    collen += 1;
    let num_nodes = rowlen * collen - 2;

    let mut graph = G::with_capacity(num_nodes, num_nodes);

    let nodes: Vec<G::NodeId> = (0..num_nodes)
        .map(|_| graph.add_node(default_node_weight()))
        .collect();

    // Add column edges
    for j in 0..(rowlen - 2) {
        graph.add_edge(nodes[j], nodes[j + 1], default_edge_weight());
        if bidirectional {
            graph.add_edge(nodes[j + 1], nodes[j], default_edge_weight());
        }
    }
    for i in 1..(collen - 1) {
        for j in 0..(rowlen - 1) {
            graph.add_edge(
                nodes[i * rowlen + j - 1],
                nodes[i * rowlen + j],
                default_edge_weight(),
            );
            if bidirectional {
                graph.add_edge(
                    nodes[i * rowlen + j],
                    nodes[i * rowlen + j - 1],
                    default_edge_weight(),
                );
            }
        }
    }
    for j in 0..(rowlen - 2) {
        graph.add_edge(
            nodes[(collen - 1) * rowlen + j - 1],
            nodes[(collen - 1) * rowlen + j],
            default_edge_weight(),
        );
        if bidirectional {
            graph.add_edge(
                nodes[(collen - 1) * rowlen + j],
                nodes[(collen - 1) * rowlen + j - 1],
                default_edge_weight(),
            );
        }
    }

    // Add row edges
    for j in (0..(rowlen - 1)).step_by(2) {
        graph.add_edge(nodes[j], nodes[j + rowlen - 1], default_edge_weight());
        if bidirectional {
            graph.add_edge(nodes[j + rowlen - 1], nodes[j], default_edge_weight());
        }
    }
    for i in 1..(collen - 2) {
        for j in 0..rowlen {
            if i % 2 == j % 2 {
                graph.add_edge(
                    nodes[i * rowlen + j - 1],
                    nodes[(i + 1) * rowlen + j - 1],
                    default_edge_weight(),
                );
                if bidirectional {
                    graph.add_edge(
                        nodes[(i + 1) * rowlen + j - 1],
                        nodes[i * rowlen + j - 1],
                        default_edge_weight(),
                    );
                }
            }
        }
    }
    if collen > 2 {
        for j in ((collen % 2)..rowlen).step_by(2) {
            graph.add_edge(
                nodes[(collen - 2) * rowlen + j - 1],
                nodes[(collen - 1) * rowlen + j - 1 - (collen % 2)],
                default_edge_weight(),
            );
            if bidirectional {
                graph.add_edge(
                    nodes[(collen - 1) * rowlen + j - 1 - (collen % 2)],
                    nodes[(collen - 2) * rowlen + j - 1],
                    default_edge_weight(),
                );
            }
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::hexagonal_lattice_graph;
    use crate::petgraph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_hexagonal_lattice_graph() {
        let expected_edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (0, 5),
            (2, 7),
            (4, 9),
            (6, 11),
            (8, 13),
            (10, 15),
        ];
        let g: petgraph::graph::UnGraph<(), ()> =
            hexagonal_lattice_graph(2, 2, || (), || (), false).unwrap();
        assert_eq!(g.node_count(), 16);
        assert_eq!(g.edge_count(), expected_edges.len());
        assert_eq!(
            expected_edges,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_directed_hexagonal_lattice_graph() {
        let expected_edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (0, 5),
            (2, 7),
            (4, 9),
            (6, 11),
            (8, 13),
            (10, 15),
        ];
        let g: petgraph::graph::DiGraph<(), ()> =
            hexagonal_lattice_graph(2, 2, || (), || (), false).unwrap();
        assert_eq!(g.node_count(), 16);
        assert_eq!(g.edge_count(), expected_edges.len());
        assert_eq!(
            expected_edges,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_directed_hexagonal_lattice_graph_bidirectional() {
        let expected_edges = vec![
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (5, 6),
            (6, 5),
            (6, 7),
            (7, 6),
            (7, 8),
            (8, 7),
            (8, 9),
            (9, 8),
            (9, 10),
            (10, 9),
            (11, 12),
            (12, 11),
            (12, 13),
            (13, 12),
            (13, 14),
            (14, 13),
            (14, 15),
            (15, 14),
            (0, 5),
            (5, 0),
            (2, 7),
            (7, 2),
            (4, 9),
            (9, 4),
            (6, 11),
            (11, 6),
            (8, 13),
            (13, 8),
            (10, 15),
            (15, 10),
        ];
        let g: petgraph::graph::DiGraph<(), ()> =
            hexagonal_lattice_graph(2, 2, || (), || (), true).unwrap();
        assert_eq!(g.node_count(), 16);
        assert_eq!(g.edge_count(), expected_edges.len());
        assert_eq!(
            expected_edges,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_hexagonal_lattice_error() {
        let g: petgraph::graph::UnGraph<(), ()> =
            hexagonal_lattice_graph(0, 0, || (), || (), false).unwrap();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }
}
