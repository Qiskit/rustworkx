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

pub struct HexagonalLatticeBuilder {
    rowlen: usize,    // Number of nodes in each vertical chain
    collen: usize,    // Number of vertical chains
    num_nodes: usize, // Total number of nodes
    num_edges: usize, // Total number of edges
    bidirectional: bool,
    periodic: bool,
}

impl HexagonalLatticeBuilder {
    pub fn new(
        rows: usize,
        cols: usize,
        bidirectional: bool,
        periodic: bool,
    ) -> Result<HexagonalLatticeBuilder, InvalidInputError> {
        if periodic && (cols % 2 == 1 || rows < 2 || cols < 2) {
            return Err(InvalidInputError {});
        }

        let num_edges_factor = if bidirectional { 2 } else { 1 };

        let (rowlen, collen, num_nodes, num_edges) = if periodic {
            let r_len = 2 * rows;
            (
                r_len,
                cols,
                r_len * cols,
                num_edges_factor * 3 * rows * cols,
            )
        } else {
            let r_len = 2 * rows + 2;
            (
                r_len,
                cols + 1,
                r_len * (cols + 1) - 2,
                num_edges_factor * (3 * rows * cols + 2 * (rows + cols) - 1),
            )
        };

        Ok(HexagonalLatticeBuilder {
            rowlen,
            collen,
            num_nodes,
            num_edges,
            bidirectional,
            periodic,
        })
    }

    pub fn build_with_default_node_weight<G, T, F, H, M>(
        self,
        mut default_node_weight: F,
        default_edge_weight: H,
    ) -> G
    where
        G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
        F: FnMut() -> T,
        H: FnMut() -> M,
        G::NodeId: Eq + Hash,
    {
        let mut graph = G::with_capacity(self.num_nodes, self.num_edges);
        let nodes: Vec<G::NodeId> = (0..self.num_nodes)
            .map(|_| graph.add_node(default_node_weight()))
            .collect();
        self.add_edges(&mut graph, nodes, default_edge_weight);

        graph
    }

    pub fn build_with_position_dependent_node_weight<G, T, F, H, M>(
        self,
        mut node_weight: F,
        default_edge_weight: H,
    ) -> G
    where
        G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
        F: FnMut(usize, usize) -> T,
        H: FnMut() -> M,
        G::NodeId: Eq + Hash,
    {
        let mut graph = G::with_capacity(self.num_nodes, self.num_edges);

        let lattice_position = |n| -> (usize, usize) {
            if self.periodic {
                (n / self.rowlen, n % self.rowlen)
            } else {
                // In the non-periodic case the first and last vertical
                // chains have rowlen - 1 = 2 * rows + 1 nodes. All others
                // have rowlen = 2 * rows + 2 nodes.
                if n < self.rowlen - 1 {
                    (0, n)
                } else {
                    let k = n - (self.rowlen - 1);
                    let u = k / self.rowlen + 1;
                    let v = k % self.rowlen;
                    if u == self.collen - 1 && u % 2 == 0 {
                        (u, v + 1)
                    } else {
                        (u, v)
                    }
                }
            }
        };

        let nodes: Vec<G::NodeId> = (0..self.num_nodes)
            .map(lattice_position)
            .map(|(u, v)| graph.add_node(node_weight(u, v)))
            .collect();
        self.add_edges(&mut graph, nodes, default_edge_weight);

        graph
    }

    fn add_edges<G, H, M>(&self, graph: &mut G, nodes: Vec<G::NodeId>, mut default_edge_weight: H)
    where
        G: Build + NodeIndexable + Data<EdgeWeight = M>,
        H: FnMut() -> M,
    {
        let mut add_edge = |u, v| {
            graph.add_edge(nodes[u], nodes[v], default_edge_weight());
            if self.bidirectional {
                graph.add_edge(nodes[v], nodes[u], default_edge_weight());
            }
        };

        if self.periodic {
            // Add column edges
            for i in 0..self.collen {
                let col_start = i * self.rowlen;
                for j in col_start..(col_start + self.rowlen - 1) {
                    add_edge(j, j + 1);
                }
                add_edge(col_start + self.rowlen - 1, col_start);
            }
            // Add row edges
            for i in 0..self.collen {
                let col_start = i * self.rowlen + i % 2;
                for j in (col_start..(col_start + self.rowlen)).step_by(2) {
                    add_edge(j, (j + self.rowlen) % self.num_nodes);
                }
            }
        } else {
            // Add column edges
            for j in 0..(self.rowlen - 2) {
                add_edge(j, j + 1);
            }
            for i in 1..(self.collen - 1) {
                for j in 0..(self.rowlen - 1) {
                    add_edge(i * self.rowlen + j - 1, i * self.rowlen + j);
                }
            }
            for j in 0..(self.rowlen - 2) {
                add_edge(
                    (self.collen - 1) * self.rowlen + j - 1,
                    (self.collen - 1) * self.rowlen + j,
                );
            }

            // Add row edges
            for j in (0..(self.rowlen - 1)).step_by(2) {
                add_edge(j, j + self.rowlen - 1);
            }
            for i in 1..(self.collen - 2) {
                for j in 0..self.rowlen {
                    if i % 2 == j % 2 {
                        add_edge(i * self.rowlen + j - 1, (i + 1) * self.rowlen + j - 1);
                    }
                }
            }
            if self.collen > 2 {
                for j in ((self.collen % 2)..self.rowlen).step_by(2) {
                    add_edge(
                        (self.collen - 2) * self.rowlen + j - 1,
                        (self.collen - 1) * self.rowlen + j - 1 - (self.collen % 2),
                    );
                }
            }
        }
    }
}

/// Generate a hexagonal lattice graph
///
/// Arguments:
///
/// * `rows` - The number of rows to generate the graph with.
/// * `cols` - The number of columns to generate the graph with.
/// * `default_node_weight` - A callable that will return the weight to use
///   for newly created nodes.
/// * `default_edge_weight` - A callable that will return the weight object
///   to use for newly created edges.
/// * `bidirectional` - Whether edges are added bidirectionally. If set to
///   `true` then for any edge `(u, v)` an edge `(v, u)` will also be added.
///   If the graph is undirected this will result in a parallel edge.
/// * `periodic` - If set to `true`, the boundaries of the lattice will be
///   joined to form a periodic grid. Requires `cols` to be even,
///   `rows > 1`, and `cols > 1`.
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
///     false,
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
    default_node_weight: F,
    default_edge_weight: H,
    bidirectional: bool,
    periodic: bool,
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

    let builder = HexagonalLatticeBuilder::new(rows, cols, bidirectional, periodic)?;

    let graph = builder
        .build_with_default_node_weight::<G, T, F, H, M>(default_node_weight, default_edge_weight);

    Ok(graph)
}

/// Generate a hexagonal lattice graph where each node is assigned a weight
/// depending on its position in the lattice.
///
/// Arguments:
///
/// * `rows` - The number of rows to generate the graph with.
/// * `cols` - The number of columns to generate the graph with.
/// * `node_weight` - A callable that will return the weight to use
///   for newly created nodes. Must take two arguments `i` and `j` of
///   type `usize`, where `(i, j)` gives the position of the node
///   in the lattice.
/// * `default_edge_weight` - A callable that will return the weight object
///   to use for newly created edges.
/// * `bidirectional` - Whether edges are added bidirectionally. If set to
///   `true` then for any edge `(u, v)` an edge `(v, u)` will also be added.
///   If the graph is undirected this will result in a parallel edge.
/// * `periodic` - If set to `true`, the boundaries of the lattice will be
///   joined to form a periodic grid. Requires `cols` to be even,
///   `rows > 1`, and `cols > 1`.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::hexagonal_lattice_graph_weighted;
/// use rustworkx_core::petgraph::visit::{IntoNodeReferences, NodeRef};
///
/// let g: petgraph::graph::UnGraph<(usize, usize), ()> = hexagonal_lattice_graph_weighted(
///     2,
///     2,
///     |u, v| {(u, v)},
///     || {()},
///     false,
///     false
/// ).unwrap();
/// let expected_node_weights = vec![
///     (0, 0),
///     (0, 1),
///     (0, 2),
///     (0, 3),
///     (0, 4),
///     (1, 0),
///     (1, 1),
///     (1, 2),
///     (1, 3),
///     (1, 4),
///     (1, 5),
///     (2, 1),
///     (2, 2),
///     (2, 3),
///     (2, 4),
///     (2, 5),
/// ];
/// assert_eq!(
///     expected_node_weights,
///     g.node_references()
///         .map(|node| *node.weight())
///         .collect::<Vec<(usize, usize)>>(),
/// )
/// ```
pub fn hexagonal_lattice_graph_weighted<G, T, F, H, M>(
    rows: usize,
    cols: usize,
    node_weight: F,
    default_edge_weight: H,
    bidirectional: bool,
    periodic: bool,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut(usize, usize) -> T,
    H: FnMut() -> M,
    G::NodeId: Eq + Hash,
{
    if rows == 0 || cols == 0 {
        return Ok(G::with_capacity(0, 0));
    }

    let builder = HexagonalLatticeBuilder::new(rows, cols, bidirectional, periodic)?;

    let graph = builder.build_with_position_dependent_node_weight::<G, T, F, H, M>(
        node_weight,
        default_edge_weight,
    );

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::{hexagonal_lattice_graph, hexagonal_lattice_graph_weighted};
    use crate::petgraph;
    use crate::petgraph::visit::{EdgeRef, IntoNodeReferences};
    use std::collections::HashSet;

    fn check_expected_edges_directed<T>(
        graph: &petgraph::graph::DiGraph<T, ()>,
        expected_edges: &[(usize, usize)],
    ) {
        assert_eq!(graph.edge_count(), expected_edges.len());

        let edge_set: HashSet<(usize, usize)> = graph
            .edge_references()
            .map(|edge| (edge.source().index(), edge.target().index()))
            .collect();
        let expected_set: HashSet<(usize, usize)> = expected_edges.iter().copied().collect();
        assert_eq!(edge_set, expected_set);
    }

    fn check_expected_edges_undirected(
        graph: &petgraph::graph::UnGraph<(), ()>,
        expected_edges: &[(usize, usize)],
    ) {
        assert_eq!(graph.edge_count(), expected_edges.len());

        let sorted_pair = |(a, b)| {
            if a > b { (b, a) } else { (a, b) }
        };

        let edge_set: HashSet<(usize, usize)> = graph
            .edge_references()
            .map(|edge| (edge.source().index(), edge.target().index()))
            .map(&sorted_pair)
            .collect();
        let expected_set: HashSet<(usize, usize)> =
            expected_edges.iter().copied().map(&sorted_pair).collect();
        assert_eq!(edge_set, expected_set);
    }

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
            hexagonal_lattice_graph(2, 2, || (), || (), false, false).unwrap();
        assert_eq!(g.node_count(), 16);
        check_expected_edges_undirected(&g, &expected_edges);
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
            hexagonal_lattice_graph(2, 2, || (), || (), false, false).unwrap();
        assert_eq!(g.node_count(), 16);
        check_expected_edges_directed(&g, &expected_edges);

        let g_weighted: petgraph::graph::DiGraph<(usize, usize), ()> =
            hexagonal_lattice_graph_weighted(2, 2, |u, v| (u, v), || (), false, false).unwrap();
        assert_eq!(g_weighted.node_count(), 16);
        check_expected_edges_directed(&g_weighted, &expected_edges);
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
            hexagonal_lattice_graph(2, 2, || (), || (), true, false).unwrap();
        assert_eq!(g.node_count(), 16);
        check_expected_edges_directed(&g, &expected_edges);
    }

    #[test]
    fn test_hexagonal_lattice_error() {
        let g: petgraph::graph::UnGraph<(), ()> =
            hexagonal_lattice_graph(0, 0, || (), || (), false, false).unwrap();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_hexagonal_lattice_periodic_error() {
        match hexagonal_lattice_graph::<petgraph::graph::UnGraph<(), ()>, (), _, _, ()>(
            5,
            3,
            || (),
            || (),
            false,
            true,
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, crate::generators::InvalidInputError),
        }
    }

    #[test]
    fn test_hexagonal_lattice_graph_periodic() {
        let expected_edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (2, 6),
            (5, 1),
            (7, 3),
        ];
        let g: petgraph::graph::UnGraph<(), ()> =
            hexagonal_lattice_graph(2, 2, || (), || (), false, true).unwrap();
        assert_eq!(g.node_count(), 8);
        check_expected_edges_undirected(&g, &expected_edges);
    }

    #[test]
    fn test_directed_hexagonal_lattice_graph_periodic() {
        let expected_edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (2, 6),
            (5, 1),
            (7, 3),
        ];
        let g: petgraph::graph::DiGraph<(), ()> =
            hexagonal_lattice_graph(2, 2, || (), || (), false, true).unwrap();
        assert_eq!(g.node_count(), 8);
        check_expected_edges_directed(&g, &expected_edges);

        let g_weighted: petgraph::graph::DiGraph<(usize, usize), ()> =
            hexagonal_lattice_graph_weighted(2, 2, |u, v| (u, v), || (), false, true).unwrap();
        assert_eq!(g_weighted.node_count(), 8);
        check_expected_edges_directed(&g_weighted, &expected_edges);
    }

    #[test]
    fn test_hexagonal_lattice_graph_node_weights() {
        let g: petgraph::graph::UnGraph<(usize, usize), ()> =
            hexagonal_lattice_graph_weighted(2, 2, |u, v| (u, v), || (), false, false).unwrap();
        let expected_node_weights = vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
        ];
        assert_eq!(
            expected_node_weights,
            g.node_references()
                .map(|node| *node.1)
                .collect::<Vec<(usize, usize)>>(),
        )
    }

    #[test]
    fn test_directed_hexagonal_lattice_graph_bidirectional_periodic() {
        let expected_edges = vec![
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 0),
            (0, 3),
            (4, 5),
            (5, 4),
            (5, 6),
            (6, 5),
            (6, 7),
            (7, 6),
            (7, 4),
            (4, 7),
            (0, 4),
            (4, 0),
            (2, 6),
            (6, 2),
            (5, 1),
            (1, 5),
            (7, 3),
            (3, 7),
        ];
        let g: petgraph::graph::DiGraph<(), ()> =
            hexagonal_lattice_graph(2, 2, || (), || (), true, true).unwrap();
        assert_eq!(g.node_count(), 8);
        check_expected_edges_directed(&g, &expected_edges);
    }
}
