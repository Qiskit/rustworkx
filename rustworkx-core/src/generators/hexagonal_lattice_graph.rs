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

use super::utils::get_num_nodes;
use super::InvalidInputError;

/// Generate a hexagonal lattice graph
///
/// Arguments:
///
/// * rows: The number of rows to generate the graph with.
/// * cols: The number of columns to generate the graph with.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes. This is ignored if `weights` is specified,
///     as the weights from that argument will be used instead.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
/// * `bidirectional` - Whether edges are added bidirectionally, if set to
///     `true` then for any edge `(u, v)` an edge `(v, u)` will also be added.
///     If the graph is undirected this will result in a pallel edge.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::hexagonal_lattice_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let g: petgraph::graph::UnGraph<(), ()> = hexagonal_lattice_graph(
///     Some(4),
///     None,
///     || {()},
///     || {()},
///     false
/// ).unwrap();
/// assert_eq!(
///     vec![(0, 1), (1, 2), (2, 3), (3, 0)],
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
    if rows == 0 && cols == 0 {
        return Err(InvalidInputError {});
    }

    let mut rowlen = rows;
    let mut collen = cols;

    // Needs two times the number of nodes vertically
    rowlen = 2 * rowlen + 2;
    collen += 1;
    let num_nodes = rowlen * collen - 2;

    let mut graph = G::with_capacity(num_nodes, num_nodes);
    let nodes: Vec<G::NodeId> = (0..num_nodes).map(|_| graph.add_node(default_node_weight())).collect();

    // Add column edges
    // first column
    for j in 0..(rowlen - 2) {
        graph.add_edge(nodes[j], nodes[j + 1], default_edge_weight());
    }

    for i in 1..(collen - 1) {
        for j in 0..(rowlen - 1) {
            graph.add_edge(nodes[i * rowlen + j - 1], nodes[i * rowlen + j], default_edge_weight());
        }
    }

    // last column
    for j in 0..(rowlen - 2) {
        graph.add_edge(
            nodes[(collen - 1) * rowlen + j - 1],
            nodes[(collen - 1) * rowlen + j],
            default_edge_weight(),
        );
    }

    // Add row edges
    for j in (0..(rowlen - 1)).step_by(2) {
        graph.add_edge(nodes[j], nodes[j + rowlen - 1], default_edge_weight());
    }

    for i in 1..(collen - 2) {
        for j in 0..rowlen {
            if i % 2 == j % 2 {
                graph.add_edge(
                    nodes[i * rowlen + j - 1],
                    nodes[(i + 1) * rowlen + j - 1],
                    default_edge_weight(),
                );
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
        }
    }
    // let node_len = get_num_nodes(&num_nodes, &weights);
    // let num_edges = if bidirectional {
    //     2 * node_len
    // } else {
    //     node_len
    // };
    // let mut graph = G::with_capacity(node_len, num_edges);
    // if node_len == 0 {
    //     return Ok(graph);
    // }

    // match weights {
    //     Some(weights) => {
    //         for weight in weights {
    //             graph.add_node(weight);
    //         }
    //     }
    //     None => {
    //         for _ in 0..node_len {
    //             graph.add_node(default_node_weight());
    //         }
    //     }
    // };
    // for a in 0..node_len - 1 {
    //     let node_a = graph.from_index(a);
    //     let node_b = graph.from_index(a + 1);
    //     graph.add_edge(node_a, node_b, default_edge_weight());
    //     if bidirectional {
    //         graph.add_edge(node_b, node_a, default_edge_weight());
    //     }
    // }
    // let last_node_index = graph.from_index(node_len - 1);
    // let first_node_index = graph.from_index(0);
    // graph.add_edge(last_node_index, first_node_index, default_edge_weight());
    // if bidirectional {
    //     graph.add_edge(first_node_index, last_node_index, default_edge_weight());
    // }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::hexagonal_lattice_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_with_weights() {
        let g: petgraph::graph::UnGraph<usize, ()> =
            hexagonal_lattice_graph(None, Some(vec![0, 1, 2, 3]), || 4, || (), false).unwrap();
        assert_eq!(
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
        assert_eq!(
            vec![0, 1, 2, 3],
            g.node_weights().copied().collect::<Vec<usize>>(),
        );
    }

    #[test]
    fn test_bidirectional() {
        let g: petgraph::graph::DiGraph<(), ()> =
            hexagonal_lattice_graph(Some(4), None, || (), || (), true).unwrap();
        assert_eq!(
            vec![
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 2),
                (3, 0),
                (0, 3)
            ],
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_error() {
        match hexagonal_lattice_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
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
