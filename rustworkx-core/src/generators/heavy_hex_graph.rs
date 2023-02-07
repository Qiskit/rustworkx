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

/// Generate a heavy hex graph.
///
/// Fig. 2 of <https://arxiv.org/abs/1907.09528>
/// An ASCII diagram of the graph is given by:
/// ```text
///    ... D-S-D   D ...
///        |   |   |
///    ...-F   F-S-F ...
///        |   |   |
///    ... D   D   D ...
///        |   |   |
///    ... F-S-F   F-...
///        |   |   |
///        .........
///        |   |   |
///    ... D   D   D ...
///        |   |   |
///    ...-F   F-S-F ...
///        |   |   |
///    ... D   D   D ...
///        |   |   |
///    ... F-S-F   F-...
///        |   |   |
///        .........
///        |   |   |
///    ... D   D   D ...
///        |   |   |
///    ...-F   F-S-F ...
///        |   |   |
///    ... D   D   D ...
///        |   |   |
///    ... F-S-F   F-...
///        |   |   |
///    ... D   D-S-D ...
///```
///
/// Arguments:
///
/// * `d` - Distance of the code. If `d` is set to `1` a graph with a
///     single node will be returned. `d` must be an odd number.
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
/// use rustworkx_core::generators::heavy_hex_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let expected_edge_list = vec![
///     (0, 13),
///     (1, 13),
///     (1, 14),
///     (2, 14),
///     (3, 15),
///     (4, 15),
///     (4, 16),
///     (5, 16),
///     (6, 17),
///     (7, 17),
///     (7, 18),
///     (8, 18),
///     (0, 9),
///     (3, 9),
///     (5, 12),
///     (8, 12),
///     (10, 14),
///     (10, 16),
///     (11, 15),
///     (11, 17),
/// ];
/// let d = 3;
/// let g: petgraph::graph::UnGraph<(), ()> = heavy_hex_graph(d, || (), || (), false).unwrap();
/// assert_eq!(g.node_count(), (5 * d * d - 2 * d - 1) / 2);
/// assert_eq!(g.edge_count(), 2 * d * (d - 1) + (d + 1) * (d - 1));
/// assert_eq!(
///     expected_edge_list,
///     g.edge_references()
///         .map(|edge| (edge.source().index(), edge.target().index()))
///         .collect::<Vec<(usize, usize)>>(),
/// )
/// ```
pub fn heavy_hex_graph<G, T, F, H, M>(
    d: usize,
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
    if d % 2 == 0 {
        return Err(InvalidInputError {});
    }
    let num_nodes = (5 * d * d - 2 * d - 1) / 2;
    let num_edges = 2 * d * (d - 1) + (d + 1) * (d - 1);
    let mut graph = G::with_capacity(num_nodes, num_edges);

    if d == 1 {
        graph.add_node(default_node_weight());
        return Ok(graph);
    }
    let num_data = d * d;
    let num_syndrome = (d - 1) * (d + 1) / 2;
    let num_flag = d * (d - 1);

    let nodes_data: Vec<G::NodeId> = (0..num_data)
        .map(|_| graph.add_node(default_node_weight()))
        .collect();
    let nodes_syndrome: Vec<G::NodeId> = (0..num_syndrome)
        .map(|_| graph.add_node(default_node_weight()))
        .collect();
    let nodes_flag: Vec<G::NodeId> = (0..num_flag)
        .map(|_| graph.add_node(default_node_weight()))
        .collect();

    // connect data and flags
    for (i, flag_chunk) in nodes_flag.chunks(d - 1).enumerate() {
        for (j, flag) in flag_chunk.iter().enumerate() {
            graph.add_edge(nodes_data[i * d + j], *flag, default_edge_weight());
            graph.add_edge(nodes_data[i * d + j + 1], *flag, default_edge_weight());
            if bidirectional {
                graph.add_edge(*flag, nodes_data[i * d + j], default_edge_weight());
                graph.add_edge(*flag, nodes_data[i * d + j + 1], default_edge_weight());
            }
        }
    }

    // connect data and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks((d + 1) / 2).enumerate() {
        if i % 2 == 0 {
            graph.add_edge(nodes_data[i * d], syndrome_chunk[0], default_edge_weight());
            graph.add_edge(
                nodes_data[(i + 1) * d],
                syndrome_chunk[0],
                default_edge_weight(),
            );
            if bidirectional {
                graph.add_edge(syndrome_chunk[0], nodes_data[i * d], default_edge_weight());
                graph.add_edge(
                    syndrome_chunk[0],
                    nodes_data[(i + 1) * d],
                    default_edge_weight(),
                );
            }
        } else if i % 2 == 1 {
            graph.add_edge(
                nodes_data[i * d + (d - 1)],
                syndrome_chunk[syndrome_chunk.len() - 1],
                default_edge_weight(),
            );
            graph.add_edge(
                nodes_data[i * d + (2 * d - 1)],
                syndrome_chunk[syndrome_chunk.len() - 1],
                default_edge_weight(),
            );
            if bidirectional {
                graph.add_edge(
                    syndrome_chunk[syndrome_chunk.len() - 1],
                    nodes_data[i * d + (d - 1)],
                    default_edge_weight(),
                );
                graph.add_edge(
                    syndrome_chunk[syndrome_chunk.len() - 1],
                    nodes_data[i * d + (2 * d - 1)],
                    default_edge_weight(),
                );
            }
        }
    }

    // connect flag and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks((d + 1) / 2).enumerate() {
        if i % 2 == 0 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != 0 {
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[i * (d - 1) + 2 * (j - 1) + 1],
                        default_edge_weight(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + 2 * (j - 1) + 1],
                        default_edge_weight(),
                    );
                    if bidirectional {
                        graph.add_edge(
                            nodes_flag[i * (d - 1) + 2 * (j - 1) + 1],
                            *syndrome,
                            default_edge_weight(),
                        );
                        graph.add_edge(
                            nodes_flag[(i + 1) * (d - 1) + 2 * (j - 1) + 1],
                            *syndrome,
                            default_edge_weight(),
                        );
                    }
                }
            }
        } else if i % 2 == 1 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != syndrome_chunk.len() - 1 {
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[i * (d - 1) + 2 * j],
                        default_edge_weight(),
                    );
                    graph.add_edge(
                        *syndrome,
                        nodes_flag[(i + 1) * (d - 1) + 2 * j],
                        default_edge_weight(),
                    );
                    if bidirectional {
                        graph.add_edge(
                            nodes_flag[i * (d - 1) + 2 * j],
                            *syndrome,
                            default_edge_weight(),
                        );
                        graph.add_edge(
                            nodes_flag[(i + 1) * (d - 1) + 2 * j],
                            *syndrome,
                            default_edge_weight(),
                        );
                    }
                }
            }
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::heavy_hex_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_heavy_hex_graph_3() {
        let expected_edge_list = vec![
            (0, 13),
            (1, 13),
            (1, 14),
            (2, 14),
            (3, 15),
            (4, 15),
            (4, 16),
            (5, 16),
            (6, 17),
            (7, 17),
            (7, 18),
            (8, 18),
            (0, 9),
            (3, 9),
            (5, 12),
            (8, 12),
            (10, 14),
            (10, 16),
            (11, 15),
            (11, 17),
        ];
        let d = 3;
        let g: petgraph::graph::UnGraph<(), ()> = heavy_hex_graph(d, || (), || (), false).unwrap();
        assert_eq!(g.node_count(), (5 * d * d - 2 * d - 1) / 2);
        assert_eq!(g.edge_count(), 2 * d * (d - 1) + (d + 1) * (d - 1));
        assert_eq!(
            expected_edge_list,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_heavy_hex_error() {
        let d = 2;
        match heavy_hex_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
            d,
            || (),
            || (),
            false,
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }
}
