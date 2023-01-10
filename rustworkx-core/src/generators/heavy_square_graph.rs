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

/// Generate an directed heavy square graph. Fig. 6 of
/// https://arxiv.org/abs/1907.09528.
/// An ASCII diagram of the graph is given by:
///
/// .. code-block:: console
///
///     ...       S   ...
///        \     / \
///        ... D   D   D ...
///            |   |   |
///        ... F-S-F-S-F-...
///            |   |   |
///        ... D   D   D ...
///            |   |   |
///        ... F-S-F-S-F-...
///            |   |   |
///            .........
///            |   |   |
///        ... D   D   D ...
///             \ /     \
///        ...   S       ...
///
/// NOTE: This function generates the four-frequency variant of the heavy square code.
/// This function implements Fig 10.b left of the `paper <https://arxiv.org/abs/1907.09528>`_.
/// This function doesn't support the variant Fig 10.b right.
///
/// :param int d: distance of the code. If ``d`` is set to ``1`` a
///     :class:`~rustworkx.PyDiGraph` with a single node will be returned.
/// :param bool multigraph: When set to False the output
///     :class:`~rustworkx.PyDiGraph` object will not be not be a multigraph and
///     won't  allow parallel edges to be added. Instead
///     calls which would create a parallel edge will update the existing edge.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::heavy_square_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let g: petgraph::graph::UnGraph<(), ()> = heavy_square_graph(
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
pub fn heavy_square_graph<G, T, F, H, M>(
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
    num_nodes = 3 * d * d - 2 * d;
    num_edges = 2 * d * (d - 1) + 2 * d * (d - 1);
    let mut graph = G::with_capacity(node_len, num_edges);

    if d == 1 {
        graph.add_node(default_node_weight());
        return Ok(graph);
    }
    let num_data = d * d;
    let num_syndrome = d * (d - 1);
    let num_flag = d * (d - 1);

    let nodes_data: Vec<G::NodeId> = (0..num_data).map(|_| graph.add_node(default_node_weight())).collect();
    let nodes_syndrome: Vec<G::NodeId> = (0..num_syndrome)
        .map(|_| graph.add_node(default_node_weight()))
        .collect();
    let nodes_flag: Vec<G::NodeId> = (0..num_flag).map(|_| graph.add_node(default_node_weight())).collect();

    // connect data and flags
    for (i, flag_chunk) in nodes_flag.chunks(d - 1).enumerate() {
        for (j, flag) in flag_chunk.iter().enumerate() {
            graph.add_edge(nodes_data[i * d + j], *flag, default_edge_weight());
            graph.add_edge(*flag, nodes_data[i * d + j + 1], default_edge_weight());
            if bidirectional {
                graph.add_edge(*flag, nodes_data[i * d + j], default_edge_weight());
                graph.add_edge(nodes_data[i * d + j + 1], *flag, default_edge_weight());
            }
        }
    }

    // connect data and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks(d).enumerate() {
        if i % 2 == 0 {
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
        } else if i % 2 == 1 {
            graph.add_edge(nodes_data[i * d], syndrome_chunk[0], default_edge_weight());
            graph.add_edge(nodes_data[(i + 1) * d], syndrome_chunk[0], default_edge_weight());
            if bidirectional {
                graph.add_edge(syndrome_chunk[0], nodes_data[i * d], default_edge_weight());
                graph.add_edge(syndrome_chunk[0], nodes_data[(i + 1) * d], default_edge_weight());
            }
        }
    }

    // connect flag and syndromes
    for (i, syndrome_chunk) in nodes_syndrome.chunks(d).enumerate() {
        if i % 2 == 0 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != syndrome_chunk.len() - 1 {
                    graph.add_edge(*syndrome, nodes_flag[i * (d - 1) + j], default_edge_weight());
                    graph.add_edge(*syndrome, nodes_flag[(i + 1) * (d - 1) + j], default_edge_weight());
                    if bidirectional {
                        graph.add_edge(nodes_flag[i * (d - 1) + j], *syndrome, default_edge_weight());
                        graph.add_edge(nodes_flag[(i + 1) * (d - 1) + j], *syndrome, default_edge_weight());
                    }
                }
            }
        } else if i % 2 == 1 {
            for (j, syndrome) in syndrome_chunk.iter().enumerate() {
                if j != 0 {
                    graph.add_edge(*syndrome, nodes_flag[i * (d - 1) + j - 1], default_edge_weight());
                    graph.add_edge(*syndrome, nodes_flag[(i + 1) * (d - 1) + j - 1], default_edge_weight());
                    if bidirectional {
                        graph.add_edge(nodes_flag[i * (d - 1) + j - 1], *syndrome, default_edge_weight());
                        graph.add_edge(nodes_flag[(i + 1) * (d - 1) + j - 1], *syndrome, default_edge_weight());
                    }
                }
            }
        }
    }

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::heavy_square_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_with_weights() {
        let g: petgraph::graph::UnGraph<usize, ()> =
            heavy_square_graph(None, Some(vec![0, 1, 2, 3]), || 4, || (), false).unwrap();
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
            heavy_square_graph(Some(4), None, || (), || (), true).unwrap();
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
        match heavy_square_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
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
