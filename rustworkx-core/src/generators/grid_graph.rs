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

use petgraph::data::{Build, Create};
use petgraph::visit::{Data, NodeIndexable};

use super::InvalidInputError;

/// Generate a grid graph
///
/// Arguments:
///
/// * `rows` - The number of rows to generate the graph with.
///   If specified, cols also need to be specified.
/// * `cols`: The number of columns to generate the graph with.
///   If specified, rows also need to be specified. rows*cols
///   defines the number of nodes in the graph.
/// * `weights`: A `Vec` of node weights. Nodes are filled row wise.
///   If rows and cols are not specified, then a linear graph containing
///   all the values in weights list is created.
///   If number of nodes(rows*cols) is less than length of
///   weights list, the trailing weights are ignored.
///   If number of nodes(rows*cols) is greater than length of
///   weights list, extra nodes with None weight are appended.
/// * `default_node_weight` - A callable that will return the weight to use
///   for newly created nodes. This is ignored if `weights` is specified.
/// * `default_edge_weight` - A callable that will return the weight object
///   to use for newly created edges.
/// * `bidirectional` - Whether edges are added bidirectionally. If set to
///   `true` then for any edge `(u, v)` an edge `(v, u)` will also be added.
///   If the graph is undirected this will result in a parallel edge.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::grid_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let g: petgraph::graph::UnGraph<(), ()> = grid_graph(
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
pub fn grid_graph<G, T, F, H, M>(
    rows: Option<usize>,
    cols: Option<usize>,
    weights: Option<Vec<T>>,
    mut default_node_weight: F,
    mut default_edge_weight: H,
    bidirectional: bool,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut() -> T,
    H: FnMut() -> M,
{
    if weights.is_none() && (rows.is_none() || cols.is_none()) {
        return Err(InvalidInputError {});
    }
    let mut rowlen = rows.unwrap_or(0);
    let mut collen = cols.unwrap_or(0);
    let mut num_nodes = rowlen * collen;
    let mut num_edges = 0;
    if num_nodes != 0 {
        num_edges = (rowlen - 1) * collen + (collen - 1) * rowlen;
    }
    if bidirectional {
        num_edges *= 2;
    }
    let mut graph = G::with_capacity(num_nodes, num_edges);
    if num_nodes == 0 && weights.is_none() {
        return Ok(graph);
    }
    match weights {
        Some(weights) => {
            if num_nodes < weights.len() && rowlen == 0 {
                collen = weights.len();
                rowlen = 1;
                num_nodes = collen;
            }

            let mut node_cnt = num_nodes;

            for weight in weights {
                if node_cnt == 0 {
                    break;
                }
                graph.add_node(weight);
                node_cnt -= 1;
            }
            for _i in 0..node_cnt {
                graph.add_node(default_node_weight());
            }
        }
        None => {
            (0..num_nodes).for_each(|_| {
                graph.add_node(default_node_weight());
            });
        }
    };

    for i in 0..rowlen {
        for j in 0..collen {
            if i + 1 < rowlen {
                let node_a = graph.from_index(i * collen + j);
                let node_b = graph.from_index((i + 1) * collen + j);
                graph.add_edge(node_a, node_b, default_edge_weight());
                if bidirectional {
                    let node_a = graph.from_index((i + 1) * collen + j);
                    let node_b = graph.from_index(i * collen + j);
                    graph.add_edge(node_a, node_b, default_edge_weight());
                }
            }

            if j + 1 < collen {
                let node_a = graph.from_index(i * collen + j);
                let node_b = graph.from_index(i * collen + j + 1);
                graph.add_edge(node_a, node_b, default_edge_weight());
                if bidirectional {
                    let node_a = graph.from_index(i * collen + j + 1);
                    let node_b = graph.from_index(i * collen + j);
                    graph.add_edge(node_a, node_b, default_edge_weight());
                }
            }
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::grid_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_directed_grid_simple_row_col() {
        let g: petgraph::graph::DiGraph<(), ()> =
            grid_graph(Some(3), Some(3), None, || (), || (), false).unwrap();
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
    fn test_grid_simple_row_col() {
        let g: petgraph::graph::UnGraph<(), ()> =
            grid_graph(Some(3), Some(3), None, || (), || (), false).unwrap();
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
    fn test_directed_grid_weights() {
        let g: petgraph::graph::DiGraph<usize, ()> = grid_graph(
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
    fn test_directed_grid_more_weights() {
        let g: petgraph::graph::DiGraph<usize, ()> = grid_graph(
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
    fn test_directed_grid_less_weights() {
        let g: petgraph::graph::DiGraph<usize, ()> =
            grid_graph(Some(2), Some(3), Some(vec![0, 1, 2, 3]), || 6, || (), false).unwrap();
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
    fn test_directed_grid_bidirectional() {
        let g: petgraph::graph::DiGraph<(), ()> =
            grid_graph(Some(2), Some(3), None, || (), || (), true).unwrap();
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
    fn test_grid_error() {
        match grid_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
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
