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
use petgraph::graph::NodeIndex;

use super::utils::get_num_nodes;
use super::InvalidInputError;

/// Generate a cycle graph
///
/// Arguments:
///
/// * `num_nodes` - The number of nodes to create a cycle graph for. Either this or
///     `weights must be specified. If both this and `weights are specified, weights
///     will take priorty and this argument will be ignored.
/// * `weights` - A `Vec` of node weight objects.
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
/// use rustworkx_core::generators::cycle_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let g: petgraph::graph::UnGraph<(), ()> = cycle_graph(
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
pub fn cycle_graph<G, T, F, H, M>(
    num_nodes: Option<usize>,
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
                graph.add_node();
            }
        }
        None => {
            (0..num_nodes).for_each(|_| {
                graph.add_node();
            });
        }
    };

    for i in 0..rowlen {
        for j in 0..collen {
            if i + 1 < rowlen {
                graph.add_edge(
                    NodeIndex::new(i * collen + j),
                    NodeIndex::new((i + 1) * collen + j),
                    None,
                );
                if bidirectional {
                    graph.add_edge(
                        NodeIndex::new((i + 1) * collen + j),
                        NodeIndex::new(i * collen + j),
                        None,
                    );
                }
            }

            if j + 1 < collen {
                graph.add_edge(
                    NodeIndex::new(i * collen + j),
                    NodeIndex::new(i * collen + j + 1),
                    None,
                );
                if bidirectional {
                    graph.add_edge(
                        NodeIndex::new(i * collen + j + 1),
                        NodeIndex::new(i * collen + j),
                        None,
                    );
                }
            }
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::cycle_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_with_weights() {
        let g: petgraph::graph::UnGraph<usize, ()> =
            cycle_graph(None, Some(vec![0, 1, 2, 3]), || 4, || (), false).unwrap();
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
            cycle_graph(Some(4), None, || (), || (), true).unwrap();
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
        match cycle_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
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
