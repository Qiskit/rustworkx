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
use petgraph::visit::{Data, EdgeRef, GraphBase, IntoEdgeReferences, NodeIndexable};

use super::InvalidInputError;

/// Generate a binomial tree graph
///
/// Arguments:
///
/// * `order` - The order of the binomial tree.
/// * `weights` - A `Vec` of node weight objects. If the number of weights is
///     less than 2**order, extra nodes with None will be appended.
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
/// use rustworkx_core::generators::binomial_tree_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let expected_edge_list = vec![
///     (0, 1),
///     (2, 3),
///     (0, 2),
///     (4, 5),
///     (6, 7),
///     (4, 6),
///     (0, 4),
///     (8, 9),
///     (10, 11),
///     (8, 10),
///     (12, 13),
///     (14, 15),
///     (12, 14),
///     (8, 12),
///     (0, 8),
/// ];
/// let g: petgraph::graph::UnGraph<(), ()> = binomial_tree_graph(
///     4,
///     None,
///     || {()},
///     || {()},
///     false
/// ).unwrap();
/// assert_eq!(
///     expected_edge_list,
///     g.edge_references()
///         .map(|edge| (edge.source().index(), edge.target().index()))
///         .collect::<Vec<(usize, usize)>>(),
/// )
/// ```
pub fn binomial_tree_graph<G, T, F, H, M>(
    order: u32,
    weights: Option<Vec<T>>,
    mut default_node_weight: F,
    mut default_edge_weight: H,
    bidirectional: bool,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoEdgeReferences + Copy,
    F: FnMut() -> T,
    H: FnMut() -> M,
    T: Clone,
{
    if order >= 60 {
        return Err(InvalidInputError {});
    }
    let num_nodes = usize::pow(2, order);
    let num_edges = usize::pow(2, order) - 1;
    let mut graph = G::with_capacity(num_nodes, num_edges);

    for i in 0..num_nodes {
        match weights {
            Some(ref weights) => {
                if weights.len() > num_nodes {
                    return Err(InvalidInputError {});
                }
                if i < weights.len() {
                    graph.add_node(weights[i].clone())
                } else {
                    graph.add_node(default_node_weight())
                }
            }
            None => graph.add_node(default_node_weight()),
        };
    }

    fn find_edge<G>(graph: &mut G, source: usize, target: usize) -> bool
    where
        G: NodeIndexable,
        for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoEdgeReferences,
    {
        let mut found = false;
        for edge in graph.edge_references() {
            if graph.to_index(edge.source()) == source && graph.to_index(edge.target()) == target {
                found = true;
                break;
            }
        }
        found
    }

    let mut n = 1;
    let zero_index = 0;
    for _ in 0..order {
        let edges: Vec<(usize, usize)> = graph
            .edge_references()
            .map(|e| (graph.to_index(e.source()), graph.to_index(e.target())))
            .collect();

        for (source, target) in edges {
            let source_index = source + n;
            let target_index = target + n;

            if !find_edge(&mut graph, source_index, target_index) {
                graph.add_edge(
                    graph.from_index(source_index),
                    graph.from_index(target_index),
                    default_edge_weight(),
                );
            }
            if bidirectional && !find_edge(&mut graph, target_index, source_index) {
                graph.add_edge(
                    graph.from_index(target_index),
                    graph.from_index(source_index),
                    default_edge_weight(),
                );
            }
        }
        if !find_edge(&mut graph, zero_index, n) {
            graph.add_edge(
                graph.from_index(zero_index),
                graph.from_index(n),
                default_edge_weight(),
            );
        }
        if bidirectional && !find_edge(&mut graph, n, zero_index) {
            graph.add_edge(
                graph.from_index(n),
                graph.from_index(zero_index),
                default_edge_weight(),
            );
        }
        n *= 2;
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::binomial_tree_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_binomial_tree_graph() {
        let expected_edge_list = vec![
            (0, 1),
            (2, 3),
            (0, 2),
            (4, 5),
            (6, 7),
            (4, 6),
            (0, 4),
            (8, 9),
            (10, 11),
            (8, 10),
            (12, 13),
            (14, 15),
            (12, 14),
            (8, 12),
            (0, 8),
        ];
        let g: petgraph::graph::UnGraph<(), ()> =
            binomial_tree_graph(4, None, || (), || (), false).unwrap();
        assert_eq!(
            expected_edge_list,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_directed_binomial_tree_graph() {
        let expected_edge_list = vec![
            (0, 1),
            (2, 3),
            (0, 2),
            (4, 5),
            (6, 7),
            (4, 6),
            (0, 4),
            (8, 9),
            (10, 11),
            (8, 10),
            (12, 13),
            (14, 15),
            (12, 14),
            (8, 12),
            (0, 8),
        ];
        let g: petgraph::graph::DiGraph<(), ()> =
            binomial_tree_graph(4, None, || (), || (), false).unwrap();
        assert_eq!(
            expected_edge_list,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_binomial_tree_error() {
        match binomial_tree_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
            75,
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
