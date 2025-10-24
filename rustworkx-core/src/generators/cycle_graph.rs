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
use super::utils::get_num_nodes;

/// Generate a cycle graph
///
/// Arguments:
///
/// * `num_nodes` - The number of nodes to create a cycle graph for. Either this or
///   `weights` must be specified. If both this and `weights` are specified, `weights`
///   will take priority and this argument will be ignored.
/// * `weights` - A `Vec` of node weight objects.
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
    if weights.is_none() && num_nodes.is_none() {
        return Err(InvalidInputError {});
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    let num_edges = if bidirectional {
        2 * node_len
    } else {
        node_len
    };
    let mut graph = G::with_capacity(node_len, num_edges);
    if node_len == 0 {
        return Ok(graph);
    }

    match weights {
        Some(weights) => {
            for weight in weights {
                graph.add_node(weight);
            }
        }
        None => {
            for _ in 0..node_len {
                graph.add_node(default_node_weight());
            }
        }
    };
    for a in 0..node_len - 1 {
        let node_a = graph.from_index(a);
        let node_b = graph.from_index(a + 1);
        graph.add_edge(node_a, node_b, default_edge_weight());
        if bidirectional {
            graph.add_edge(node_b, node_a, default_edge_weight());
        }
    }
    let last_node_index = graph.from_index(node_len - 1);
    let first_node_index = graph.from_index(0);
    graph.add_edge(last_node_index, first_node_index, default_edge_weight());
    if bidirectional {
        graph.add_edge(first_node_index, last_node_index, default_edge_weight());
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::InvalidInputError;
    use crate::generators::cycle_graph;
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
