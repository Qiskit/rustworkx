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
use petgraph::visit::{Data, GraphProp, NodeIndexable};

use super::utils::get_num_nodes;
use super::InvalidInputError;

/// Generate a complete graph
///
/// Arguments:
///
/// * `num_nodes` - The number of nodes to create a complete graph for. Either this or
///     `weights` must be specified. If both this and `weights` are specified, `weights`
///     will take priority and this argument will be ignored
/// * `weights` - A `Vec` of node weight objects.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes. This is ignored if `weights` is specified.
/// * `default_edge_weight` - A callable that will return the weight object
///     to use for newly created edges.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::complete_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let g: petgraph::graph::UnGraph<(), ()> = complete_graph(
///     Some(4),
///     None,
///     || {()},
///     || {()},
/// ).unwrap();
/// assert_eq!(
///     vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
///     g.edge_references()
///         .map(|edge| (edge.source().index(), edge.target().index()))
///         .collect::<Vec<(usize, usize)>>(),
/// )
/// ```
pub fn complete_graph<G, T, F, H, M>(
    num_nodes: Option<usize>,
    weights: Option<Vec<T>>,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable + GraphProp,
    F: FnMut() -> T,
    H: FnMut() -> M,
{
    if weights.is_none() && num_nodes.is_none() {
        return Err(InvalidInputError {});
    }
    let node_len = get_num_nodes(&num_nodes, &weights);
    let mut graph = G::with_capacity(node_len, node_len);
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
    for i in 0..node_len - 1 {
        for j in i + 1..node_len {
            let node_i = graph.from_index(i);
            let node_j = graph.from_index(j);
            graph.add_edge(node_i, node_j, default_edge_weight());
            if graph.is_directed() {
                graph.add_edge(node_j, node_i, default_edge_weight());
            }
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::complete_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph::graph::{DiGraph, NodeIndex, UnGraph};
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_directed_complete_graph() {
        let g: DiGraph<(), ()> = complete_graph(Some(10), None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 10);
        assert_eq!(g.edge_count(), 90);
        let mut elist = vec![];
        for i in 0..10 {
            for j in i..10 {
                if i != j {
                    elist.push((i, j));
                    elist.push((j, i));
                }
            }
        }
        assert_eq!(
            elist,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_directed_complete_graph_weights() {
        let g: DiGraph<usize, ()> =
            complete_graph(None, Some(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), || 4, || ()).unwrap();
        assert_eq!(g.node_count(), 10);
        assert_eq!(g.edge_count(), 90);
        let mut elist = vec![];
        for i in 0..10 {
            for j in i..10 {
                if i != j {
                    elist.push((i, j));
                    elist.push((j, i));
                }
            }
            assert_eq!(*g.node_weight(NodeIndex::new(i)).unwrap(), i);
        }
        assert_eq!(
            elist,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_compete_graph_error() {
        match complete_graph::<DiGraph<(), ()>, (), _, _, ()>(None, None, || (), || ()) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }

    #[test]
    fn test_complete_graph() {
        let g: UnGraph<(), ()> = complete_graph(Some(10), None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 10);
        assert_eq!(g.edge_count(), 45);
        let mut elist = vec![];
        for i in 0..10 {
            for j in i..10 {
                if i != j {
                    elist.push((i, j));
                }
            }
        }
        assert_eq!(
            elist,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_complete_graph_weights() {
        let g: UnGraph<usize, ()> =
            complete_graph(None, Some(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), || 4, || ()).unwrap();
        assert_eq!(g.node_count(), 10);
        assert_eq!(g.edge_count(), 45);
        let mut elist = vec![];
        for i in 0..10 {
            for j in i..10 {
                if i != j {
                    elist.push((i, j));
                }
            }
            assert_eq!(*g.node_weight(NodeIndex::new(i)).unwrap(), i);
        }
        assert_eq!(
            elist,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }
}
