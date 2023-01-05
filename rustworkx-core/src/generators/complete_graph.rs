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

//use petgraph::GraphProp;
use petgraph::data::{Build, Create};
use petgraph::visit::{Data, GraphProp, NodeIndexable};

use super::utils::get_num_nodes;
use super::InvalidInputError;

/// Generate a complete graph
///
/// Arguments:
///
/// * `num_nodes` - The number of nodes to create a complete graph for. Either this or
///     `weights must be specified. If both this and `weights are specified, weights
///     will take priorty and this argument will be ignored
/// * `weights` - A `Vec` of node weight objects.
/// * `default_node_weight` - A callable that will return the weight to use
///     for newly created nodes. This is ignored if `weights` is specified,
///     as the weights from that argument will be used instead.
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
///     false
/// ).unwrap();
/// assert_eq!(
///     vec![(0, 1), (1, 2), (2, 3)],
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
    use crate::petgraph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_directed_complete_graph() {
        let g: petgraph::graph::DiGraph<(), ()> =
            complete_graph(Some(10), None, || (), || ()).unwrap();
        assert_eq!(g.node_count(), 10);
        assert_eq!(g.edge_count(), 90);
        let elist = vec![];
        for i in 0..10 {
            for j in 0..19_i32.iter().rev() {
                if i != j {
                    elist.push((i, j));
                }
            }
            assert_eq!(g.edges(i), elist);
        }
    //         ls = []
    //         for j in range(19, -1, -1):
    //             if i != j:
    //                 ls.append((i, j, None))
    //         self.assertEqual(graph.out_edges(i), ls)
    }

    #[test]
    fn test_directed_complete_graph_weights() {
    }

    #[test]
    fn test_error() {
        match complete_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
            None,
            None,
            || (),
            || (),
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }
}
    //     graph = rustworkx.generators.directed_mesh_graph(20)
    //     self.assertEqual(len(graph), 20)
    //     self.assertEqual(len(graph.edges()), 380)
    //     for i in range(20):
    //         ls = []
    //         for j in range(19, -1, -1):
    //             if i != j:
    //                 ls.append((i, j, None))
    //         self.assertEqual(graph.out_edges(i), ls)

    // def test_directed_mesh_graph_weights(self):
    //     graph = rustworkx.generators.directed_mesh_graph(weights=list(range(20)))
    //     self.assertEqual(len(graph), 20)
    //     self.assertEqual([x for x in range(20)], graph.nodes())
    //     self.assertEqual(len(graph.edges()), 380)
    //     for i in range(20):
    //         ls = []
    //         for j in range(19, -1, -1):
    //             if i != j:
    //                 ls.append((i, j, None))
    //         self.assertEqual(graph.out_edges(i), ls)

    // def test_mesh_directed_no_weights_or_num(self):
    //     with self.assertRaises(IndexError):
    //         rustworkx.generators.directed_mesh_graph()

    // def test_mesh_graph(self):
    //     graph = rustworkx.generators.mesh_graph(20)
    //     self.assertEqual(len(graph), 20)
    //     self.assertEqual(len(graph.edges()), 190)

    // def test_mesh_graph_weights(self):
    //     graph = rustworkx.generators.mesh_graph(weights=list(range(20)))
    //     self.assertEqual(len(graph), 20)
    //     self.assertEqual([x for x in range(20)], graph.nodes())
    //     self.assertEqual(len(graph.edges()), 190)

    // def test_mesh_no_weights_or_num(self):
    //     with self.assertRaises(IndexError):
    //         rustworkx.generators.mesh_graph()

    // def test_zero_size_mesh_graph(self):
    //     graph = rustworkx.generators.mesh_graph(0)
    //     self.assertEqual(0, len(graph))

    // def test_zero_size_directed_mesh_graph(self):
    //     graph = rustworkx.generators.directed_mesh_graph(0)
    //     self.assertEqual(0, len(graph))
