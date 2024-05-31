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

use petgraph::{data::Create, visit::Data};

use super::InvalidInputError;

/// Generate a Dorogovtsev-Goltsev-Mendes graph
///
/// Generate a graph following the recursive procedure in [1].
/// Starting from the two-node, one-edge graph, iterating `n` times generates
/// a graph with `(3**n + 3) // 2` nodes and `3**n` edges.
///
///
/// Arguments:
///
/// * `n` - The number of iterations to perform. n=0 returns the two-node, one-edge graph.
/// * `default_node_weight` - A callable that will return the weight to use for newly created nodes.
/// * `default_edge_weight` - A callable that will return the weight object to use for newly created edges.
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::generators::dorogovtsev_goltsev_mendes_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let g: petgraph::graph::UnGraph<(), ()> = dorogovtsev_goltsev_mendes_graph(2, || (), || ()).unwrap();
/// assert_eq!(g.node_count(), 6);
/// assert_eq!(
///     vec![(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (2, 4), (1, 5), (2, 5)],
///     g.edge_references()
///       .map(|edge| (edge.source().index(), edge.target().index()))
///       .collect::<Vec<(usize, usize)>>(),
/// );
/// ```
///
/// .. [1] S. N. Dorogovtsev, A. V. Goltsev and J. F. F. Mendes
///    “Pseudofractal scale-free web”
///    Physical Review E 65, 066122, 2002
///    https://arxiv.org/abs/cond-mat/0112143
///
pub fn dorogovtsev_goltsev_mendes_graph<G, T, F, H, M>(
    n: usize,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: Create + Data<NodeWeight = T, EdgeWeight = M>,
    F: FnMut() -> T,
    H: FnMut() -> M,
{
    let n_edges = usize::pow(3, n as u32);
    let n_nodes = (n_edges + 3) / 2;
    let mut graph = G::with_capacity(n_nodes, n_edges);

    let node_0 = graph.add_node(default_node_weight());
    let node_1 = graph.add_node(default_node_weight());
    graph
        .add_edge(node_0, node_1, default_edge_weight())
        .unwrap();
    let mut current_endpoints = vec![(node_0, node_1)];

    for _ in 0..n {
        let mut new_endpoints = vec![];
        for (source, target) in current_endpoints.iter() {
            let new_node = graph.add_node(default_node_weight());
            graph.add_edge(*source, new_node, default_edge_weight());
            new_endpoints.push((*source, new_node));
            graph.add_edge(*target, new_node, default_edge_weight());
            new_endpoints.push((*target, new_node));
        }
        current_endpoints.extend(new_endpoints);
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::dorogovtsev_goltsev_mendes_graph;
    use crate::petgraph::graph::Graph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_dorogovtsev_goltsev_mendes_graph() {
        for n in 0..6 {
            let graph: Graph<(), ()> = match dorogovtsev_goltsev_mendes_graph(n, || (), || ()) {
                Ok(graph) => graph,
                Err(_) => panic!("Error generating graph"),
            };
            assert_eq!(graph.node_count(), (usize::pow(3, n as u32) + 3) / 2);
            assert_eq!(graph.edge_count(), usize::pow(3, n as u32));
        }
    }

    #[test]
    fn test_dorogovtsev_goltsev_mendes_graph_edges() {
        let n = 2;
        let expected_edge_list = vec![
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (0, 4),
            (2, 4),
            (1, 5),
            (2, 5),
        ];
        let graph: Graph<(), ()> = match dorogovtsev_goltsev_mendes_graph(n, || (), || ()) {
            Ok(graph) => graph,
            Err(_) => panic!("Error generating graph"),
        };
        assert_eq!(
            expected_edge_list,
            graph
                .edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        )
    }
}
