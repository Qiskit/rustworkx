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

use std::collections::VecDeque;
use std::hash::Hash;

use petgraph::data::{Build, Create};
use petgraph::visit::{Data, NodeIndexable};

use super::InvalidInputError;

/// Generate a full r-ary tree of `n` nodes.
/// Sometimes called a k-ary, n-ary, or m-ary tree.
///
/// Arguments:
///
/// * `branching factor` - The number of children at each node.
/// * `num_nodes` - The number of nodes in the graph.
/// * `weights` - A list of node weights. If the number of weights is
///   less than n, extra nodes with None weight will be appended.
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
/// use rustworkx_core::generators::full_rary_tree_graph;
/// use rustworkx_core::petgraph::visit::EdgeRef;
///
/// let expected_edge_list = vec![
///     (0, 1),
///     (0, 2),
///     (1, 3),
///     (1, 4),
///     (2, 5),
///     (2, 6),
///     (3, 7),
///     (3, 8),
///     (4, 9),
/// ];
/// let g: petgraph::graph::UnGraph<(), ()> = full_rary_tree_graph(
///     2,
///     10,
///     None,
///     || {()},
///     || {()},
/// ).unwrap();
/// assert_eq!(
///     expected_edge_list,
///     g.edge_references()
///         .map(|edge| (edge.source().index(), edge.target().index()))
///         .collect::<Vec<(usize, usize)>>(),
/// )
/// ```
pub fn full_rary_tree_graph<G, T, F, H, M>(
    branching_factor: usize,
    num_nodes: usize,
    weights: Option<Vec<T>>,
    mut default_node_weight: F,
    mut default_edge_weight: H,
) -> Result<G, InvalidInputError>
where
    G: Build + Create + Data<NodeWeight = T, EdgeWeight = M> + NodeIndexable,
    F: FnMut() -> T,
    H: FnMut() -> M,
    G::NodeId: Eq + Hash,
{
    if let Some(wt) = weights.as_ref() {
        if wt.len() > num_nodes {
            return Err(InvalidInputError {});
        }
    }
    let mut graph = G::with_capacity(num_nodes, num_nodes * branching_factor);

    let nodes: Vec<G::NodeId> = match weights {
        Some(weights) => {
            let mut node_list: Vec<G::NodeId> = Vec::with_capacity(num_nodes);
            let node_count = num_nodes - weights.len();
            for weight in weights {
                let index = graph.add_node(weight);
                node_list.push(index);
            }
            for _ in 0..node_count {
                let index = graph.add_node(default_node_weight());
                node_list.push(index);
            }
            node_list
        }
        None => (0..num_nodes)
            .map(|_| graph.add_node(default_node_weight()))
            .collect(),
    };
    if !nodes.is_empty() {
        let mut parents = VecDeque::from(vec![graph.to_index(nodes[0])]);
        let mut nod_it: usize = 1;

        while !parents.is_empty() {
            let source: usize = parents.pop_front().unwrap(); //If is empty it will never try to pop
            for _ in 0..branching_factor {
                if nod_it < num_nodes {
                    let target: usize = graph.to_index(nodes[nod_it]);
                    parents.push_back(target);
                    nod_it += 1;
                    graph.add_edge(nodes[source], nodes[target], default_edge_weight());
                }
            }
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use crate::generators::full_rary_tree_graph;
    use crate::generators::InvalidInputError;
    use crate::petgraph;
    use crate::petgraph::visit::EdgeRef;

    #[test]
    fn test_full_rary_graph() {
        let expected_edge_list = vec![
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
            (3, 7),
            (3, 8),
            (4, 9),
        ];
        let g: petgraph::graph::UnGraph<(), ()> =
            full_rary_tree_graph(2, 10, None, || (), || ()).unwrap();
        assert_eq!(
            expected_edge_list,
            g.edge_references()
                .map(|edge| (edge.source().index(), edge.target().index()))
                .collect::<Vec<(usize, usize)>>(),
        );
    }

    #[test]
    fn test_full_rary_error() {
        match full_rary_tree_graph::<petgraph::graph::DiGraph<(), ()>, (), _, _, ()>(
            3,
            2,
            Some(vec![(), (), (), ()]),
            || (),
            || (),
        ) {
            Ok(_) => panic!("Returned a non-error"),
            Err(e) => assert_eq!(e, InvalidInputError),
        };
    }
}
