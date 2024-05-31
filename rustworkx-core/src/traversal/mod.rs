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

//! Module for graph traversal algorithms.

mod bfs_visit;
mod dfs_edges;
mod dfs_visit;
mod dijkstra_visit;

use petgraph::prelude::*;
use petgraph::visit::GraphRef;
use petgraph::visit::IntoNeighborsDirected;
use petgraph::visit::Reversed;
use petgraph::visit::VisitMap;
use petgraph::visit::Visitable;

pub use bfs_visit::{breadth_first_search, BfsEvent};
pub use dfs_edges::dfs_edges;
pub use dfs_visit::{depth_first_search, DfsEvent};
pub use dijkstra_visit::{dijkstra_search, DijkstraEvent};

/// Return if the expression is a break value, execute the provided statement
/// if it is a prune value.
/// https://github.com/petgraph/petgraph/blob/0.6.0/src/visit/dfsvisit.rs#L27
macro_rules! try_control {
    ($e:expr, $p:stmt) => {
        try_control!($e, $p, ());
    };
    ($e:expr, $p:stmt, $q:stmt) => {
        match $e {
            x => {
                if x.should_break() {
                    return x;
                } else if x.should_prune() {
                    $p
                } else {
                    $q
                }
            }
        }
    };
}

use try_control;

struct AncestryWalker<G, N, VM> {
    graph: G,
    walker: Bfs<N, VM>,
}

impl<
        G: GraphRef + Visitable + IntoNeighborsDirected<NodeId = N>,
        N: Copy + Clone + PartialEq,
        VM: VisitMap<N>,
    > Iterator for AncestryWalker<G, N, VM>
{
    type Item = N;
    fn next(&mut self) -> Option<Self::Item> {
        self.walker.next(self.graph)
    }
}

/// Return the ancestors of a node in a graph.
///
/// `node` is included in the output
///
/// # Arguments:
///
/// * `node` - The node to find the ancestors of
///
/// # Returns
///
/// An iterator where each item is a node id for an ancestor of ``node``.
/// This includes ``node`` in the returned ids.
///
/// # Example
///
/// ```rust
/// use rustworkx_core::traversal::ancestors;
/// use rustworkx_core::petgraph::stable_graph::{StableDiGraph, NodeIndex};
///
/// let graph: StableDiGraph<(), ()> = StableDiGraph::from_edges(&[
///     (0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)
/// ]);
/// let ancestors: Vec<usize> = ancestors(&graph, NodeIndex::new(3)).map(|x| x.index()).collect();
/// assert_eq!(vec![3_usize, 1, 0], ancestors);
/// ```
pub fn ancestors<G>(graph: G, node: G::NodeId) -> impl Iterator<Item = G::NodeId>
where
    G: GraphRef + Visitable + IntoNeighborsDirected,
{
    let reversed = Reversed(graph);
    AncestryWalker {
        graph: reversed,
        walker: Bfs::new(reversed, node),
    }
}

/// Return the descendants of a node in a graph.
///
/// `node` is included in the output.
/// # Arguments:
///
/// * `node` - The node to find the ancestors of
///
/// # Returns
///
/// An iterator where each item is a node id for an ancestor of ``node``.
/// This includes ``node`` in the returned ids.
///
/// # Example
///
/// ```rust
/// use rustworkx_core::traversal::descendants;
/// use rustworkx_core::petgraph::stable_graph::{StableDiGraph, NodeIndex};
///
/// let graph: StableDiGraph<(), ()> = StableDiGraph::from_edges(&[
///     (0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)
/// ]);
/// let descendants: Vec<usize> = descendants(&graph, NodeIndex::new(3)).map(|x| x.index()).collect();
/// assert_eq!(vec![3_usize, 4, 5], descendants);
/// ```
pub fn descendants<G>(graph: G, node: G::NodeId) -> impl Iterator<Item = G::NodeId>
where
    G: GraphRef + Visitable + IntoNeighborsDirected,
{
    AncestryWalker {
        graph,
        walker: Bfs::new(graph, node),
    }
}

#[cfg(test)]
mod test_ancestry {
    use super::{ancestors, descendants};
    use crate::petgraph::graph::DiGraph;
    use crate::petgraph::stable_graph::{NodeIndex, StableDiGraph};

    #[test]
    fn test_ancestors_digraph() {
        let graph: DiGraph<(), ()> =
            DiGraph::from_edges(&[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]);
        let ancestors: Vec<usize> = ancestors(&graph, NodeIndex::new(3))
            .map(|x| x.index())
            .collect();
        assert_eq!(vec![3_usize, 1, 0], ancestors);
    }

    #[test]
    fn test_descendants() {
        let graph: DiGraph<(), ()> =
            DiGraph::from_edges(&[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]);
        let descendants: Vec<usize> = descendants(&graph, NodeIndex::new(3))
            .map(|x| x.index())
            .collect();
        assert_eq!(vec![3_usize, 4, 5], descendants);
    }

    #[test]
    fn test_no_ancestors() {
        let mut graph: StableDiGraph<(), ()> = StableDiGraph::new();
        let index = graph.add_node(());
        let res = ancestors(&graph, index);
        assert_eq!(vec![index], res.collect::<Vec<NodeIndex>>())
    }

    #[test]
    fn test_no_descendants() {
        let mut graph: StableDiGraph<(), ()> = StableDiGraph::new();
        let index = graph.add_node(());
        let res = descendants(&graph, index);
        assert_eq!(vec![index], res.collect::<Vec<NodeIndex>>())
    }
}
