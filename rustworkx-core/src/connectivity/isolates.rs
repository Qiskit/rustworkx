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

use petgraph::visit::{IntoNeighborsDirected, IntoNodeIdentifiers, NodeIndexable};
use petgraph::Direction::{Incoming, Outgoing};

/// Return the isolates in a graph object
///
/// An isolate is a node without any neighbors meaning it has a degree of 0. For
/// directed graphs this means the in-degree and out-degree are both 0.
///
/// Arguments:
///
/// * `graph` - The graph in which to find the isolates.
///
/// # Example
/// ```rust
/// use petgraph::prelude::*;
/// use rustworkx_core::connectivity::isolates;
///
/// let edge_list = vec![
///     (0, 1),
///     (3, 0),
///     (0, 5),
///     (8, 0),
///     (1, 2),
///     (1, 6),
///     (2, 3),
///     (3, 4),
///     (4, 5),
///     (6, 7),
///     (7, 8),
///     (8, 9),
/// ];
/// let mut graph = DiGraph::<i32, i32>::from_edges(&edge_list);
/// graph.add_node(10);
/// graph.add_node(11);
/// let res: Vec<usize> = isolates(&graph).into_iter().map(|x| x.index()).collect();
/// assert_eq!(res, [10, 11])
/// ```
pub fn isolates<G>(graph: G) -> Vec<G::NodeId>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoNeighborsDirected,
{
    graph
        .node_identifiers()
        .filter(|x| {
            graph
                .neighbors_directed(*x, Incoming)
                .chain(graph.neighbors_directed(*x, Outgoing))
                .next()
                .is_none()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::connectivity::isolates;
    use petgraph::prelude::*;

    #[test]
    fn test_isolates_directed_empty() {
        let graph = DiGraph::<i32, i32>::new();
        let res: Vec<NodeIndex> = isolates(&graph);
        assert_eq!(res, []);
    }

    #[test]
    fn test_isolates_undirected_empty() {
        let graph = UnGraph::<i32, i32>::default();
        let res: Vec<NodeIndex> = isolates(&graph);
        assert_eq!(res, []);
    }

    #[test]
    fn test_isolates_directed_no_isolates() {
        let graph = DiGraph::<i32, i32>::from_edges([(0, 1), (1, 2)]);
        let res: Vec<NodeIndex> = isolates(&graph);
        assert_eq!(res, []);
    }

    #[test]
    fn test_isolates_undirected_no_isolates() {
        let graph = UnGraph::<i32, i32>::from_edges([(0, 1), (1, 2)]);
        let res: Vec<NodeIndex> = isolates(&graph);
        assert_eq!(res, []);
    }

    #[test]
    fn test_isolates_directed() {
        let edge_list = vec![
            (0, 1),
            (3, 0),
            (0, 5),
            (8, 0),
            (1, 2),
            (1, 6),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
        ];
        let mut graph = DiGraph::<i32, i32>::from_edges(&edge_list);
        graph.add_node(10);
        graph.add_node(11);
        let res: Vec<usize> = isolates(&graph).into_iter().map(|x| x.index()).collect();
        assert_eq!(res, [10, 11])
    }

    #[test]
    fn test_isolates_undirected() {
        let edge_list = vec![
            (0, 1),
            (3, 0),
            (0, 5),
            (8, 0),
            (1, 2),
            (1, 6),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
        ];
        let mut graph = UnGraph::<i32, i32>::from_edges(&edge_list);
        graph.add_node(10);
        graph.add_node(11);
        let res: Vec<usize> = isolates(&graph).into_iter().map(|x| x.index()).collect();
        assert_eq!(res, [10, 11])
    }
}
