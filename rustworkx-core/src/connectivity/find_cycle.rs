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

use hashbrown::{HashMap, HashSet};
use petgraph::algo;
use petgraph::visit::{
    EdgeCount, GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, Visitable,
};
use petgraph::Direction::Outgoing;
use std::hash::Hash;

/// Return the first cycle encountered during DFS of a given directed graph.
/// Empty list is returned if no cycle is found.
///
/// Arguments:
///
/// * `graph` - The directed graph in which to find the first cycle.
/// * `source` - Optional node index for starting the search. If not specified,
///     an arbitrary node is chosen to start the search.
///
/// # Example
/// ```rust
/// use petgraph::prelude::*;
/// use rustworkx_core::connectivity::find_cycle;
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
/// let graph = DiGraph::<i32, i32>::from_edges(&edge_list);
/// let mut res: Vec<(usize, usize)> = find_cycle(&graph, Some(NodeIndex::new(0)))
///     .iter()
///     .map(|(s, t)| (s.index(), t.index()))
///     .collect();
/// assert_eq!(res, [(0, 1), (1, 2), (2, 3), (3, 0)]);
/// ```
pub fn find_cycle<G>(graph: G, source: Option<G::NodeId>) -> Vec<(G::NodeId, G::NodeId)>
where
    G: Copy,
    G: GraphBase,
    G: NodeCount,
    G: EdgeCount,
    for<'b> &'b G:
        GraphBase<NodeId = G::NodeId> + IntoNodeIdentifiers + IntoNeighborsDirected + Visitable,
    G::NodeId: Eq + Hash,
{
    // Find a cycle in the given graph and return it as a list of edges
    let mut cycle: Vec<(G::NodeId, G::NodeId)> = Vec::with_capacity(graph.edge_count());
    // If source is not set get a node in an arbitrary cycle if it exists,
    // otherwise return that there is no cycle
    let source_index = match source {
        Some(source_value) => source_value,
        None => match find_node_in_arbitrary_cycle(graph) {
            Some(node_in_cycle) => node_in_cycle,
            None => {
                return Vec::new();
            }
        },
    };
    // Stack (ie "pushdown list") of vertices already in the spanning tree
    let mut stack: Vec<G::NodeId> = vec![source_index];
    // map to store parent of a node
    let mut pred: HashMap<G::NodeId, G::NodeId> = HashMap::new();
    // a node is in the visiting set if at least one of its child is unexamined
    let mut visiting = HashSet::new();
    // a node is in visited set if all of its children have been examined
    let mut visited = HashSet::new();
    while !stack.is_empty() {
        let mut z = *stack.last().unwrap();
        visiting.insert(z);

        let children = graph.neighbors_directed(z, Outgoing);
        for child in children {
            //cycle is found
            if visiting.contains(&child) {
                cycle.push((z, child));
                //backtrack
                loop {
                    if z == child {
                        cycle.reverse();
                        break;
                    }
                    cycle.push((pred[&z], z));
                    z = pred[&z];
                }
                return cycle;
            }
            //if an unexplored node is encountered
            if !visited.contains(&child) {
                stack.push(child);
                pred.insert(child, z);
            }
        }
        let top = *stack.last().unwrap();
        //if no further children and explored, move to visited
        if top == z {
            stack.pop();
            visiting.remove(&z);
            visited.insert(z);
        }
    }
    cycle
}

fn find_node_in_arbitrary_cycle<G>(graph: G) -> Option<G::NodeId>
where
    G: GraphBase,
    G: NodeCount,
    G: EdgeCount,
    for<'b> &'b G:
        GraphBase<NodeId = G::NodeId> + IntoNodeIdentifiers + IntoNeighborsDirected + Visitable,
    G::NodeId: Eq + Hash,
{
    for scc in algo::kosaraju_scc(&graph) {
        if scc.len() > 1 {
            // TODO: build
        } else {
            // TODO: check
            let node = scc[0];
            return Some(node);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::connectivity::find_cycle;
    use petgraph::prelude::*;

    #[test]
    fn test_find_cycle_source() {
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
        let graph = DiGraph::<i32, i32>::from_edges(edge_list);
        let mut res: Vec<(usize, usize)> = find_cycle(&graph, Some(NodeIndex::new(0)))
            .iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect();
        assert_eq!(res, [(0, 1), (1, 2), (2, 3), (3, 0)]);
        res = find_cycle(&graph, Some(NodeIndex::new(1)))
            .iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect();
        assert_eq!(res, [(1, 2), (2, 3), (3, 0), (0, 1)]);
        res = find_cycle(&graph, Some(NodeIndex::new(5)))
            .iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect();
        assert_eq!(res, []);
    }

    #[test]
    fn test_self_loop() {
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
        let mut graph = DiGraph::<i32, i32>::from_edges(edge_list);
        graph.add_edge(NodeIndex::new(1), NodeIndex::new(1), 0);
        let res: Vec<(usize, usize)> = find_cycle(&graph, Some(NodeIndex::new(0)))
            .iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect();
        assert_eq!(res, [(1, 1)]);
    }
}
