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

use hashbrown::HashSet;
use std::collections::VecDeque;
use std::hash::Hash;

use petgraph::{
    visit::{
        EdgeCount, GraphProp, IntoEdges, IntoNodeIdentifiers, NodeIndexable,
        Visitable, VisitMap, IntoNodeReferences
    },
    Undirected,
};

/// Given an undirected graph, a start node and the visit_map for
/// the graph, this function returns a connected component set.
///
/// :param Graph graph: The input graph to find the connected
///     components for.
/// :param NodeIndex start: The node to start from.
/// :param Visitable::Map discovered: The visit map for the graph.
///
/// :return: A set of connected components for the start node
/// :rtype: HashSet<usize>
pub fn bfs_undirected<G>(
    graph: G,
    start: G::NodeId,
    discovered: &mut G::Map,
) -> HashSet<G::NodeId>
where
    G: GraphProp<EdgeType = Undirected>
        + EdgeCount
        + IntoEdges
        + Visitable
        + NodeIndexable
        + IntoNodeIdentifiers
        + IntoNodeReferences,
    G::NodeId: Eq + Hash,
{
    let mut component = HashSet::new();
    component.insert(start);
    let mut stack = VecDeque::new();
    stack.push_front(start);

    while let Some(node) = stack.pop_front() {
        for succ in graph.neighbors(node) {
            if discovered.visit(succ) {
                stack.push_back(succ);
                component.insert(succ);
            }
        }
    }

    component
}

/// Given an undirected graph, find a list of all the
/// connected components.
///
/// :param Graph graph: The input graph to find the connected
///     components for.
///
/// :return: A list of all the sets of connected components.
/// :rtype: Vec<HashSet<usize>>
pub fn connected_components<G>(
    graph: G,
) -> Vec<HashSet<G::NodeId>>
where
    G: GraphProp<EdgeType = Undirected>
        + EdgeCount
        + IntoEdges
        + Visitable
        + NodeIndexable
        + IntoNodeIdentifiers
        + IntoNodeReferences,
    G::NodeId: Eq + Hash,
{
    let mut conn_components = Vec::new();
    let mut discovered = graph.visit_map();

    for start in graph.node_identifiers() {
        if !discovered.visit(start) {
            continue;
        }

        let component = bfs_undirected(graph, start, &mut discovered);
        conn_components.push(component)
    }

    conn_components
}

/// Given an undirected graph, find a the number of
/// connected components.
///
/// :param Graph graph: The input graph to find the number of connected
///     components for.
///
/// :return: The number of connected components.
/// :rtype: usize
pub fn number_connected_components<G>(
    graph: G,
) -> usize
where
    G: GraphProp<EdgeType = Undirected>
        + EdgeCount
        + IntoEdges
        + Visitable
        + NodeIndexable
        + IntoNodeIdentifiers
        + IntoNodeReferences,
    G::NodeId: Eq + Hash,
{
    let mut num_components = 0;

    let mut discovered = graph.visit_map();
    for start in graph.node_identifiers() {
        if !discovered.visit(start) {
            continue;
        }

        num_components += 1;
        bfs_undirected(graph, start, &mut discovered);
    }

    num_components
}
