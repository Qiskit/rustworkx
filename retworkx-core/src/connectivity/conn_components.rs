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

//use petgraph::visit::{GraphBase, VisitMap, Visitable};
use petgraph::EdgeType;
use petgraph::graph::Graph;
use petgraph::{
    visit::{
        EdgeCount, GraphProp, IntoEdges, IntoNodeIdentifiers, NodeIndexable,
        Visitable, VisitMap,
    },
    Undirected,
};


pub fn bfs_undirected(
    graph: &Graph<(), (), Undirected>,
    start: &Graph::NodeId,
    discovered: &mut Graph::Map,
) -> HashSet<usize>
// where
//     G: GraphProp<EdgeType = Undirected>
//         + EdgeCount
//         + IntoEdges
//         + Visitable
//         + NodeIndexable
//         + IntoNodeIdentifiers,
//     G::NodeId: Eq + Hash,
{
    let mut component = HashSet::new();
    component.insert(start.index());
    let mut stack = VecDeque::new();
    stack.push_front(start);

    while let Some(node) = stack.pop_front() {
        for succ in graph.neighbors_undirected(node) {
            if discovered.visit(succ) {
                stack.push_back(succ);
                component.insert(succ.index());
            }
        }
    }

    component
}

pub fn connected_components<G>(graph: G)-> Vec<HashSet<usize>>
where
    G: GraphProp<EdgeType = Undirected>
        + EdgeCount
        + IntoEdges
        + Visitable
        + NodeIndexable
        + IntoNodeIdentifiers
        + VisitMap<G>,
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

pub fn number_connected_components<G>(graph: G)-> usize
where
    G: GraphProp<EdgeType = Undirected>
        + EdgeCount
        + IntoEdges
        + Visitable
        + NodeIndexable
        + IntoNodeIdentifiers,
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
