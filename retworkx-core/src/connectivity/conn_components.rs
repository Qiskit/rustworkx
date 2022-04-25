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

use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::{Visitable, VisitMap};
use petgraph::Undirected;

pub fn bfs_undirected(
    graph: &mut Graph<(), (), Undirected>,
    start: NodeIndex,
    discovered: &mut <Graph<(), (), Undirected> as Visitable>::Map,
) -> HashSet<usize> {
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

pub fn connected_components(
    graph: &mut Graph<(), (), Undirected>
) -> Vec<HashSet<usize>> {
    let mut conn_components = Vec::new();
    let mut discovered = graph.visit_map();

    for start in graph.node_indices() {
        if !discovered.visit(start) {
            continue;
        }

        let component = bfs_undirected(graph, start, &mut discovered);
        conn_components.push(component)
    }

    conn_components
}

pub fn number_connected_components(
    graph: &mut Graph<(), (), Undirected>
) -> usize {
    let mut num_components = 0;

    let mut discovered = graph.visit_map();
    for start in graph.node_indices() {
        if !discovered.visit(start) {
            continue;
        }

        num_components += 1;
        bfs_undirected(graph, start, &mut discovered);
    }

    num_components
}

#[test]
fn test_connected_components() {
    let mut graph = Graph::<(), (), Undirected>::from_edges(&[
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)
    ]);
    let components = connected_components(&mut graph);
    let exp1: HashSet<usize> = [0, 1, 3, 2].iter().cloned().collect();
    let exp2: HashSet<usize> = [7, 5, 4, 6].iter().cloned().collect();
    let expected: Vec<_> = vec![exp1, exp2];
    assert_eq!(expected, components);
}