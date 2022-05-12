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

use petgraph::visit::{GraphBase, VisitMap, Visitable};
use petgraph::EdgeType;

use crate::StablePyGraph;

pub fn bfs_undirected<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
    start: <StablePyGraph<Ty> as GraphBase>::NodeId,
    discovered: &mut <StablePyGraph<Ty> as Visitable>::Map,
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

pub fn connected_components<Ty>(graph: &StablePyGraph<Ty>) -> Vec<HashSet<usize>>
where
    Ty: EdgeType,
{
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

pub fn number_connected_components<Ty>(graph: &StablePyGraph<Ty>) -> usize
where
    Ty: EdgeType,
{
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
