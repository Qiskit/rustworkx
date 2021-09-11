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

use pyo3::prelude::*;

use petgraph::prelude::*;
use petgraph::visit::{depth_first_search, DfsEvent, NodeIndexable, Time};

const NULL: usize = std::usize::MAX;

#[inline]
/// Finds the position of ``elem`` in vector ``xs``.
fn search<T>(xs: &[T], elem: T) -> Option<usize>
where
    T: std::cmp::Eq,
{
    xs.iter()
        .enumerate()
        .find(|&(_, x)| *x == elem)
        .map(|(idx, _)| idx)
}

#[inline]
/// Returns the unique elements of type ``T`` contained
/// in a vector that holds pairs of ``T`` values.
fn flattened<T>(xs: &[[T; 2]]) -> HashSet<T>
where
    T: std::hash::Hash + std::cmp::Eq + Clone,
{
    xs.into_iter().flatten().cloned().collect()
}

#[inline]
fn is_root(parent: &[usize], u: usize) -> bool {
    parent[u] == NULL
}

pub fn articulation_points(
    graph: &StableUnGraph<PyObject, PyObject>,
    components: Option<&mut Vec<HashSet<usize>>>,
) -> HashSet<usize> {
    let num_nodes = graph.node_bound();

    let mut low = vec![NULL; num_nodes];
    let mut disc = vec![NULL; num_nodes];
    let mut parent = vec![NULL; num_nodes];

    let mut root_children: usize = 0;
    let mut points = HashSet::new();

    let mut edge_stack = Vec::new();
    let mut tmp_components = Vec::new();

    depth_first_search(graph, graph.node_indices(), |event| match event {
        DfsEvent::Discover(u, Time(t)) => {
            let u = u.index();
            low[u] = t;
            disc[u] = t;
        }
        DfsEvent::TreeEdge(u, v) => {
            let u = u.index();
            let v = v.index();
            parent[v] = u;
            if is_root(&parent, u) {
                root_children += 1;
            }
            if components.is_some() {
                edge_stack.push([u, v]);
            }
        }
        DfsEvent::BackEdge(u, v) => {
            let u = u.index();
            let v = v.index();

            // do *not* consider ``(u, v)`` as a back edge if ``(v, u)`` is a tree edge.
            if v != parent[u] {
                low[u] = low[u].min(disc[v]);
                if components.is_some() {
                    edge_stack.push([u, v]);
                }
            }
        }
        DfsEvent::Finish(u, _) => {
            let u = u.index();
            if is_root(&parent, u) {
                if root_children > 1 {
                    points.insert(u);
                }
                // restart ``root_children`` for the remaining connected components
                root_children = 0;
            } else {
                let pu = parent[u];
                low[pu] = low[pu].min(low[u]);

                if !is_root(&parent, pu) && low[u] >= disc[pu] {
                    points.insert(pu);
                    // now find a biconnected component that the
                    // current articulation point belongs.
                    if components.is_some() {
                        if let Some(at) = search(&edge_stack, [pu, u]) {
                            tmp_components.push(flattened(&edge_stack[at..]));
                            edge_stack.truncate(at);
                        }
                    }
                }

                if is_root(&parent, pu) && components.is_some() {
                    if let Some(at) = search(&edge_stack, [pu, u]) {
                        tmp_components.push(flattened(&edge_stack[at..]));
                    }
                }
            }
        }
        _ => {}
    });

    if let Some(x) = components {
        x.append(&mut tmp_components);
    }

    points
}
