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
use std::hash::Hash;

use petgraph::{
    visit::{
        depth_first_search, DfsEvent, GraphProp, IntoNeighbors,
        IntoNodeIdentifiers, NodeIndexable, Time, Visitable,
    },
    Undirected,
};

const NULL: usize = std::usize::MAX;

/// Returns the unique elements of type ``T`` contained
/// in a vector that holds pairs of ``T`` values.
#[inline]
fn flattened<T>(xs: &[[T; 2]]) -> HashSet<T>
where
    T: Eq + Hash + Copy,
{
    xs.iter().flatten().copied().collect()
}

#[inline]
fn is_root(parent: &[usize], u: usize) -> bool {
    parent[u] == NULL
}

/// Return the articulation points of an undirected graph.
///
/// An articulation point or cut vertex is any node whose removal (along with
/// all its incident edges) increases the number of connected components of
/// a graph. An undirected connected graph without articulation points is
/// biconnected.
///
/// At the same time, you can record the biconnected components in `components`.
///
/// Biconnected components are maximal subgraphs such that the removal
/// of a node (and all edges incident on that node) will not disconnect
/// the subgraph. Note that nodes may be part of more than one biconnected
/// component. Those nodes are articulation points, or cut vertices.
///
/// # Note
/// The function implicitly assumes that there are no parallel edges
/// or self loops. It may produce incorrect/unexpected results if the
/// input graph has self loops or parallel edges.
pub fn articulation_points<G>(
    graph: G,
    components: Option<&mut Vec<HashSet<G::NodeId>>>,
) -> HashSet<G::NodeId>
where
    G: GraphProp<EdgeType = Undirected>
        + IntoNeighbors
        + Visitable
        + NodeIndexable
        + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
{
    let num_nodes = graph.node_bound();

    let mut low = vec![NULL; num_nodes];
    let mut disc = vec![NULL; num_nodes];
    let mut parent = vec![NULL; num_nodes];

    let mut root_children: usize = 0;
    let mut points = HashSet::new();

    let mut edge_stack = Vec::new();
    let mut tmp_components = Vec::new();

    depth_first_search(graph, graph.node_identifiers(), |event| match event {
        DfsEvent::Discover(u_id, Time(t)) => {
            let u = graph.to_index(u_id);
            low[u] = t;
            disc[u] = t;
        }
        DfsEvent::TreeEdge(u_id, v_id) => {
            let u = graph.to_index(u_id);
            let v = graph.to_index(v_id);
            parent[v] = u;
            if is_root(&parent, u) {
                root_children += 1;
            }
            if components.is_some() {
                edge_stack.push([u_id, v_id]);
            }
        }
        DfsEvent::BackEdge(u_id, v_id) => {
            let u = graph.to_index(u_id);
            let v = graph.to_index(v_id);

            // do *not* consider ``(u, v)`` as a back edge if ``(v, u)`` is a tree edge.
            if v != parent[u] {
                low[u] = low[u].min(disc[v]);
                if components.is_some() {
                    edge_stack.push([u_id, v_id]);
                }
            }
        }
        DfsEvent::Finish(u_id, _) => {
            let u = graph.to_index(u_id);
            if is_root(&parent, u) {
                if root_children > 1 {
                    points.insert(u_id);
                }
                // restart ``root_children`` for the remaining connected components
                root_children = 0;
            } else {
                let pu = parent[u];
                let pu_id = graph.from_index(pu);
                low[pu] = low[pu].min(low[u]);

                if !is_root(&parent, pu) && low[u] >= disc[pu] {
                    points.insert(pu_id);
                    // now find a biconnected component that the
                    // current articulation point belongs.
                    if components.is_some() {
                        if let Some(at) =
                            edge_stack.iter().rposition(|&x| x == [pu_id, u_id])
                        {
                            tmp_components.push(flattened(&edge_stack[at..]));
                            edge_stack.truncate(at);
                        }
                    }
                }

                if is_root(&parent, pu) && components.is_some() {
                    if let Some(at) =
                        edge_stack.iter().position(|&x| x == [pu_id, u_id])
                    {
                        tmp_components.push(flattened(&edge_stack[at..]));
                        edge_stack.truncate(at);
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
