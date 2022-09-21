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

use std::cmp::Eq;
use std::hash::Hash;

use hashbrown::HashMap;

use petgraph::visit::{
    GraphProp, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable, VisitMap, Visitable,
};
use petgraph::Undirected;

use crate::traversal::{depth_first_search, DfsEvent};

fn _build_chain<G, VM: VisitMap<G::NodeId>>(
    graph: G,
    parent: &[usize],
    mut u_id: G::NodeId,
    mut v_id: G::NodeId,
    visited: &mut VM,
) -> Vec<(G::NodeId, G::NodeId)>
where
    G: Visitable + NodeIndexable,
{
    let mut chain = Vec::new();
    while visited.visit(v_id) {
        chain.push((u_id, v_id));
        u_id = v_id;
        let u = graph.to_index(u_id);
        let v = parent[u];
        v_id = graph.from_index(v);
    }
    chain.push((u_id, v_id));

    chain
}

/// Returns the chain decomposition of a graph.
///
/// The *chain decomposition* of a graph with respect to a depth-first
/// search tree is a set of cycles or paths derived from the set of
/// fundamental cycles of the tree in the following manner. Consider
/// each fundamental cycle with respect to the given tree, represented
/// as a list of edges beginning with the nontree edge oriented away
/// from the root of the tree. For each fundamental cycle, if it
/// overlaps with any previous fundamental cycle, just take the initial
/// non-overlapping segment, which is a path instead of a cycle. Each
/// cycle or path is called a *chain*. For more information,
/// see [`Schmidt`](https://doi.org/10.1016/j.ipl.2013.01.016).
///
/// The graph should be undirected. If `source` is specified only the chain
/// decomposition for the connected component containing this node will be returned.
/// This node indicates the root of the depth-first search tree. If it's not
/// specified, a source will be chosen arbitrarly and repeated until all components
/// of the graph are searched.
///
/// Returns a list of list of edges where each inner list is a chain.
///
/// # Note
/// The function implicitly assumes that there are no parallel edges
/// or self loops. It may produce incorrect/unexpected results if the
/// input graph has self loops or parallel edges.
///
/// # Example
/// ```rust
/// use rustworkx_core::connectivity::chain_decomposition;
/// use rustworkx_core::petgraph::graph::{NodeIndex, UnGraph};
///
/// let mut graph : UnGraph<(), ()> = UnGraph::new_undirected();
/// let a = graph.add_node(()); // node with no weight
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// let d = graph.add_node(());
/// let e = graph.add_node(());
/// let f = graph.add_node(());
/// let g = graph.add_node(());
/// let h = graph.add_node(());
///
/// graph.extend_with_edges(&[
///     (a, b),
///     (b, c),
///     (c, d),
///     (d, a),
///     (e, f),
///     (b, e),
///     (f, g),
///     (g, h),
///     (h, e)
/// ]);
/// // a ---- b ---- e ---- f
/// // |      |      |      |
/// // d ---- c      h ---- g
///
/// let chains = chain_decomposition(&graph, None);
/// assert_eq!(
///     chains,
///     vec![
///         vec![(a, d), (d, c), (c, b), (b, a)],
///         vec![(e, h), (h, g), (g, f), (f, e)]
///     ]
/// );
/// ```
pub fn chain_decomposition<G>(
    graph: G,
    source: Option<G::NodeId>,
) -> Vec<Vec<(G::NodeId, G::NodeId)>>
where
    G: IntoNodeIdentifiers
        + IntoEdges
        + Visitable
        + NodeIndexable
        + NodeCount
        + GraphProp<EdgeType = Undirected>,
    G::NodeId: Eq + Hash,
{
    let roots = match source {
        Some(node) => vec![node],
        None => graph.node_identifiers().collect(),
    };

    let mut parent = vec![std::usize::MAX; graph.node_bound()];
    let mut back_edges: HashMap<G::NodeId, Vec<G::NodeId>> = HashMap::new();

    // depth-first-index (DFI) ordered nodes.
    let mut nodes = Vec::with_capacity(graph.node_count());
    depth_first_search(graph, roots, |event| match event {
        DfsEvent::Discover(u, _) => {
            nodes.push(u);
        }
        DfsEvent::TreeEdge(u, v, _) => {
            let u = graph.to_index(u);
            let v = graph.to_index(v);
            parent[v] = u;
        }
        DfsEvent::BackEdge(u_id, v_id, _) => {
            let u = graph.to_index(u_id);
            let v = graph.to_index(v_id);

            // do *not* consider ``(u, v)`` as a back edge if ``(v, u)`` is a tree edge.
            if parent[u] != v {
                back_edges
                    .entry(v_id)
                    .and_modify(|v_edges| v_edges.push(u_id))
                    .or_insert(vec![u_id]);
            }
        }
        _ => {}
    });

    let visited = &mut graph.visit_map();
    nodes
        .into_iter()
        .filter_map(|u| {
            visited.visit(u);
            back_edges.get(&u).map(|vs| {
                vs.iter()
                    .map(|v| _build_chain(graph, &parent, u, *v, visited))
                    .collect::<Vec<Vec<(G::NodeId, G::NodeId)>>>()
            })
        })
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::node_index as ni;
    use petgraph::prelude::*;

    #[test]
    fn test_decomposition() {
        let graph = UnGraph::<(), ()>::from_edges(&[
            //  DFS tree edges.
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (5, 9),
            (9, 10),
            //  Nontree edges.
            (1, 3),
            (1, 4),
            (2, 5),
            (5, 10),
            (6, 8),
        ]);

        let chains = chain_decomposition(&graph, Some(NodeIndex::new(1)));

        let expected: Vec<Vec<(NodeIndex<usize>, NodeIndex<usize>)>> = vec![
            vec![(ni(1), ni(3)), (ni(3), ni(2)), (ni(2), ni(1))],
            vec![(ni(1), ni(4)), (ni(4), ni(3))],
            vec![(ni(2), ni(5)), (ni(5), ni(3))],
            vec![(ni(5), ni(10)), (ni(10), ni(9)), (ni(9), ni(5))],
            vec![(ni(6), ni(8)), (ni(8), ni(7)), (ni(7), ni(6))],
        ];
        assert_eq!(chains.len(), expected.len());
    }
}
