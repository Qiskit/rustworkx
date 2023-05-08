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

use std::hash::Hash;

use hashbrown::{HashMap, HashSet};

use petgraph::visit::{
    EdgeCount, IntoNeighbors, IntoNodeIdentifiers, NodeCount, NodeIndexable, Visitable,
};

/// Return an edge list of the tree edges from a depth-first traversal.
///
/// The pseudo-code for the DFS algorithm is listed below. The output
/// contains the tree edges found by the procedure.
///
/// ```norust
/// DFS(G, v)
///   let S be a stack
///   label v as discovered
///   PUSH(S, (v, iterator of G.neighbors(v)))
///   while (S != Ø)
///       let (v, iterator) := LAST(S)
///       if hasNext(iterator) then
///           w := next(iterator)
///           if w is not labeled as discovered then
///               label w as discovered                   # (v, w) is a tree edge
///               PUSH(S, (w, iterator of G.neighbors(w)))
///       else
///           POP(S)
///   end while
/// ```
///
/// Arguments:
///
/// * `graph` - the graph to run on
/// * `source` - the optional node index to use as the starting node for the
///     depth-first search. If specified the edge list will only return edges
///     in the components reachable from this index. If this is not specified
///     then a source will be chosen arbitrarily and repeated until all
///     components of the graph are searched
///
/// # Example
/// ```rust
/// use rustworkx_core::petgraph;
/// use rustworkx_core::traversal::dfs_edges;
///
/// let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[
///     (0, 1), (1, 2), (1, 3), (2, 4), (3, 4)
/// ]);
/// let dfs_edges = dfs_edges(&g, Some(petgraph::graph::NodeIndex::new(0)));
/// assert_eq!(vec![(0, 1), (1, 2), (2, 4), (4, 3)], dfs_edges);
/// ```
pub fn dfs_edges<G>(graph: G, source: Option<G::NodeId>) -> Vec<(usize, usize)>
where
    G: IntoNodeIdentifiers + NodeIndexable + IntoNeighbors + NodeCount + EdgeCount + Visitable,
    G::NodeId: Eq + Hash,
{
    let nodes: Vec<G::NodeId> = match source {
        Some(start) => vec![start],
        None => graph.node_identifiers().collect(),
    };
    let node_count = graph.node_count();
    let mut visited: HashSet<G::NodeId> = HashSet::with_capacity(node_count);
    // Avoid potential overallocation if source node is provided
    let mut out_vec: Vec<(usize, usize)> = if source.is_some() {
        Vec::new()
    } else {
        Vec::with_capacity(core::cmp::min(graph.node_count() - 1, graph.edge_count()))
    };
    for start in nodes {
        if visited.contains(&start) {
            continue;
        }
        visited.insert(start);
        let mut children: Vec<G::NodeId> = graph.neighbors(start).collect();
        children.reverse();
        let mut stack: Vec<(G::NodeId, Vec<G::NodeId>)> = vec![(start, children)];
        // Used to track the last position in children vec across iterations
        let mut index_map: HashMap<G::NodeId, usize> = HashMap::with_capacity(node_count);
        index_map.insert(start, 0);
        while !stack.is_empty() {
            let temp_parent = stack.last().unwrap();
            let parent = temp_parent.0;
            let children = temp_parent.1.clone();
            let count = *index_map.get(&parent).unwrap();
            let mut found = false;
            let mut index = count;
            for child in &children[index..] {
                index += 1;
                if !visited.contains(child) {
                    out_vec.push((graph.to_index(parent), graph.to_index(*child)));
                    visited.insert(*child);
                    let mut grandchildren: Vec<G::NodeId> = graph.neighbors(*child).collect();
                    grandchildren.reverse();
                    stack.push((*child, grandchildren));
                    index_map.insert(*child, 0);
                    *index_map.get_mut(&parent).unwrap() = index;
                    found = true;
                    break;
                }
            }
            if !found || children.is_empty() {
                stack.pop();
            }
        }
    }
    out_vec
}
