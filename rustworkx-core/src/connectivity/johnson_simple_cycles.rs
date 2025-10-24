// Licensed under the apache license, version 2.0 (the "license"); you may
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

use foldhash::fast::RandomState;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use std::hash::Hash;

use petgraph::Directed;
use petgraph::algo::kosaraju_scc;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::{
    EdgeCount, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, IntoNeighbors,
    IntoNeighborsDirected, IntoNodeReferences, NodeCount, NodeFiltered, NodeIndexable, NodeRef,
    Visitable,
};

use crate::graph_ext::EdgeFindable;

fn build_subgraph<G>(
    graph: G,
    nodes: &[NodeIndex],
) -> (StableDiGraph<(), ()>, HashMap<NodeIndex, NodeIndex>)
where
    G: EdgeCount + NodeCount + IntoEdgeReferences + IntoNodeReferences + GraphBase + NodeIndexable,
    <G as GraphBase>::NodeId: Hash + Eq,
{
    let node_set: HashSet<NodeIndex> = nodes.iter().copied().collect();
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(nodes.len());
    let node_filter =
        |node: G::NodeId| -> bool { node_set.contains(&NodeIndex::new(graph.to_index(node))) };
    // Overallocates edges, but not a big deal as this is temporary for the lifetime of the
    // subgraph
    let mut out_graph = StableDiGraph::<(), ()>::with_capacity(nodes.len(), graph.edge_count());
    let filtered = NodeFiltered::from_fn(graph, node_filter);
    for node in filtered.node_references() {
        let new_node = out_graph.add_node(());
        node_map.insert(NodeIndex::new(graph.to_index(node.id())), new_node);
    }
    for edge in filtered.edge_references() {
        let new_source = *node_map
            .get(&NodeIndex::new(graph.to_index(edge.source())))
            .unwrap();
        let new_target = *node_map
            .get(&NodeIndex::new(graph.to_index(edge.target())))
            .unwrap();
        out_graph.add_edge(
            NodeIndex::new(new_source.index()),
            NodeIndex::new(new_target.index()),
            (),
        );
    }
    (out_graph, node_map)
}

fn unblock(
    node: NodeIndex,
    blocked: &mut HashSet<NodeIndex>,
    block: &mut HashMap<NodeIndex, HashSet<NodeIndex>>,
) {
    let mut stack: IndexSet<NodeIndex, RandomState> = IndexSet::with_hasher(RandomState::default());
    stack.insert(node);
    while let Some(stack_node) = stack.pop() {
        if blocked.remove(&stack_node) {
            match block.get_mut(&stack_node) {
                // stack.update(block[stack_node]):
                Some(block_set) => {
                    block_set.drain().for_each(|n| {
                        stack.insert(n);
                    });
                }
                // If block doesn't have stack_node treat it as an empty set
                // (so no updates to stack) and populate it with an empty
                // set.
                None => {
                    block.insert(stack_node, HashSet::new());
                }
            }
            blocked.remove(&stack_node);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_stack(
    start_node: NodeIndex,
    stack: &mut Vec<(NodeIndex, IndexSet<NodeIndex, foldhash::fast::RandomState>)>,
    path: &mut Vec<NodeIndex>,
    closed: &mut HashSet<NodeIndex>,
    blocked: &mut HashSet<NodeIndex>,
    block: &mut HashMap<NodeIndex, HashSet<NodeIndex>>,
    subgraph: &StableDiGraph<(), ()>,
    reverse_node_map: &HashMap<NodeIndex, NodeIndex>,
) -> Option<Vec<NodeIndex>> {
    while let Some((this_node, neighbors)) = stack.last_mut() {
        if let Some(next_node) = neighbors.pop() {
            if next_node == start_node {
                // Out path in input graph basis
                let mut out_path: Vec<NodeIndex> = Vec::with_capacity(path.len());
                for n in path {
                    out_path.push(reverse_node_map[n]);
                    closed.insert(*n);
                }
                return Some(out_path);
            } else if blocked.insert(next_node) {
                path.push(next_node);
                stack.push((
                    next_node,
                    subgraph
                        .neighbors(next_node)
                        .collect::<IndexSet<NodeIndex, foldhash::fast::RandomState>>(),
                ));
                closed.remove(&next_node);
                blocked.insert(next_node);
                continue;
            }
        }
        if neighbors.is_empty() {
            if closed.contains(this_node) {
                unblock(*this_node, blocked, block);
            } else {
                for neighbor in subgraph.neighbors(*this_node) {
                    let block_neighbor = block.entry(neighbor).or_insert_with(HashSet::new);
                    block_neighbor.insert(*this_node);
                }
            }
            stack.pop();
            path.pop();
        }
    }
    None
}

/// An iterator of simple cycles in a graph
///
/// `SimpleCycleIter` does not itself borrow the graph, and because of this
/// you can run the algorithm while retaining mutable access to it, if you
/// use like it the following example:
///
/// ```
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::connectivity::johnson_simple_cycles;
///
/// let mut graph = DiGraph::<_,()>::new();
/// let a = graph.add_node(0);
///
/// let mut cycle_iter = johnson_simple_cycles(&graph, None);
/// while let Some(cycle) = cycle_iter.next(&graph) {
///     // We can access `graph` mutably here still
///     graph[a] += 1;
/// }
///
/// assert_eq!(graph[a], 0);
/// ```
pub struct SimpleCycleIter {
    scc: Vec<Vec<NodeIndex>>,
    self_cycles: Option<Vec<NodeIndex>>,
    path: Vec<NodeIndex>,
    blocked: HashSet<NodeIndex>,
    closed: HashSet<NodeIndex>,
    block: HashMap<NodeIndex, HashSet<NodeIndex>>,
    stack: Vec<(NodeIndex, IndexSet<NodeIndex, RandomState>)>,
    start_node: NodeIndex,
    node_map: HashMap<NodeIndex, NodeIndex>,
    reverse_node_map: HashMap<NodeIndex, NodeIndex>,
    subgraph: StableDiGraph<(), ()>,
}

impl SimpleCycleIter {
    /// Return the next cycle found, if `None` is returned the algorithm is complete and all
    /// cycles have been found.
    pub fn next<G>(&mut self, graph: G) -> Option<Vec<NodeIndex>>
    where
        G: IntoEdgeReferences
            + IntoNodeReferences
            + GraphBase
            + EdgeCount
            + NodeCount
            + NodeIndexable,
        <G as GraphBase>::NodeId: Hash + Eq,
    {
        if self.self_cycles.is_some() {
            let self_cycles = self.self_cycles.as_mut().unwrap();
            let cycle_node = self_cycles.pop().unwrap();
            if self_cycles.is_empty() {
                self.self_cycles = None;
            }
            return Some(vec![cycle_node]);
        }
        // Restore previous state if it exists
        let mut stack: Vec<(NodeIndex, IndexSet<NodeIndex, foldhash::fast::RandomState>)> =
            std::mem::take(&mut self.stack);
        let mut path: Vec<NodeIndex> = std::mem::take(&mut self.path);
        let mut closed: HashSet<NodeIndex> = std::mem::take(&mut self.closed);
        let mut blocked: HashSet<NodeIndex> = std::mem::take(&mut self.blocked);
        let mut block: HashMap<NodeIndex, HashSet<NodeIndex>> = std::mem::take(&mut self.block);
        let mut subgraph: StableDiGraph<(), ()> = std::mem::take(&mut self.subgraph);
        let mut reverse_node_map: HashMap<NodeIndex, NodeIndex> =
            std::mem::take(&mut self.reverse_node_map);
        let mut node_map: HashMap<NodeIndex, NodeIndex> = std::mem::take(&mut self.node_map);
        if let Some(res) = process_stack(
            self.start_node,
            &mut stack,
            &mut path,
            &mut closed,
            &mut blocked,
            &mut block,
            &subgraph,
            &reverse_node_map,
        ) {
            // Store internal state on yield
            self.stack = stack;
            self.path = path;
            self.closed = closed;
            self.blocked = blocked;
            self.block = block;
            self.subgraph = subgraph;
            self.reverse_node_map = reverse_node_map;
            self.node_map = node_map;
            return Some(res);
        } else {
            subgraph.remove_node(self.start_node);
            self.scc
                .extend(kosaraju_scc(&subgraph).into_iter().filter_map(|scc| {
                    if scc.len() > 1 {
                        let res = scc
                            .iter()
                            .map(|n| reverse_node_map[n])
                            .collect::<Vec<NodeIndex>>();
                        Some(res)
                    } else {
                        None
                    }
                }));
        }
        while let Some(mut scc) = self.scc.pop() {
            let temp = build_subgraph(graph, &scc);
            subgraph = temp.0;
            node_map = temp.1;
            reverse_node_map = node_map.iter().map(|(k, v)| (*v, *k)).collect();
            // start_node, path, blocked, closed, block and stack all in subgraph basis
            self.start_node = node_map[&scc.pop().unwrap()];
            path = vec![self.start_node];
            blocked = path.iter().copied().collect();
            // Nodes in cycle all
            closed = HashSet::new();
            block = HashMap::new();
            stack = vec![(
                self.start_node,
                subgraph
                    .neighbors(self.start_node)
                    .collect::<IndexSet<NodeIndex, foldhash::fast::RandomState>>(),
            )];
            if let Some(res) = process_stack(
                self.start_node,
                &mut stack,
                &mut path,
                &mut closed,
                &mut blocked,
                &mut block,
                &subgraph,
                &reverse_node_map,
            ) {
                // Store internal state on yield
                self.stack = stack;
                self.path = path;
                self.closed = closed;
                self.blocked = blocked;
                self.block = block;
                self.subgraph = subgraph;
                self.reverse_node_map = reverse_node_map;
                self.node_map = node_map;
                return Some(res);
            }
            subgraph.remove_node(self.start_node);
            self.scc
                .extend(kosaraju_scc(&subgraph).into_iter().filter_map(|scc| {
                    if scc.len() > 1 {
                        let res = scc
                            .iter()
                            .map(|n| reverse_node_map[n])
                            .collect::<Vec<NodeIndex>>();
                        Some(res)
                    } else {
                        None
                    }
                }));
        }
        None
    }
}

/// /// Find all simple cycles of a graph
///
/// A "simple cycle" (called an elementary circuit in [^Johnson75]) is a cycle (or closed path)
/// where no node appears more than once.
///
/// This function is a an implementation of Johnson's algorithm [^Johnson75] also based
/// on the non-recursive implementation found in NetworkX[^NetworkDevs24] with code available on Github[^GitHub24].
///
/// To handle self cycles in a manner consistent with the NetworkX implementation you should
/// use the ``self_cycles`` argument to collect manually collected self cycle and then remove
/// the edges leading to a self cycle from the graph. If you don't do this
/// the self cycle may or may not be returned by the iterator. The underlying algorithm is not
/// able to consistently detect self cycles so it is best to handle them before calling this
/// function. The example below shows a pattern for doing this. You will need to clone the graph
/// to do this detection without modifying the graph.
///
/// # Returns
///
/// This function returns a `SimpleCycleIter` iterator which returns a `Vec` of `NodeIndex`.
/// Note the `NodeIndex` type is not necessarily the same as the input graph, as it's built
/// using an internal `StableGraph` used by the algorithm. If your input `graph` uses a
/// different node index type that differs from the default `NodeIndex<u32>`/`NodeIndex<DefaultIx>`
/// you will want to convert these objects to your native `NodeIndex` type.
///
/// The return from this function is not guaranteed to have a particular order for either the
/// cycles or the indices in each cycle.
///
/// [^Johnson75]: <https://doi.org/10.1137/0204007>
/// [^NetworkDevs24]: <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.simple_cycles.html>
/// [^GitHub24]: <https://github.com/networkx/networkx/blob/networkx-2.8.4/networkx/algorithms/cycles.py#L98-L222>
///
/// # Example:
///
/// ```rust
/// use rustworkx_core::petgraph::prelude::*;
/// use rustworkx_core::connectivity::johnson_simple_cycles;
///
/// let mut graph = DiGraph::<(), ()>::new();
/// graph.extend_with_edges([(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]);
///
/// // Handle self cycles
/// let self_cycles_vec: Vec<NodeIndex> = graph
///     .node_indices()
///     .filter(|n| graph.neighbors(*n).any(|x| x == *n))
///     .collect();
/// for node in &self_cycles_vec {
///     while let Some(edge_index) = graph.find_edge(*node, *node) {
///         graph.remove_edge(edge_index);
///     }
/// }
/// let self_cycles = if self_cycles_vec.is_empty() {
///            None
/// } else {
///     Some(self_cycles_vec)
/// };
///
/// let mut cycles_iter = johnson_simple_cycles(&graph, self_cycles);
///
/// let mut cycles = Vec::new();
/// while let Some(mut cycle) = cycles_iter.next(&graph) {
///     cycle.sort();
///     cycles.push(cycle);
/// }
///
/// let expected = vec![
///     vec![NodeIndex::new(0)],
///     vec![NodeIndex::new(2)],
///     vec![NodeIndex::new(0), NodeIndex::new(1), NodeIndex::new(2)],
///     vec![NodeIndex::new(0), NodeIndex::new(2)],
///     vec![NodeIndex::new(1), NodeIndex::new(2)],
/// ];
///
/// assert_eq!(expected.len(), cycles.len());
/// for cycle in cycles {
///     assert!(expected.contains(&cycle));
/// }
/// ```
pub fn johnson_simple_cycles<G>(
    graph: G,
    self_cycles: Option<Vec<<G as GraphBase>::NodeId>>,
) -> SimpleCycleIter
where
    G: IntoEdgeReferences
        + IntoNodeReferences
        + GraphBase
        + EdgeCount
        + NodeCount
        + NodeIndexable
        + Clone
        + IntoNeighbors
        + IntoNeighborsDirected
        + Visitable
        + EdgeFindable
        + GraphProp<EdgeType = Directed>,
    <G as GraphBase>::NodeId: Hash + Eq,
{
    let self_cycles = self_cycles.map(|self_cycles_vec| {
        self_cycles_vec
            .into_iter()
            .map(|n| NodeIndex::new(graph.to_index(n)))
            .collect()
    });
    let strongly_connected_components: Vec<Vec<NodeIndex>> = kosaraju_scc(graph)
        .into_iter()
        .filter_map(|component| {
            if component.len() > 1 {
                Some(
                    component
                        .into_iter()
                        .map(|n| NodeIndex::new(graph.to_index(n)))
                        .collect(),
                )
            } else {
                None
            }
        })
        .collect();
    SimpleCycleIter {
        scc: strongly_connected_components,
        self_cycles,
        path: Vec::new(),
        blocked: HashSet::new(),
        closed: HashSet::new(),
        block: HashMap::new(),
        stack: Vec::new(),
        start_node: NodeIndex::new(u32::MAX as usize),
        node_map: HashMap::new(),
        reverse_node_map: HashMap::new(),
        subgraph: StableDiGraph::new(),
    }
}

#[cfg(test)]
mod test_longest_path {
    use super::*;
    use petgraph::graph::DiGraph;
    use petgraph::stable_graph::NodeIndex;
    use petgraph::stable_graph::StableDiGraph;

    #[test]
    fn test_empty_graph() {
        let graph: DiGraph<(), ()> = DiGraph::new();
        let mut result: Vec<_> = Vec::new();
        let mut cycle_iter = johnson_simple_cycles(&graph, None);
        while let Some(cycle) = cycle_iter.next(&graph) {
            result.push(cycle);
        }
        let expected: Vec<Vec<NodeIndex>> = Vec::new();
        assert_eq!(expected, result)
    }

    #[test]
    fn test_empty_stable_graph() {
        let graph: StableDiGraph<(), ()> = StableDiGraph::new();
        let mut result: Vec<_> = Vec::new();
        let mut cycle_iter = johnson_simple_cycles(&graph, None);
        while let Some(cycle) = cycle_iter.next(&graph) {
            result.push(cycle);
        }
        let expected: Vec<Vec<NodeIndex>> = Vec::new();
        assert_eq!(expected, result)
    }

    #[test]
    fn test_figure_1() {
        for k in 3..10 {
            let mut graph: DiGraph<(), ()> = DiGraph::new();
            let mut edge_list: Vec<[usize; 2]> = Vec::new();
            for n in 2..k + 2 {
                edge_list.push([1, n]);
                edge_list.push([n, k + 2]);
            }
            edge_list.push([2 * k + 1, 1]);
            for n in k + 2..2 * k + 2 {
                edge_list.push([n, 2 * k + 2]);
                edge_list.push([n, n + 1]);
            }
            edge_list.push([2 * k + 3, k + 2]);
            for n in 2 * k + 3..3 * k + 3 {
                edge_list.push([2 * k + 2, n]);
                edge_list.push([n, 3 * k + 3]);
            }
            edge_list.push([3 * k + 3, 2 * k + 2]);
            graph.extend_with_edges(
                edge_list
                    .into_iter()
                    .map(|x| (NodeIndex::new(x[0]), NodeIndex::new(x[1]))),
            );
            let mut cycles_iter = johnson_simple_cycles(&graph, None);
            let mut res = 0;
            while cycles_iter.next(&graph).is_some() {
                res += 1;
            }
            assert_eq!(res, 3 * k);
        }
    }

    #[test]
    fn test_figure_1_stable_graph() {
        for k in 3..10 {
            let mut graph: StableDiGraph<(), ()> = StableDiGraph::new();
            let mut edge_list: Vec<[usize; 2]> = Vec::new();
            for n in 2..k + 2 {
                edge_list.push([1, n]);
                edge_list.push([n, k + 2]);
            }
            edge_list.push([2 * k + 1, 1]);
            for n in k + 2..2 * k + 2 {
                edge_list.push([n, 2 * k + 2]);
                edge_list.push([n, n + 1]);
            }
            edge_list.push([2 * k + 3, k + 2]);
            for n in 2 * k + 3..3 * k + 3 {
                edge_list.push([2 * k + 2, n]);
                edge_list.push([n, 3 * k + 3]);
            }
            edge_list.push([3 * k + 3, 2 * k + 2]);
            graph.extend_with_edges(
                edge_list
                    .into_iter()
                    .map(|x| (NodeIndex::new(x[0]), NodeIndex::new(x[1]))),
            );
            let mut cycles_iter = johnson_simple_cycles(&graph, None);
            let mut res = 0;
            while cycles_iter.next(&graph).is_some() {
                res += 1;
            }
            assert_eq!(res, 3 * k);
        }
    }
}
