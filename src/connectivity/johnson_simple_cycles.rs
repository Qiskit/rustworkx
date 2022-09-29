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

use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;

use crate::digraph::PyDiGraph;
use crate::StablePyGraph;
use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use petgraph::visit::IntoNodeReferences;
use petgraph::visit::NodeFiltered;
use petgraph::Directed;

use pyo3::iter::IterNextOutput;
use pyo3::prelude::*;

use crate::iterators::NodeIndices;

fn build_subgraph(
    graph: &StablePyGraph<Directed>,
    nodes: &[NodeIndex],
) -> (StableDiGraph<(), ()>, HashMap<NodeIndex, NodeIndex>) {
    let node_set: HashSet<NodeIndex> = nodes.iter().copied().collect();
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(nodes.len());
    let node_filter = |node: NodeIndex| -> bool { node_set.contains(&node) };
    // Overallocates edges, but not a big deal as this is temporary for the lifetime of the
    // subgraph
    let mut out_graph = StableDiGraph::<(), ()>::with_capacity(nodes.len(), graph.edge_count());
    let filtered = NodeFiltered(&graph, node_filter);
    for node in filtered.node_references() {
        let new_node = out_graph.add_node(());
        node_map.insert(node.0, new_node);
    }
    for edge in filtered.edge_references() {
        let new_source = *node_map.get(&edge.source()).unwrap();
        let new_target = *node_map.get(&edge.target()).unwrap();
        out_graph.add_edge(new_source, new_target, ());
    }
    (out_graph, node_map)
}

#[pyclass(module = "rustworkx")]
pub struct SimpleCycleIter {
    graph_clone: StablePyGraph<Directed>,
    scc: Vec<Vec<NodeIndex>>,
    self_cycles: Option<Vec<NodeIndex>>,
    path: Vec<NodeIndex>,
    blocked: HashSet<NodeIndex>,
    closed: HashSet<NodeIndex>,
    block: HashMap<NodeIndex, HashSet<NodeIndex>>,
    stack: Vec<(NodeIndex, IndexSet<NodeIndex, ahash::RandomState>)>,
    start_node: NodeIndex,
    node_map: HashMap<NodeIndex, NodeIndex>,
    reverse_node_map: HashMap<NodeIndex, NodeIndex>,
    subgraph: StableDiGraph<(), ()>,
}

impl SimpleCycleIter {
    pub fn new(graph: &PyDiGraph) -> Self {
        // Copy graph to remove self edges before running johnson's algorithm
        let mut graph_clone = graph.graph.clone();

        // For compatibility with networkx manually insert self cycles and filter
        // from Johnson's algorithm
        let self_cycles_vec: Vec<NodeIndex> = graph_clone
            .node_indices()
            .filter(|n| graph_clone.neighbors(*n).any(|x| x == *n))
            .collect();
        for node in &self_cycles_vec {
            while let Some(edge_index) = graph_clone.find_edge(*node, *node) {
                graph_clone.remove_edge(edge_index);
            }
        }
        let self_cycles = if self_cycles_vec.is_empty() {
            None
        } else {
            Some(self_cycles_vec)
        };
        let strongly_connected_components: Vec<Vec<NodeIndex>> = kosaraju_scc(&graph_clone)
            .into_iter()
            .filter(|component| component.len() > 1)
            .collect();
        SimpleCycleIter {
            graph_clone,
            scc: strongly_connected_components,
            self_cycles,
            path: Vec::new(),
            blocked: HashSet::new(),
            closed: HashSet::new(),
            block: HashMap::new(),
            stack: Vec::new(),
            start_node: NodeIndex::new(std::u32::MAX as usize),
            node_map: HashMap::new(),
            reverse_node_map: HashMap::new(),
            subgraph: StableDiGraph::new(),
        }
    }
}

fn unblock(
    node: NodeIndex,
    blocked: &mut HashSet<NodeIndex>,
    block: &mut HashMap<NodeIndex, HashSet<NodeIndex>>,
) {
    let mut stack: IndexSet<NodeIndex, ahash::RandomState> =
        IndexSet::with_hasher(ahash::RandomState::new());
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
    stack: &mut Vec<(NodeIndex, IndexSet<NodeIndex, ahash::RandomState>)>,
    path: &mut Vec<NodeIndex>,
    closed: &mut HashSet<NodeIndex>,
    blocked: &mut HashSet<NodeIndex>,
    block: &mut HashMap<NodeIndex, HashSet<NodeIndex>>,
    subgraph: &StableDiGraph<(), ()>,
    reverse_node_map: &HashMap<NodeIndex, NodeIndex>,
) -> Option<IterNextOutput<NodeIndices, &'static str>> {
    while let Some((this_node, neighbors)) = stack.last_mut() {
        if let Some(next_node) = neighbors.pop() {
            if next_node == start_node {
                // Out path in input graph basis
                let mut out_path: Vec<usize> = Vec::with_capacity(path.len());
                for n in path {
                    out_path.push(reverse_node_map[n].index());
                    closed.insert(*n);
                }
                return Some(IterNextOutput::Yield(NodeIndices { nodes: out_path }));
            } else if blocked.insert(next_node) {
                path.push(next_node);
                stack.push((
                    next_node,
                    subgraph
                        .neighbors(next_node)
                        .collect::<IndexSet<NodeIndex, ahash::RandomState>>(),
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

#[pymethods]
impl SimpleCycleIter {
    fn __iter__(slf: PyRef<Self>) -> Py<SimpleCycleIter> {
        slf.into()
    }

    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<IterNextOutput<NodeIndices, &'static str>> {
        if slf.self_cycles.is_some() {
            let self_cycles = slf.self_cycles.as_mut().unwrap();
            let cycle_node = self_cycles.pop().unwrap();
            if self_cycles.is_empty() {
                slf.self_cycles = None;
            }
            return Ok(IterNextOutput::Yield(NodeIndices {
                nodes: vec![cycle_node.index()],
            }));
        }
        // Restore previous state if it exists
        let mut stack: Vec<(NodeIndex, IndexSet<NodeIndex, ahash::RandomState>)> =
            std::mem::take(&mut slf.stack);
        let mut path: Vec<NodeIndex> = std::mem::take(&mut slf.path);
        let mut closed: HashSet<NodeIndex> = std::mem::take(&mut slf.closed);
        let mut blocked: HashSet<NodeIndex> = std::mem::take(&mut slf.blocked);
        let mut block: HashMap<NodeIndex, HashSet<NodeIndex>> = std::mem::take(&mut slf.block);
        let mut subgraph: StableDiGraph<(), ()> = std::mem::take(&mut slf.subgraph);
        let mut reverse_node_map: HashMap<NodeIndex, NodeIndex> =
            std::mem::take(&mut slf.reverse_node_map);
        let mut node_map: HashMap<NodeIndex, NodeIndex> = std::mem::take(&mut slf.node_map);

        if let Some(res) = process_stack(
            slf.start_node,
            &mut stack,
            &mut path,
            &mut closed,
            &mut blocked,
            &mut block,
            &subgraph,
            &reverse_node_map,
        ) {
            // Store internal state on yield
            slf.stack = stack;
            slf.path = path;
            slf.closed = closed;
            slf.blocked = blocked;
            slf.block = block;
            slf.subgraph = subgraph;
            slf.reverse_node_map = reverse_node_map;
            slf.node_map = node_map;
            return Ok(res);
        } else {
            subgraph.remove_node(slf.start_node);
            slf.scc
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
        while let Some(mut scc) = slf.scc.pop() {
            let temp = build_subgraph(&slf.graph_clone, &scc);
            subgraph = temp.0;
            node_map = temp.1;
            reverse_node_map = node_map.iter().map(|(k, v)| (*v, *k)).collect();
            // start_node, path, blocked, closed, block and stack all in subgraph basis
            slf.start_node = node_map[&scc.pop().unwrap()];
            path = vec![slf.start_node];
            blocked = path.iter().copied().collect();
            // Nodes in cycle all
            closed = HashSet::new();
            block = HashMap::new();
            stack = vec![(
                slf.start_node,
                subgraph
                    .neighbors(slf.start_node)
                    .collect::<IndexSet<NodeIndex, ahash::RandomState>>(),
            )];
            if let Some(res) = process_stack(
                slf.start_node,
                &mut stack,
                &mut path,
                &mut closed,
                &mut blocked,
                &mut block,
                &subgraph,
                &reverse_node_map,
            ) {
                // Store internal state on yield
                slf.stack = stack;
                slf.path = path;
                slf.closed = closed;
                slf.blocked = blocked;
                slf.block = block;
                slf.subgraph = subgraph;
                slf.reverse_node_map = reverse_node_map;
                slf.node_map = node_map;
                return Ok(res);
            }
            subgraph.remove_node(slf.start_node);
            slf.scc
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
        Ok(IterNextOutput::Return("Ended"))
    }
}
