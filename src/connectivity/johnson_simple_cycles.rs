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

use crate::digraph::PyDiGraph;
use crate::StablePyGraph;
use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::visit::IntoEdgeReferences;
use petgraph::visit::IntoNodeReferences;
use petgraph::visit::NodeFiltered;
use petgraph::Directed;

use pyo3::Python;

fn build_subgraph(
    py: Python,
    graph: &StablePyGraph<Directed>,
    nodes: &[NodeIndex],
) -> (StablePyGraph<Directed>, HashMap<NodeIndex, NodeIndex>) {
    let node_set: HashSet<NodeIndex> = nodes.iter().copied().collect();
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(nodes.len());
    let node_filter = |node: NodeIndex| -> bool { node_set.contains(&node) };
    // Overallocates edges, but not a big deal as this is temporary for the lifetime of the
    // subgraph
    let mut out_graph = StablePyGraph::<Directed>::with_capacity(nodes.len(), graph.edge_count());
    let filtered = NodeFiltered(&graph, node_filter);
    for node in filtered.node_references() {
        let new_node = out_graph.add_node(node.1.clone_ref(py));
        node_map.insert(node.0, new_node);
    }
    for edge in filtered.edge_references() {
        let new_source = *node_map.get(&edge.source()).unwrap();
        let new_target = *node_map.get(&edge.target()).unwrap();
        out_graph.add_edge(new_source, new_target, edge.weight().clone_ref(py));
    }
    (out_graph, node_map)
}

#[inline]
fn set_pop(set: &mut HashSet<NodeIndex>) -> Option<NodeIndex> {
    if set.is_empty() {
        None
    } else {
        let set_node = set.iter().next().copied().unwrap();
        set.remove(&set_node);
        Some(set_node)
    }
}

pub fn simple_cycles(py: Python, graph: &PyDiGraph) -> Vec<Vec<usize>> {
    let mut out_cycles: Vec<Vec<usize>> = Vec::new();
    let unblock = |node: NodeIndex,
                   blocked: &mut HashSet<NodeIndex>,
                   block: &mut HashMap<NodeIndex, HashSet<NodeIndex>>| {
        let mut stack: HashSet<NodeIndex> = HashSet::new();
        stack.insert(node);
        while let Some(stack_node) = set_pop(&mut stack) {
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
    };
    // Copy graph to remove self edges before running johnson's algorithm
    let mut graph_clone = graph.graph.clone();

    // For compatibility with networkx manually insert self cycles and filter
    // from Johnson's algorithm
    let self_cycles: Vec<NodeIndex> = graph_clone
        .node_indices()
        .filter(|n| graph_clone.neighbors(*n).any(|x| x == *n))
        .collect();
    for node in self_cycles {
        out_cycles.push(vec![node.index()]);
        // Remove all self edges
        while let Some(edge_index) = graph_clone.find_edge(node, node) {
            graph_clone.remove_edge(edge_index);
        }
    }

    let mut strongly_connected_components: Vec<Vec<NodeIndex>> = kosaraju_scc(&graph_clone)
        .into_iter()
        .filter(|component| component.len() > 1)
        .collect();

    while let Some(mut scc) = strongly_connected_components.pop() {
        let (subgraph, node_map) = build_subgraph(py, &graph_clone, &scc);
        let reverse_node_map: HashMap<NodeIndex, NodeIndex> =
            node_map.iter().map(|(k, v)| (*v, *k)).collect();
        // start_node, path, blocked, closed, block and stack all in subgraph basis
        let start_node = node_map[&scc.pop().unwrap()];
        let mut path: Vec<NodeIndex> = vec![start_node];
        let mut blocked: HashSet<NodeIndex> = path.iter().copied().collect();
        // Nodes in cycle all
        let mut closed: HashSet<NodeIndex> = HashSet::new();
        let mut block: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();
        let mut stack: Vec<(NodeIndex, HashSet<NodeIndex>)> = vec![(
            start_node,
            subgraph
                .neighbors(start_node)
                .collect::<HashSet<NodeIndex>>(),
        )];
        while let Some((this_node, neighbors)) = stack.last_mut() {
            if let Some(next_node) = set_pop(neighbors) {
                if next_node == start_node {
                    // Out path in input graph basis
                    let mut out_path: Vec<usize> = Vec::with_capacity(path.len());
                    for n in &path {
                        out_path.push(reverse_node_map[n].index());
                        closed.insert(*n);
                    }
                    out_cycles.push(out_path);
                } else if blocked.insert(next_node) {
                    path.push(next_node);
                    stack.push((
                        next_node,
                        subgraph
                            .neighbors(next_node)
                            .collect::<HashSet<NodeIndex>>(),
                    ));
                    closed.remove(&next_node);
                    blocked.insert(next_node);
                    continue;
                }
            }
            if neighbors.is_empty() {
                if closed.contains(this_node) {
                    unblock(*this_node, &mut blocked, &mut block);
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
        let (h, node_map) = build_subgraph(
            py,
            &subgraph,
            &scc.iter().map(|n| node_map[n]).collect::<Vec<NodeIndex>>(),
        );
        let h_reverse_node_map: HashMap<NodeIndex, NodeIndex> =
            node_map.iter().map(|(k, v)| (*v, *k)).collect();
        strongly_connected_components.extend(kosaraju_scc(&h).into_iter().filter_map(|scc| {
            if scc.len() > 1 {
                let res = scc
                    .iter()
                    .map(|n| reverse_node_map[&h_reverse_node_map[n]])
                    .collect::<Vec<NodeIndex>>();
                Some(res)
            } else {
                None
            }
        }));
    }

    out_cycles
}
