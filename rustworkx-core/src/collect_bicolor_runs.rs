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
use std::error::Error;
use std::hash::Hash;
use std::fmt::{Debug, Display, Formatter};

use petgraph::algo;
use petgraph::visit::Data;
use petgraph::data::DataMap;
use petgraph::visit::{EdgeRef, GraphBase, GraphProp, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable, Visitable, IntoEdgesDirected};


#[derive(Debug)]
pub enum CollectBicolorError {
    DAGWouldCycle,
}

impl Display for CollectBicolorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CollectBicolorError::DAGWouldCycle => fmt_dag_would_cycle(f),
        }
    }
}

impl Error for CollectBicolorError {}

#[derive(Debug)]
pub enum CollectBicolorSimpleError<E: Error> {
    DAGWouldCycle,
    MergeError(E), //placeholder, may remove if not used
}

impl<E: Error> Display for CollectBicolorSimpleError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CollectBicolorSimpleError::DAGWouldCycle => fmt_dag_would_cycle(f),
            CollectBicolorSimpleError::MergeError(ref e) => fmt_merge_error(f, e),
        }
    }
}

impl<E: Error> Error for CollectBicolorSimpleError<E> {}

fn fmt_dag_would_cycle(f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(f, "The operation would introduce a cycle.")
}

fn fmt_merge_error<E: Error>(f: &mut Formatter<'_>, inner: &E) -> std::fmt::Result {
    write!(f, "The prov failed with: {:?}", inner)
}
/// Collect runs that match a filter function given edge colors
///
/// A bicolor run is a list of group of nodes connected by edges of exactly
/// two colors. In addition, all nodes in the group must match the given
/// condition. Each node in the graph can appear in only a single group
/// in the bicolor run.
///
/// :param PyDiGraph graph: The graph to find runs in
/// :param filter_fn: The filter function to use for matching nodes. It takes
///     in one argument, the node data payload/weight object, and will return a
///     boolean whether the node matches the conditions or not.
///     If it returns ``True``, it will continue the bicolor chain.
///     If it returns ``False``, it will stop the bicolor chain.
///     If it returns ``None`` it will skip that node.
/// :param color_fn: The function that gives the color of the edge. It takes
///     in one argument, the edge data payload/weight object, and will
///     return a non-negative integer, the edge color. If the color is None,
///     the edge is ignored.
///
/// :returns: a list of groups with exactly two edge colors, where each group
///     is a list of node data payload/weight for the nodes in the bicolor run
/// :rtype: list
pub fn collect_bicolor_runs<G, F, C, B, E>(
    graph: G,
    filter_fn: F,
    color_fn: C,
) -> Result<Vec<Vec<usize>>, CollectBicolorSimpleError<E>> //OG type: PyResult<Vec<Vec<PyObject>>>
where
    E: Error,
    // add option because of line 135
    F: FnMut(&Option<&<G as Data>::NodeWeight>) -> Result<Option<bool>, CollectBicolorSimpleError<E>>, //OG input: &PyObject, OG return: PyResult<Option<bool>>
    C: FnMut(&<G as Data>::EdgeWeight) -> Result<Option<usize>, CollectBicolorSimpleError<E>>, //OG input: &PyObject, OG return: PyResult<Option<usize>>
    G: NodeIndexable //can take node index type and convert to usize. It restricts node index type.
        + IntoNodeIdentifiers //turn graph into list of nodes
        + IntoNeighborsDirected // toposort
        + IntoEdgesDirected
        + Visitable // toposort
        + GraphProp // gives access to whether graph is directed
        + NodeCount
        + DataMap,
    <G as GraphBase>::NodeId: Eq + Hash,
{
    let mut pending_list = Vec::new(); //OG type: Vec<Vec<PyObject>>
    let mut block_id = Vec::new(); //OG type: Vec<Option<usize>>
    let mut block_list = Vec::new(); //OG type: Vec<Vec<PyObject>> -> return

    let filter_node = |node: &Option<&<G as Data>::NodeWeight>| -> Result<Option<bool>, CollectBicolorSimpleError<E>>{
        let res = filter_fn(node);
        res
    };

    let color_edge = |edge: &<G as Data>::EdgeWeight| -> Result<Option<usize>, CollectBicolorSimpleError<E>>{
        let res = color_fn(edge);
        res
    };

    let nodes = match algo::toposort(&graph, None){
        Ok(nodes) => nodes,
        Err(_err) => return Err(CollectBicolorSimpleError::DAGWouldCycle)
    };

    // Utility for ensuring pending_list has the color index
    macro_rules! ensure_vector_has_index {
        ($pending_list: expr, $block_id: expr, $color: expr) => {
            if $color >= $pending_list.len() {
                $pending_list.resize($color + 1, Vec::new());
                $block_id.resize($color + 1, None);
            }
        };
    }

    // tried unsuccessfully &NodeIndexable::from_index(&graph, node)
    for node in nodes {
        if let Some(is_match) = filter_node(&graph.node_weight(node))? {
            let raw_edges = graph
                .edges_directed(node, petgraph::Direction::Outgoing);

            // Remove all edges that do not yield errors from color_fn
            let colors = raw_edges
                .map(|edge| {
                    let edge_weight = edge.weight();
                    color_edge(edge_weight)
                })
                .collect::<Result<Vec<Option<usize>>, _>>()?;

            // Remove null edges from color_fn
            let colors = colors.into_iter().flatten().collect::<Vec<usize>>();

            if colors.len() <= 2 && is_match {
                if colors.len() == 1 {
                    let c0 = colors[0];
                    ensure_vector_has_index!(pending_list, block_id, c0);
                    if let Some(c0_block_id) = block_id[c0] {
                        block_list[c0_block_id].push(graph.node_weight(node));
                    } else {
                        pending_list[c0].push(graph.node_weight(node));
                    }
                } else if colors.len() == 2 {
                    let c0 = colors[0];
                    let c1 = colors[1];
                    ensure_vector_has_index!(pending_list, block_id, c0);
                    ensure_vector_has_index!(pending_list, block_id, c1);

                    if block_id[c0].is_some()
                        && block_id[c1].is_some()
                        && block_id[c0] == block_id[c1]
                    {
                        block_list[block_id[c0].unwrap_or_default()]
                            .push(graph.node_weight(node));
                    } else {
                        let mut new_block: Vec<Option<&<G as Data>::NodeWeight>> =
                            Vec::with_capacity(pending_list[c0].len() + pending_list[c1].len() + 1);

                        // Clears pending lits and add to new block
                        new_block.append(&mut pending_list[c0]);
                        new_block.append(&mut pending_list[c1]);

                        new_block.push(graph.node_weight(node));

                        // Create new block, assign its id to color pair
                        block_id[c0] = Some(block_list.len());
                        block_id[c1] = Some(block_list.len());
                        block_list.push(new_block);
                    }
                }
            } else {
                for color in colors {
                    ensure_vector_has_index!(pending_list, block_id, color);
                    if let Some(color_block_id) = block_id[color] {
                        block_list[color_block_id].append(&mut pending_list[color]);
                    }
                    block_id[color] = None;
                    pending_list[color].clear();
                }
            }
        }
    }

    Ok(block_list)
}