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

use crate::graph;
use crate::iterators::EdgeList;

use hashbrown::HashMap;
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use rustworkx_core::token_swapper;

/// This module performs an approximately optimal Token Swapping algorithm
/// Supports partial mappings (i.e. not-permutations) for graphs with missing tokens.
///
/// Based on the paper: Approximation and Hardness for Token Swapping by Miltzow et al. (2016)
/// ArXiV: https://arxiv.org/abs/1602.05150
/// and generalization based on our own work.
///
/// The inputs are a partial ``mapping`` to be implemented in swaps, and the number of ``trials``
/// to perform the mapping. It's minimized over the trials.
///
/// It returns a list of tuples representing the swaps to perform.
///
/// :param PyGraph graph: The input graph
/// :param dict[int: int] mapping: Map of (node, token)
/// :param int trials: The number of trials to run
/// :param int seed: The random seed to be used in producing random ints for selecting
///      which nodes to process next
///
/// :returns: A list of tuples which are the swaps to be applied to the mapping to rearrange
///      the tokens.
/// :rtype: EdgeList
#[pyfunction]
#[pyo3(text_signature = "(graph, mapping, /, trials=None, seed=None)")]
pub fn graph_token_swapper(
    graph: &graph::PyGraph,
    mapping: HashMap<usize, usize>,
    trials: Option<usize>,
    seed: Option<u64>,
) -> EdgeList {
    let map: HashMap<NodeIndex, NodeIndex> = mapping
        .iter()
        .map(|(s, t)| (NodeIndex::new(*s), NodeIndex::new(*t)))
        .collect();
    let swaps = token_swapper::token_swapper(&graph.graph, map, trials, seed);
    EdgeList {
        edges: swaps
            .into_iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect(),
    }
}
