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

// This module was forked from petgraph:
//
// https://github.com/petgraph/petgraph/blob/9ff688872b467d3e1b5adef19f5c52f519d3279c/src/algo/simple_paths.rs
//
// to add support for returning all simple paths to a list of targets instead
// of just between a single node pair.

use std::fmt::Debug;

use crate::dictmap::*;
use hashbrown::{HashMap, HashSet};
use petgraph::visit::{GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount};
use petgraph::Direction::{Incoming, Outgoing};
use std::hash::Hash;

/// Return the first cycle encountered during DFS of a given directed graph.
/// Empty list is returned if no cycle is found.
///
/// Arguments:
///
/// * `graph` - The directed graph in which to find the first cycle.
/// * `source` - Optional node index for starting the search. If not specified,
///     an arbitrary node is chosen to start the search.
///
/// # Example
/// ```rust
/// use petgraph::prelude::*;
/// use rustworkx_core::connectivity::find_cycle;
///
/// let edge_list = vec![
///     (0, 1),
///     (3, 0),
///     (0, 5),
///     (8, 0),
///     (1, 2),
///     (1, 6),
///     (2, 3),
///     (3, 4),
///     (4, 5),
///     (6, 7),
///     (7, 8),
///     (8, 9),
/// ];
/// let graph = DiGraph::<i32, i32>::from_edges(&edge_list);
/// let mut res: Vec<(usize, usize)> = find_cycle(&graph, Some(NodeIndex::new(0)))
///     .iter()
///     .map(|(s, t)| (s.index(), t.index()))
///     .collect();
/// assert_eq!(res, [(0, 1), (1, 2), (2, 3), (3, 0)]);
/// ```
pub fn core_number<G>(graph: G) -> DictMap<G::NodeId, usize>
where
    G: GraphBase,
    G: NodeCount,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoNodeIdentifiers + IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
    <G as GraphBase>::NodeId: Debug,
{
    let node_num = graph.node_count();
    if node_num == 0 {
        return DictMap::new();
    }

    let mut cores: DictMap<G::NodeId, usize> = DictMap::with_capacity(node_num);
    let mut node_vec: Vec<G::NodeId> = graph.node_identifiers().collect();
    let mut degree_map: HashMap<G::NodeId, usize> = HashMap::with_capacity(node_num);
    let mut nbrs: HashMap<G::NodeId, HashSet<G::NodeId>> = HashMap::with_capacity(node_num);
    let mut node_pos: HashMap<G::NodeId, usize> = HashMap::with_capacity(node_num);

    for k in node_vec.iter() {
        let k_nbrs: HashSet<G::NodeId> = graph
            .neighbors_directed(*k, Incoming)
            .chain(graph.neighbors_directed(*k, Outgoing))
            .collect();
        let k_deg = k_nbrs.len();

        nbrs.insert(*k, k_nbrs);
        cores.insert(*k, k_deg);
        degree_map.insert(*k, k_deg);
    }
    node_vec.sort_by_key(|k| degree_map.get(k));

    let mut bin_boundaries: Vec<usize> =
        Vec::with_capacity(degree_map[&node_vec[node_num - 1]] + 1);
    bin_boundaries.push(0);
    let mut curr_degree = 0;
    for (i, v) in node_vec.iter().enumerate() {
        node_pos.insert(*v, i);
        let v_degree = degree_map[v];
        if v_degree > curr_degree {
            for _ in 0..v_degree - curr_degree {
                bin_boundaries.push(i);
            }
            curr_degree = v_degree;
        }
    }

    for v_ind in 0..node_vec.len() {
        let v = node_vec[v_ind];
        let v_nbrs = nbrs[&v].clone();
        for u in v_nbrs {
            if cores[&u] > cores[&v] {
                nbrs.get_mut(&u).unwrap().remove(&v);
                let pos = node_pos[&u];
                let bin_start = bin_boundaries[cores[&u]];
                *node_pos.get_mut(&u).unwrap() = bin_start;
                *node_pos.get_mut(&node_vec[bin_start]).unwrap() = pos;
                node_vec.swap(bin_start, pos);
                bin_boundaries[cores[&u]] += 1;
                *cores.get_mut(&u).unwrap() -= 1;
            }
        }
    }
    cores
    // let out_dict = PyDict::new(py);
    // for (v_index, core) in cores {
    //     out_dict.set_item(v_index.index(), core)?;
    // }
    // Ok(out_dict.into())
}

#[cfg(test)]
mod tests {
    use crate::connectivity::find_cycle;
    use petgraph::prelude::*;

    #[test]
    fn test_find_cycle_source() {
        let edge_list = vec![
            (0, 1),
            (3, 0),
            (0, 5),
            (8, 0),
            (1, 2),
            (1, 6),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
        ];
        let graph = DiGraph::<i32, i32>::from_edges(&edge_list);
        let mut res: Vec<(usize, usize)> = find_cycle(&graph, Some(NodeIndex::new(0)))
            .iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect();
        assert_eq!(res, [(0, 1), (1, 2), (2, 3), (3, 0)]);
        res = find_cycle(&graph, Some(NodeIndex::new(1)))
            .iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect();
        assert_eq!(res, [(1, 2), (2, 3), (3, 0), (0, 1)]);
        res = find_cycle(&graph, Some(NodeIndex::new(5)))
            .iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect();
        assert_eq!(res, []);
    }

    #[test]
    fn test_self_loop() {
        let edge_list = vec![
            (0, 1),
            (3, 0),
            (0, 5),
            (8, 0),
            (1, 2),
            (1, 6),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
        ];
        let mut graph = DiGraph::<i32, i32>::from_edges(&edge_list);
        graph.add_edge(NodeIndex::new(1), NodeIndex::new(1), 0);
        let res: Vec<(usize, usize)> = find_cycle(&graph, Some(NodeIndex::new(0)))
            .iter()
            .map(|(s, t)| (s.index(), t.index()))
            .collect();
        assert_eq!(res, [(1, 1)]);
    }
}
