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
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::visit::{GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount};
use rayon::prelude::*;

use crate::dictmap::*;

/// Return the core number for each node in the graph.
///
/// A k-core is a maximal subgraph that contains nodes of degree k or more.
///
/// The function implicitly assumes that there are no parallel edges
/// or self loops. It may produce incorrect/unexpected results if the
/// input graph has self loops or parallel edges.
///
/// Arguments:
///
/// * `graph` - The graph in which to find the core numbers.
///
/// # Example
/// ```rust
/// use petgraph::prelude::*;
/// use rustworkx_core::connectivity::core_number;
///
/// let edge_list = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
/// let graph = DiGraph::<i32, i32>::from_edges(&edge_list);
/// let res: Vec<(usize, usize)> = core_number(graph)
///     .iter()
///     .map(|(k, v)| (k.index(), *v))
///     .collect();
/// assert_eq!(res, vec![(0, 3), (1, 3), (2, 3), (3, 3)]);
/// ```
pub fn core_number<G>(graph: G) -> DictMap<G::NodeId, usize>
where
    G: GraphBase + NodeCount,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId> + IntoNodeIdentifiers + IntoNeighborsDirected,
    G::NodeId: Eq + Hash + Send + Sync,
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
    node_vec.par_sort_by_key(|k| degree_map.get(k));

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
}

#[cfg(test)]
mod tests {
    use crate::connectivity::core_number;
    use petgraph::prelude::*;

    #[test]
    fn test_directed_empty() {
        let graph = DiGraph::<i32, i32>::new();
        let res: Vec<(usize, usize)> = core_number(graph)
            .iter()
            .map(|(k, v)| (k.index(), *v))
            .collect();
        assert_eq!(res, vec![]);
    }

    #[test]
    fn test_directed_all_0() {
        let mut graph = DiGraph::<i32, i32>::new();
        for _ in 0..4 {
            graph.add_node(0);
        }
        let res: Vec<(usize, usize)> = core_number(graph)
            .iter()
            .map(|(k, v)| (k.index(), *v))
            .collect();
        assert_eq!(res, vec![(0, 0), (1, 0), (2, 0), (3, 0)]);
    }

    #[test]
    fn test_directed_all_3() {
        let edge_list = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let graph = DiGraph::<i32, i32>::from_edges(edge_list);
        let res: Vec<(usize, usize)> = core_number(graph)
            .iter()
            .map(|(k, v)| (k.index(), *v))
            .collect();
        assert_eq!(res, vec![(0, 3), (1, 3), (2, 3), (3, 3)]);
    }

    #[test]
    fn test_directed_paper_example() {
        // This is the example graph in Figure 1 from Batagelj and
        // Zaversnik's paper titled An O(m) Algorithm for Cores
        // Decomposition of Networks, 2003,
        // http://arXiv.org/abs/cs/0310049.  With nodes labeled as
        // shown, the 3-core is given by nodes 0-7, the 2-core by nodes
        // 8-15, the 1-core by nodes 16-19 and node 20 is in the
        // 0-core.
        let edge_list = [
            (0, 2),
            (0, 3),
            (0, 5),
            (1, 4),
            (1, 6),
            (1, 7),
            (2, 3),
            (3, 5),
            (2, 5),
            (5, 6),
            (4, 6),
            (4, 7),
            (6, 7),
            (5, 8),
            (6, 8),
            (6, 9),
            (8, 9),
            (0, 10),
            (1, 10),
            (1, 11),
            (10, 11),
            (12, 13),
            (13, 15),
            (14, 15),
            (12, 14),
            (8, 19),
            (11, 16),
            (11, 17),
            (12, 18),
        ];
        let mut example_core = vec![];
        for i in 0..8 {
            example_core.push((i, 3));
        }
        for i in 8..16 {
            example_core.push((i, 2));
        }
        for i in 16..20 {
            example_core.push((i, 1));
        }
        example_core.push((20, 0));
        let mut graph = DiGraph::<i32, i32>::from_edges(edge_list);
        graph.add_node(0);
        let res: Vec<(usize, usize)> = core_number(graph)
            .iter()
            .map(|(k, v)| (k.index(), *v))
            .collect();
        assert_eq!(res, example_core);
    }
}
