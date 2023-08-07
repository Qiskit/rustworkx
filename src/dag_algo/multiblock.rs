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

use std::mem;

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use pyo3::prelude::*;
use pyo3::Python;

use super::lexicographical_sort::lexicographical_topological_sort;
use crate::digraph;

/// DSU function for finding root of set of items
/// If my parent is myself, I am the root. Otherwise we recursively
/// find the root for my parent. After that, we assign my parent to be
/// my root, saving recursion in the future.
fn find_set(
    index: usize,
    parent: &mut HashMap<usize, usize>,
    groups: &mut HashMap<usize, Vec<usize>>,
    op_groups: &mut HashMap<usize, Vec<usize>>,
) -> usize {
    let mut update_index: Vec<usize> = Vec::new();
    let mut index_iter = index;
    while parent.get(&index_iter) != Some(&index_iter) {
        if parent.get(&index_iter).is_none() {
            parent.insert(index_iter, index_iter);
            groups.insert(index_iter, vec![index_iter]);
            op_groups.insert(index_iter, Vec::new());
        }
        if parent.get(&index_iter) != Some(&index_iter) {
            update_index.push(index_iter);
        }
        index_iter = parent[&index_iter];
    }
    for index in update_index {
        parent.insert(index, index_iter);
    }
    parent[&index_iter]
}

fn combine_sets(op_groups: &mut HashMap<usize, Vec<usize>>, set_a: usize, set_b: usize) {
    let mut other_groups = op_groups.get_mut(&set_b).unwrap().clone();
    op_groups.get_mut(&set_a).unwrap().append(&mut other_groups);
    op_groups.get_mut(&set_b).unwrap().clear();
}

/// DSU function for unioning two sets together
/// Find the roots of each set. Then assign one to have the other
/// as its parent, thus liking the sets.
/// Merges smaller set into larger set in order to have better runtime
fn union_set(
    set_a_ind: usize,
    set_b_ind: usize,
    parent: &mut HashMap<usize, usize>,
    groups: &mut HashMap<usize, Vec<usize>>,
    op_groups: &mut HashMap<usize, Vec<usize>>,
) {
    let mut set_a = find_set(set_a_ind, parent, groups, op_groups);
    let mut set_b = find_set(set_b_ind, parent, groups, op_groups);

    if set_a == set_b {
        return;
    }
    if op_groups[&set_a].len() < op_groups[&set_b].len() {
        mem::swap(&mut set_a, &mut set_b);
    }
    parent.insert(set_b, set_a);
    combine_sets(op_groups, set_a, set_b);
    combine_sets(groups, set_a, set_b)
}

fn update_set(
    group_index: usize,
    parent: &mut HashMap<usize, usize>,
    groups: &mut HashMap<usize, Vec<usize>>,
    op_groups: &mut HashMap<usize, Vec<usize>>,
    block_list: &mut Vec<Vec<usize>>,
) {
    if !op_groups[&group_index].is_empty() {
        block_list.push(op_groups[&group_index].to_vec());
    }
    let cur_set: HashSet<usize> = groups[&group_index].iter().copied().collect();
    for v in cur_set {
        parent.insert(v, v);
        groups.insert(v, vec![v]);
        op_groups.insert(v, Vec::new());
    }
}

pub fn collect_multi_blocks(
    py: Python,
    dag: &digraph::PyDiGraph,
    block_size: usize,
    key: PyObject,
    group_fn: PyObject,
    filter_fn: PyObject,
) -> PyResult<Vec<Vec<usize>>> {
    let mut block_list: Vec<Vec<usize>> = Vec::new();

    let mut parent: HashMap<usize, usize> = HashMap::new();
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut op_groups: HashMap<usize, Vec<usize>> = HashMap::new();

    //    let sort_nodes = lexicographical_topological_sort(py, dag, key)?;
    for node in lexicographical_topological_sort(py, dag, key)? {
        let filter_res = filter_fn.call1(py, (&dag.graph[node],))?;
        let can_process_option: Option<bool> = filter_res.extract(py)?;
        if can_process_option.is_none() {
            continue;
        }
        let raw_cur_nodes = group_fn.call1(py, (&dag.graph[node],))?;
        let cur_groups: HashSet<usize> = raw_cur_nodes.extract(py)?;
        let can_process: bool = can_process_option.unwrap();
        let mut makes_too_big: bool = false;

        if can_process {
            let mut tops: HashSet<usize> = HashSet::new();
            for group in &cur_groups {
                tops.insert(find_set(*group, &mut parent, &mut groups, &mut op_groups));
            }
            let mut tot_size = 0;
            for group in tops {
                tot_size += groups[&group].len();
            }
            if tot_size > block_size {
                makes_too_big = true;
            }
        }
        if !can_process {
            // resolve the case where we cannot process this node
            for group_entry in &cur_groups {
                let group = find_set(*group_entry, &mut parent, &mut groups, &mut op_groups);
                if op_groups[&group].is_empty() {
                    continue;
                }
                update_set(
                    group,
                    &mut parent,
                    &mut groups,
                    &mut op_groups,
                    &mut block_list,
                );
            }
        }
        if makes_too_big {
            // Adding in all of the new groups would make the group too big
            // we must block off sub portions of the groups until the new
            // group would no longer be too big
            let mut savings: HashMap<usize, usize> = HashMap::new();
            let mut tot_size = 0;
            for group in &cur_groups {
                let top = find_set(*group, &mut parent, &mut groups, &mut op_groups);
                if !savings.contains_key(&top) {
                    savings.insert(top, groups[&top].len() - 1);
                    tot_size += groups[&top].len();
                } else {
                    *savings.get_mut(&top).unwrap() -= 1;
                }
            }
            let mut savings_list: Vec<(usize, usize)> = savings
                .into_iter()
                .map(|(item, value)| (value, item))
                .collect();
            savings_list.par_sort_unstable();
            savings_list.reverse();
            let mut savings_need = tot_size - block_size;
            for item in savings_list {
                // remove groups until the size created would be acceptable
                // start with blocking out the group that would decrease
                // the new size the most. This heuristic for which blocks we
                // create does not necessarily give the optimal blocking.
                // Other heuristics may be worth considering
                if savings_need > 0 {
                    savings_need -= item.0;
                    let item_index = item.1;
                    update_set(
                        item_index,
                        &mut parent,
                        &mut groups,
                        &mut op_groups,
                        &mut block_list,
                    );
                }
            }
        }
        if can_process {
            // if the operation is a processable, either skip it if it is too
            // large group up all of the qubits involved in the gate
            if cur_groups.len() > block_size {
                // nodes operating on more groups than block_size cannot be a
                // part of any block and thus we skip them here/
                // We have already finalized the blocks involving the node's
                // groups in the above maxkes_to_big block
                continue; // unable to be part of a group
            }
            let mut prev: Option<usize> = None;
            for group in cur_groups {
                let index = group;
                if let Some(value) = prev {
                    union_set(value, index, &mut parent, &mut groups, &mut op_groups);
                }
                prev = Some(index);
            }
            if let Some(value) = prev {
                let found_set = find_set(value, &mut parent, &mut groups, &mut op_groups);
                op_groups.get_mut(&found_set).unwrap().push(node.index());
            }
        }
    }

    for (index, parent) in parent.iter() {
        let parent_index = parent;
        if parent_index == index && !op_groups[parent].is_empty() {
            block_list.push(op_groups[index].to_vec());
        }
    }
    Ok(block_list)
}
