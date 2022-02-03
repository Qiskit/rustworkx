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

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use hashbrown::HashMap;

use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::visit::NodeCount;

use crate::digraph;

pub fn lexicographical_topological_sort(
    py: Python,
    dag: &digraph::PyDiGraph,
    key: PyObject,
) -> PyResult<Vec<NodeIndex>> {
    let key_callable = |a: &PyObject| -> PyResult<PyObject> {
        let res = key.call1(py, (a,))?;
        Ok(res.to_object(py))
    };
    // HashMap of node_index indegree
    let node_count = dag.node_count();
    let mut in_degree_map: HashMap<NodeIndex, usize> = HashMap::with_capacity(node_count);
    for node in dag.graph.node_indices() {
        in_degree_map.insert(node, dag.in_degree(node.index()));
    }

    #[derive(Clone, Eq, PartialEq)]
    struct State {
        key: String,
        node: NodeIndex,
    }

    impl Ord for State {
        fn cmp(&self, other: &State) -> Ordering {
            // Notice that the we flip the ordering on costs.
            // In case of a tie we compare positions - this step is necessary
            // to make implementations of `PartialEq` and `Ord` consistent.
            other
                .key
                .cmp(&self.key)
                .then_with(|| other.node.index().cmp(&self.node.index()))
        }
    }

    // `PartialOrd` needs to be implemented as well.
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &State) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut zero_indegree = BinaryHeap::with_capacity(node_count);
    for (node, degree) in in_degree_map.iter() {
        if *degree == 0 {
            let map_key_raw = key_callable(&dag.graph[*node])?;
            let map_key: String = map_key_raw.extract(py)?;
            zero_indegree.push(State {
                key: map_key,
                node: *node,
            });
        }
    }
    let mut out_list: Vec<NodeIndex> = Vec::with_capacity(node_count);
    let dir = petgraph::Direction::Outgoing;
    while let Some(State { node, .. }) = zero_indegree.pop() {
        let neighbors = dag.graph.neighbors_directed(node, dir);
        for child in neighbors {
            let child_degree = in_degree_map.get_mut(&child).unwrap();
            *child_degree -= 1;
            if *child_degree == 0 {
                let map_key_raw = key_callable(&dag.graph[child])?;
                let map_key: String = map_key_raw.extract(py)?;
                zero_indegree.push(State {
                    key: map_key,
                    node: child,
                });
                in_degree_map.remove(&child);
            }
        }
        out_list.push(node)
    }
    Ok(out_list)
}
