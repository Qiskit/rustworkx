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

use hashbrown::{HashMap, HashSet};

use ndarray::prelude::*;
use petgraph::prelude::*;
use petgraph::EdgeType;
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn compute_distance_matrix<Ty: EdgeType + Sync>(
    graph: &StableGraph<PyObject, PyObject, Ty>,
    parallel_threshold: usize,
    as_undirected: bool,
    null_value: f64,
) -> Array2<f64> {
    let n = graph.node_count();
    let mut matrix = Array2::<f64>::from_elem((n, n), null_value);
    let bfs_traversal = |index: usize, mut row: ArrayViewMut1<f64>| {
        let mut seen: HashMap<NodeIndex, usize> = HashMap::with_capacity(n);
        let start_index = NodeIndex::new(index);
        let mut level = 0;
        let mut next_level: HashSet<NodeIndex> = HashSet::new();
        next_level.insert(start_index);
        while !next_level.is_empty() {
            let this_level = next_level;
            next_level = HashSet::new();
            let mut found: Vec<NodeIndex> = Vec::new();
            for v in this_level {
                if !seen.contains_key(&v) {
                    seen.insert(v, level);
                    found.push(v);
                    row[[v.index()]] = level as f64;
                }
            }
            if seen.len() == n {
                return;
            }
            for node in found {
                if graph.is_directed() {
                    for v in graph
                        .neighbors_directed(node, petgraph::Direction::Outgoing)
                    {
                        next_level.insert(v);
                    }
                    if as_undirected {
                        for v in graph.neighbors_directed(
                            node,
                            petgraph::Direction::Incoming,
                        ) {
                            next_level.insert(v);
                        }
                    }
                } else {
                    for v in graph.neighbors(node) {
                        next_level.insert(v);
                    }
                }
            }
            level += 1
        }
    };
    if n < parallel_threshold {
        matrix
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    } else {
        // Parallelize by row and iterate from each row index in BFS order
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(index, row)| bfs_traversal(index, row));
    }
    matrix
}
