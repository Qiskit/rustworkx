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

#![allow(clippy::float_cmp)]

use crate::graph;

use ahash::RandomState;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use std::cmp::Reverse;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::NodeCount;

use rayon::prelude::*;

/// Color a PyGraph using a largest_first strategy greedy graph coloring.
///
/// :param PyGraph: The input PyGraph object to color
///
/// :returns: A dictionary where keys are node indices and the value is
///     the color
/// :rtype: dict
#[pyfunction]
#[pyo3(text_signature = "(graph, /)")]
fn graph_greedy_color(
    py: Python,
    graph: &graph::PyGraph,
) -> PyResult<PyObject> {
    let mut colors: IndexMap<usize, usize, RandomState> =
        IndexMap::with_hasher(RandomState::default());
    let mut node_vec: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let mut sort_map: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(graph.node_count());
    for k in node_vec.iter() {
        sort_map.insert(*k, graph.graph.edges(*k).count());
    }
    node_vec.par_sort_by_key(|k| Reverse(sort_map.get(k)));
    for u_index in node_vec {
        let mut neighbor_colors: HashSet<usize> = HashSet::new();
        for edge in graph.graph.edges(u_index) {
            let target = edge.target().index();
            let existing_color = match colors.get(&target) {
                Some(node) => node,
                None => continue,
            };
            neighbor_colors.insert(*existing_color);
        }
        let mut count: usize = 0;
        loop {
            if !neighbor_colors.contains(&count) {
                break;
            }
            count += 1;
        }
        colors.insert(u_index.index(), count);
    }
    let out_dict = PyDict::new(py);
    for (index, color) in colors {
        out_dict.set_item(index, color)?;
    }
    Ok(out_dict.into())
}
