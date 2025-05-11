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

use ndarray::prelude::*;
use petgraph::EdgeType;

use crate::NodesRemoved;
use crate::StablePyGraph;

use rustworkx_core::shortest_path;

pub fn compute_distance_matrix<Ty: EdgeType + Sync>(
    graph: &StablePyGraph<Ty>,
    parallel_threshold: usize,
    as_undirected: bool,
    null_value: f64,
) -> Array2<f64> {
    if graph.nodes_removed() {
        shortest_path::distance_matrix_compacted(
            graph,
            parallel_threshold,
            as_undirected,
            null_value,
        )
    } else {
        shortest_path::distance_matrix(graph, parallel_threshold, as_undirected, null_value)
    }
}
