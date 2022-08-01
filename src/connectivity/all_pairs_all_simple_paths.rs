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

use rayon::prelude::*;

use petgraph::graph::NodeIndex;
use petgraph::EdgeType;
use rustworkx_core::connectivity::all_simple_paths_multiple_targets;

use crate::iterators::{AllPairsMultiplePathMapping, MultiplePathMapping};
use crate::StablePyGraph;
use rustworkx_core::dictmap::*;

pub fn all_pairs_all_simple_paths<Ty: EdgeType + Sync>(
    graph: &StablePyGraph<Ty>,
    min_depth: Option<usize>,
    cutoff: Option<usize>,
) -> AllPairsMultiplePathMapping {
    if graph.node_count() == 0 {
        return AllPairsMultiplePathMapping {
            paths: DictMap::new(),
        };
    } else if graph.edge_count() == 0 {
        return AllPairsMultiplePathMapping {
            paths: graph
                .node_indices()
                .map(|i| {
                    (
                        i.index(),
                        MultiplePathMapping {
                            paths: DictMap::new(),
                        },
                    )
                })
                .collect(),
        };
    }
    let intermediate_min: usize = match min_depth {
        Some(depth) => depth - 2,
        None => 0,
    };
    let intermediate_cutoff = cutoff.map(|depth| depth - 2);
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    let node_index_set = node_indices.iter().copied().collect();
    AllPairsMultiplePathMapping {
        paths: node_indices
            .par_iter()
            .map(|u| {
                let out_paths = MultiplePathMapping {
                    paths: all_simple_paths_multiple_targets(
                        graph,
                        *u,
                        &node_index_set,
                        intermediate_min,
                        intermediate_cutoff,
                    )
                    .iter()
                    .map(|(v, path)| {
                        let output: Vec<Vec<usize>> = path
                            .iter()
                            .map(|v| v.iter().map(|i| i.index()).collect())
                            .collect();
                        (v.index(), output)
                    })
                    .collect(),
                };
                (u.index(), out_paths)
            })
            .collect(),
    }
}
