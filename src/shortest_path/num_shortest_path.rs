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

use rustworkx_core::dictmap::*;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

use petgraph::graph::NodeIndex;
use petgraph::visit::{Bfs, NodeIndexable};
use petgraph::EdgeType;

use num_bigint::{BigUint, ToBigUint};

use crate::StablePyGraph;

pub fn num_shortest_paths_unweighted<Ty: EdgeType>(
    graph: &StablePyGraph<Ty>,
    source: usize,
) -> PyResult<DictMap<usize, BigUint>> {
    let mut out_map: Vec<BigUint> = vec![0.to_biguint().unwrap(); graph.node_bound()];
    let node_index = NodeIndex::new(source);
    if graph.node_weight(node_index).is_none() {
        return Err(PyIndexError::new_err(format!(
            "No node found for index {}",
            source
        )));
    }
    let mut bfs = Bfs::new(graph, node_index);
    let mut distance: Vec<Option<usize>> = vec![None; graph.node_bound()];
    distance[node_index.index()] = Some(0);
    out_map[source] = 1.to_biguint().unwrap();
    while let Some(current) = bfs.next(graph) {
        let dist_plus_one = distance[current.index()].unwrap_or_default() + 1;
        let count_current = out_map[current.index()].clone();
        for neighbor_index in graph.neighbors_directed(current, petgraph::Direction::Outgoing) {
            let neighbor: usize = neighbor_index.index();
            if distance[neighbor].is_none() {
                distance[neighbor] = Some(dist_plus_one);
                out_map[neighbor] = count_current.clone();
            } else if distance[neighbor] == Some(dist_plus_one) {
                out_map[neighbor] += &count_current;
            }
        }
    }

    // Do not count paths to source in output
    distance[source] = None;
    out_map[source] = 0.to_biguint().unwrap();

    // Return only nodes that are reachable in the graph
    Ok(out_map
        .into_iter()
        .zip(distance.iter())
        .enumerate()
        .filter_map(|(index, (count, dist))| {
            if dist.is_some() {
                Some((index, count))
            } else {
                None
            }
        })
        .collect())
}
