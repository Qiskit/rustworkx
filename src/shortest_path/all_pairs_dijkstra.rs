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

use hashbrown::HashMap;

use rustworkx_core::dictmap::*;
use rustworkx_core::shortest_path::dijkstra;

use std::sync::RwLock;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::Python;

use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::EdgeIndexable;
use petgraph::EdgeType;

use rayon::prelude::*;

use crate::iterators::{
    AllPairsPathLengthMapping, AllPairsPathMapping, PathLengthMapping, PathMapping,
};
use crate::{CostFn, StablePyGraph};

pub fn all_pairs_dijkstra_path_lengths<Ty: EdgeType + Sync>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    edge_cost_fn: PyObject,
) -> PyResult<AllPairsPathLengthMapping> {
    if graph.node_count() == 0 {
        return Ok(AllPairsPathLengthMapping {
            path_lengths: DictMap::new(),
        });
    } else if graph.edge_count() == 0 {
        return Ok(AllPairsPathLengthMapping {
            path_lengths: graph
                .node_indices()
                .map(|i| {
                    (
                        i.index(),
                        PathLengthMapping {
                            path_lengths: DictMap::new(),
                        },
                    )
                })
                .collect(),
        });
    }
    let edge_cost_callable = CostFn::from(edge_cost_fn);
    let mut edge_weights: Vec<Option<f64>> = Vec::with_capacity(graph.edge_bound());
    for index in 0..=graph.edge_bound() {
        let raw_weight = graph.edge_weight(EdgeIndex::new(index));
        match raw_weight {
            Some(weight) => edge_weights.push(Some(edge_cost_callable.call(py, weight)?)),
            None => edge_weights.push(None),
        };
    }
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    let out_map: DictMap<usize, PathLengthMapping> = node_indices
        .into_par_iter()
        .map(|x| {
            let path_lenghts: PyResult<Vec<Option<f64>>> =
                dijkstra(graph, x, None, |e| edge_cost(e.id()), None);
            let out_map = PathLengthMapping {
                path_lengths: path_lenghts
                    .unwrap()
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, opt_cost)| {
                        if index != x.index() {
                            opt_cost.map(|cost| (index, cost))
                        } else {
                            None
                        }
                    })
                    .collect(),
            };
            (x.index(), out_map)
        })
        .collect();
    Ok(AllPairsPathLengthMapping {
        path_lengths: out_map,
    })
}

pub fn all_pairs_dijkstra_shortest_paths<Ty: EdgeType + Sync>(
    py: Python,
    graph: &StablePyGraph<Ty>,
    edge_cost_fn: PyObject,
    distances: Option<&mut HashMap<usize, DictMap<NodeIndex, f64>>>,
) -> PyResult<AllPairsPathMapping> {
    if graph.node_count() == 0 {
        return Ok(AllPairsPathMapping {
            paths: DictMap::new(),
        });
    } else if graph.edge_count() == 0 {
        return Ok(AllPairsPathMapping {
            paths: graph
                .node_indices()
                .map(|i| {
                    (
                        i.index(),
                        PathMapping {
                            paths: DictMap::new(),
                        },
                    )
                })
                .collect(),
        });
    }
    let edge_cost_callable = CostFn::from(edge_cost_fn);
    let mut edge_weights: Vec<Option<f64>> = Vec::with_capacity(graph.edge_bound());
    for index in 0..=graph.edge_bound() {
        let raw_weight = graph.edge_weight(EdgeIndex::new(index));
        match raw_weight {
            Some(weight) => edge_weights.push(Some(edge_cost_callable.call(py, weight)?)),
            None => edge_weights.push(None),
        };
    }
    let edge_cost = |e: EdgeIndex| -> PyResult<f64> {
        match edge_weights[e.index()] {
            Some(weight) => Ok(weight),
            None => Err(PyIndexError::new_err("No edge found for index")),
        }
    };
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    let temp_distances: RwLock<HashMap<usize, DictMap<NodeIndex, f64>>> = if distances.is_some() {
        RwLock::new(HashMap::with_capacity(graph.node_count()))
    } else {
        // Avoid extra allocation if HashMap isn't used
        RwLock::new(HashMap::new())
    };
    let out_map = AllPairsPathMapping {
        paths: node_indices
            .into_par_iter()
            .map(|x| {
                let mut paths: DictMap<NodeIndex, Vec<NodeIndex>> =
                    DictMap::with_capacity(graph.node_count());
                let distance =
                    dijkstra(graph, x, None, |e| edge_cost(e.id()), Some(&mut paths)).unwrap();
                if distances.is_some() {
                    temp_distances.write().unwrap().insert(x.index(), distance);
                }
                let index = x.index();
                let out_paths = PathMapping {
                    paths: paths
                        .iter()
                        .filter_map(|path_mapping| {
                            let path_index = path_mapping.0.index();
                            if index != path_index {
                                Some((
                                    path_index,
                                    path_mapping.1.iter().map(|x| x.index()).collect(),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect(),
                };
                (index, out_paths)
            })
            .collect(),
    };
    if let Some(x) = distances {
        *x = temp_distances.read().unwrap().clone()
    };
    Ok(out_map)
}
